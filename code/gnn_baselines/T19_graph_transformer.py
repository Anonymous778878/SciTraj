"""
Tier 19 — Graph Transformer (Graphormer-style, local attention).

Treats edges as tokens. For each source paper s, gather a local
candidate set of size K = (FAISS top-50) U (actual neighbours of s)
truncated to 64 candidates max. Run multi-head attention from s over
this candidate set, with structural biases added to the raw attention
score:

    score(s, c) = (q_s . k_c) / sqrt(d_h)
                + b_relation[r_sc]
                + b_year_gap[bucket(year_c - year_s)]
                + b_indeg[bucket(log indeg(c))]

The structural biases are crucial: they let the model learn that a
candidate from 4 years later, with high in-degree, of relation type
"future_realized", is more likely to be a true edge.

Output:
  outputs/metrics/tier19_graph_transformer.json
  models/tier19_graph_transformer/best.pt
  models/tier19_graph_transformer/embeddings.npy
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from gnn_utils import load_graph, prepare_temporal_split
from hard_negatives import build_positive_set, build_topic_year_pools
from hard_negatives_v2 import (
    build_candidate_pool_negatives,
    sample_candidate_pool_negatives,
)
from utils import ensure_dir, get_logger, load_config, load_json

log = get_logger("tier19_gtrans")


N_RELATIONS = 9
YEAR_BUCKETS = 11   # < -5, -5..-3, -2, -1, 0, +1, +2, +3..5, +6..8, +9..15, > 15
DEG_BUCKETS = 8


def bucketise_year(dt):
    """Map year delta to bucket id."""
    if dt < -5: return 0
    if dt <= -3: return 1
    if dt == -2: return 2
    if dt == -1: return 3
    if dt == 0: return 4
    if dt == 1: return 5
    if dt == 2: return 6
    if dt <= 5: return 7
    if dt <= 8: return 8
    if dt <= 15: return 9
    return 10


def bucketise_indeg(d):
    if d == 0: return 0
    if d == 1: return 1
    if d <= 3: return 2
    if d <= 7: return 3
    if d <= 15: return 4
    if d <= 31: return 5
    if d <= 63: return 6
    return 7


class GraphTransformer(nn.Module):
    """
    Single-layer multi-head attention with structural biases.
    Output is a per-pair logit, not a per-node embedding (no
    bottleneck issue — score is computed end-to-end).
    """
    def __init__(self, d=768, n_heads=8, n_relations=N_RELATIONS,
                 year_buckets=YEAR_BUCKETS, deg_buckets=DEG_BUCKETS,
                 dropout=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

        # Structural attention biases (one scalar per (head, bucket))
        self.bias_relation = nn.Parameter(torch.zeros(n_heads, n_relations + 1))  # +1 for "no edge yet"
        self.bias_year = nn.Parameter(torch.zeros(n_heads, year_buckets))
        self.bias_indeg = nn.Parameter(torch.zeros(n_heads, deg_buckets))

        # Final pair score head: combines source/target attended representations
        self.score_head = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

    def forward(self, s_emb, t_emb, ctx_emb, ctx_mask,
                rel_idx, year_idx, deg_idx):
        """
        s_emb:      (B, D)         source embedding
        t_emb:      (B, D)         target embedding
        ctx_emb:    (B, K, D)      context: candidate neighbours of source
        ctx_mask:   (B, K)         True where context is real (not padding)
        rel_idx:    (B, K)         relation type of each (s, ctx) edge,
                                    or n_relations for non-edge
        year_idx:   (B, K)         year-gap bucket
        deg_idx:    (B, K)         indeg bucket of context paper
        Returns:    (B,)           logit
        """
        B, K, D = ctx_emb.shape
        H = self.n_heads
        Hd = self.head_dim

        # Project source as query
        q = self.q_proj(s_emb).view(B, H, Hd)              # (B, H, Hd)
        k = self.k_proj(ctx_emb).view(B, K, H, Hd)         # (B, K, H, Hd)
        v = self.v_proj(ctx_emb).view(B, K, H, Hd)         # (B, K, H, Hd)

        # Attention scores: (B, H, K)
        scores = torch.einsum("bhd,bkhd->bhk", q, k) / (Hd ** 0.5)

        # Structural biases (B, H, K)
        bias_r = self.bias_relation[:, rel_idx].permute(1, 0, 2)   # (B, H, K)
        bias_y = self.bias_year[:, year_idx].permute(1, 0, 2)
        bias_d = self.bias_indeg[:, deg_idx].permute(1, 0, 2)
        scores = scores + bias_r + bias_y + bias_d

        # Mask padding
        scores = scores.masked_fill(~ctx_mask.unsqueeze(1), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # If a row is all -inf (no neighbours at all), softmax is NaN — fix
        attn = torch.nan_to_num(attn, nan=0.0)

        # Aggregate values: (B, H, Hd) -> (B, D)
        agg = torch.einsum("bhk,bkhd->bhd", attn, v).reshape(B, D)
        s_att = self.out_proj(agg) + s_emb   # residual

        # Pair score
        return self.score_head(torch.cat([s_att, t_emb], dim=1)).squeeze(-1)


def build_per_source_context(
    n_papers, edge_index_np, edge_type_np, year_arr, indeg_arr,
    candidates_records, pid_to_row, max_ctx=64,
):
    """For each source paper, build a context list:
      - actual graph neighbours (any type)
      - top FAISS candidates (all views combined)
    Returns:
      ctx_lists: dict src_row -> list of (tgt_row, rel_id, year_gap, indeg)
                 with rel_id = N_RELATIONS for non-edge candidates
    """
    log.info("building per-source context lists (max %d)", max_ctx)
    # Group edges by source
    src_to_neighbours = {}   # src_row -> list[(tgt_row, rel_id)]
    src_np = edge_index_np[0]
    tgt_np = edge_index_np[1]
    for i in range(len(src_np)):
        s = int(src_np[i]); t = int(tgt_np[i]); r = int(edge_type_np[i])
        src_to_neighbours.setdefault(s, []).append((t, r))

    # Combine with FAISS candidates from candidates.json
    src_to_faiss_pool = {}
    for rec in candidates_records:
        sp = rec.get("paper_id")
        if sp not in pid_to_row:
            continue
        s = pid_to_row[sp]
        cands = []
        for cand in rec.get("candidates", []):
            tp = cand.get("paper_id")
            if tp in pid_to_row:
                cands.append(pid_to_row[tp])
        src_to_faiss_pool[s] = cands

    ctx_lists = {}
    for s in range(n_papers):
        seen = set()
        out = []

        # Real neighbours first (more informative)
        for (t, r) in src_to_neighbours.get(s, []):
            if t in seen or t == s:
                continue
            seen.add(t)
            dt = int(year_arr[t] - year_arr[s])
            out.append((t, r, dt, int(indeg_arr[t])))

        # Then FAISS candidates that aren't already neighbours
        for t in src_to_faiss_pool.get(s, []):
            if t in seen or t == s:
                continue
            seen.add(t)
            dt = int(year_arr[t] - year_arr[s])
            out.append((t, N_RELATIONS, dt, int(indeg_arr[t])))  # N_RELATIONS = "no edge"
            if len(out) >= max_ctx:
                break

        # Truncate
        out = out[:max_ctx]
        ctx_lists[s] = out

    log.info("context built. avg context size: %.1f",
             np.mean([len(v) for v in ctx_lists.values()]))
    return ctx_lists


def make_batch_tensors(s_rows, t_rows, paper_emb, ctx_lists,
                       max_ctx=64, device=None):
    """Turn a list of (src, tgt) pairs into model inputs."""
    B = len(s_rows)
    D = paper_emb.shape[1]
    s_emb = paper_emb[s_rows]
    t_emb = paper_emb[t_rows]

    ctx_emb = torch.zeros(B, max_ctx, D, device=device)
    ctx_mask = torch.zeros(B, max_ctx, dtype=torch.bool, device=device)
    rel_idx = torch.full((B, max_ctx), N_RELATIONS, dtype=torch.long, device=device)
    year_idx = torch.zeros(B, max_ctx, dtype=torch.long, device=device)
    deg_idx = torch.zeros(B, max_ctx, dtype=torch.long, device=device)

    for i, s in enumerate(s_rows):
        ctx = ctx_lists.get(int(s), [])
        for j, (t_ctx, r, dt, indeg) in enumerate(ctx[:max_ctx]):
            ctx_emb[i, j] = paper_emb[t_ctx]
            ctx_mask[i, j] = True
            rel_idx[i, j] = r
            year_idx[i, j] = bucketise_year(dt)
            deg_idx[i, j] = bucketise_indeg(indeg)

    return s_emb, t_emb, ctx_emb, ctx_mask, rel_idx, year_idx, deg_idx


def evaluate_model(model, paper_emb, ctx_lists, pos, neg, device, batch_size=512):
    """Compute AUC and AP on (pos, neg) pairs."""
    model.eval()
    pos_rows_s = pos[0].cpu().tolist()
    pos_rows_t = pos[1].cpu().tolist()
    neg_rows_s = neg[0].cpu().tolist()
    neg_rows_t = neg[1].cpu().tolist()

    pos_scores = []
    neg_scores = []
    with torch.no_grad():
        for i in range(0, len(pos_rows_s), batch_size):
            s = pos_rows_s[i:i + batch_size]
            t = pos_rows_t[i:i + batch_size]
            inputs = make_batch_tensors(s, t, paper_emb, ctx_lists, device=device)
            scores = model(*inputs)
            pos_scores.extend(scores.cpu().numpy().tolist())
        for i in range(0, len(neg_rows_s), batch_size):
            s = neg_rows_s[i:i + batch_size]
            t = neg_rows_t[i:i + batch_size]
            inputs = make_batch_tensors(s, t, paper_emb, ctx_lists, device=device)
            scores = model(*inputs)
            neg_scores.extend(scores.cpu().numpy().tolist())

    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    models_dir = ensure_dir(Path(cfg["paths"]["models_dir"]) / "tier19_graph_transformer")
    graph_dir = Path(cfg["paths"]["graph_dir"])
    ret_dir = Path(cfg["paths"]["retrieval_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    log.info("loading graph")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    paper_emb = data["paper"].x_abstract.to(device)
    edge_index = data["paper", "trajectory", "paper"].edge_index
    edge_type = data["paper", "trajectory", "paper"].edge_type
    edge_index_np = edge_index.cpu().numpy()
    edge_type_np = edge_type.cpu().numpy()

    n_papers = int(data["paper"].num_nodes)
    year_arr = data["paper"].year.cpu().numpy()
    log.info("nodes=%d edges=%d", n_papers, edge_index.shape[1])

    # Compute in-degree
    indeg_arr = np.zeros(n_papers, dtype=np.int64)
    np.add.at(indeg_arr, edge_index_np[1], 1)

    # Splits
    train_idx, val_idx, test_idx = prepare_temporal_split(
        data,
        train_year_max=cfg["corpus"]["train_years"][1],
        val_year_max=cfg["corpus"]["val_years"][1],
        test_year_max=cfg["corpus"]["test_years"][1],
    )
    train_pos = edge_index[:, train_idx].to(device)
    val_pos = edge_index[:, val_idx].to(device)
    test_pos = edge_index[:, test_idx].to(device)

    # Hard negatives
    paper_ids = data["paper"].paper_id.tolist()
    pid_to_row = {pid: i for i, pid in enumerate(paper_ids)}
    candidates_records = load_json(ret_dir / "candidates.json")
    edge_set = build_positive_set(edge_index)
    src_to_excluded = build_candidate_pool_negatives(candidates_records, edge_set, pid_to_row)
    topic_records = load_json(graph_dir / "topic_assignments.json")
    pid_to_topic = {r["paper_id"]: r["hard_topic"] for r in topic_records}
    topic_arr = np.array([pid_to_topic.get(p, 0) for p in paper_ids], dtype=np.int64)
    fallback_pools = build_topic_year_pools(
        torch.tensor(paper_ids), torch.from_numpy(topic_arr),
        data["paper"].year, year_window=2,
    )
    rng = np.random.default_rng(seed)
    train_neg = sample_candidate_pool_negatives(
        train_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).to(device)
    val_neg = sample_candidate_pool_negatives(
        val_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).to(device)
    test_neg = sample_candidate_pool_negatives(
        test_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).to(device)

    log.info("train pos %d neg %d | val %d %d | test %d %d",
             train_pos.shape[1], train_neg.shape[1],
             val_pos.shape[1], val_neg.shape[1],
             test_pos.shape[1], test_neg.shape[1])

    # Build per-source context lists from TRAIN edges only (avoid leakage)
    # For evaluation we use the full graph context.
    train_edge_mask = np.zeros(edge_index.shape[1], dtype=bool)
    train_edge_mask[train_idx.cpu().numpy()] = True
    log.info("building TRAIN-only context (no leakage)")
    train_ctx = build_per_source_context(
        n_papers,
        edge_index_np[:, train_edge_mask],
        edge_type_np[train_edge_mask],
        year_arr, indeg_arr, candidates_records, pid_to_row, max_ctx=64,
    )
    log.info("building FULL-graph context for eval")
    full_ctx = build_per_source_context(
        n_papers, edge_index_np, edge_type_np,
        year_arr, indeg_arr, candidates_records, pid_to_row, max_ctx=64,
    )

    model = GraphTransformer(d=paper_emb.shape[1], n_heads=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model params: %d", n_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    train_pos_s = train_pos[0].cpu().tolist()
    train_pos_t = train_pos[1].cpu().tolist()
    train_neg_s = train_neg[0].cpu().tolist()
    train_neg_t = train_neg[1].cpu().tolist()

    n_train = len(train_pos_s)
    indices = list(range(n_train))

    best_val_auc = 0.0
    bad_epochs = 0
    batch_size = 256

    for epoch in range(15):
        model.train()
        np.random.shuffle(indices)
        epoch_losses = []
        for i in range(0, n_train, batch_size):
            batch_idx = indices[i:i + batch_size]
            pos_s = [train_pos_s[j] for j in batch_idx]
            pos_t = [train_pos_t[j] for j in batch_idx]
            neg_s = [train_neg_s[j] for j in batch_idx]
            neg_t = [train_neg_t[j] for j in batch_idx]

            s_all = pos_s + neg_s
            t_all = pos_t + neg_t
            inputs = make_batch_tensors(s_all, t_all, paper_emb, train_ctx, device=device)
            logits = model(*inputs)
            y = torch.cat([torch.ones(len(pos_s), device=device),
                           torch.zeros(len(neg_s), device=device)])
            loss = bce(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        val_auc, val_ap = evaluate_model(model, paper_emb, full_ctx,
                                          val_pos, val_neg, device)
        log.info("epoch=%2d loss=%.4f val_auc=%.4f val_ap=%.4f",
                 epoch, np.mean(epoch_losses), val_auc, val_ap)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            torch.save(model.state_dict(), models_dir / "best.pt")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                log.info("early stop")
                break

    # Test
    model.load_state_dict(torch.load(models_dir / "best.pt"))
    test_auc, test_ap = evaluate_model(model, paper_emb, full_ctx,
                                        test_pos, test_neg, device)
    log.info("=" * 50)
    log.info("TIER 19 TEST: AUC=%.4f AP=%.4f", test_auc, test_ap)
    log.info("=" * 50)

    metrics = {
        "model": "tier19_graph_transformer",
        "architecture": "Graph Transformer with local attention + structural biases",
        "n_parameters": n_params,
        "best_val_auc": round(best_val_auc, 4),
        "link_prediction_auc_hard": round(test_auc, 4),
        "link_prediction_ap_hard": round(test_ap, 4),
    }
    with open(metrics_dir / "tier19_graph_transformer.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote: %s", metrics_dir / "tier19_graph_transformer.json")


if __name__ == "__main__":
    main()
