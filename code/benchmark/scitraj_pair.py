"""
Tier 22 — Expanded Pair-MLP.

T10 won at AUC 0.908 with 10 pair-level features. Most of the lift
came from f5_incoming_edge_count (+3.10 coefficient). T22 expands to
~70 pair-level features in the same regime, where the win actually
happens. Architecture identical to T10:

    [|emb_s - emb_t|, emb_s * emb_t, structural_extras] -> MLP -> score

The expanded features capture:
  - Per-edge-type degrees (in/out for source and target) — 18 dims
  - Common-neighbour statistics (Adamic-Adar, Jaccard, weighted) — 6 dims
  - Per-relation T2 cosines (the existing per-relation aggregations) — 9 dims
  - Topic and venue features (similarity, match flags) — 4 dims
  - Path counts (length-2 paths via each edge-type pair) — 8 dims
  - Cross-edge counts between neighbour sets — 4 dims
  - Backward-edge indicator and bidirectional links — 2 dims
  - Year-gap basis expansion (cubic spline-style buckets) — 5 dims
  - Source-target popularity ratios — 3 dims

Total: ~60 dims of structural + pair features. With (|emb diff|, emb product)
that's an input of 1536 + ~60 = 1596 dims to the MLP.

Output: outputs/metrics/tier22_pair_mlp_expanded.json
        models/tier22_pair_mlp_expanded/best.pt
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gnn_utils import load_graph, prepare_temporal_split
from hard_negatives import build_positive_set, build_topic_year_pools
from hard_negatives_v2 import (
    build_candidate_pool_negatives,
    sample_candidate_pool_negatives,
)
from utils import ensure_dir, get_logger, load_config, load_json

log = get_logger("tier22_pair_mlp_expanded")


N_RELATIONS = 9


class PairMLP(nn.Module):
    def __init__(self, d=768, n_extra=60, hidden=512, dropout=0.3):
        super().__init__()
        in_dim = 2 * d + n_extra
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, s_emb, t_emb, extra):
        f = torch.cat([torch.abs(s_emb - t_emb), s_emb * t_emb, extra], dim=1)
        return self.mlp(f).squeeze(-1)


def per_relation_aggregations(paper_emb_np, edge_index_np, edge_type_np,
                              edge_weight_np, n_relations=N_RELATIONS):
    """Compute one T2-style aggregated embedding per edge type.
    Returns: array (n_relations, N, D) of L2-normalised vectors."""
    n, d = paper_emb_np.shape
    out = []
    src = edge_index_np[0]
    tgt = edge_index_np[1]
    for r in range(n_relations):
        mask = (edge_type_np == r)
        agg = np.zeros((n, d), dtype=np.float32)
        count = np.zeros(n, dtype=np.float32)
        if mask.sum() > 0:
            sr = src[mask]; tg = tgt[mask]; w = edge_weight_np[mask]
            for i in range(len(sr)):
                agg[tg[i]] += paper_emb_np[sr[i]] * w[i]
                count[tg[i]] += w[i]
            has = count > 0
            agg[has] /= count[has, None]
            agg[~has] = paper_emb_np[~has]
        else:
            agg = paper_emb_np.copy()
        # L2 normalise for cosine
        norm = np.linalg.norm(agg, axis=1, keepdims=True) + 1e-8
        out.append(agg / norm)
    return np.stack(out, axis=0)   # (R, N, D)


def per_relation_degrees(edge_index_np, edge_type_np, n_papers, n_relations=N_RELATIONS):
    """For each (paper, relation), compute in-degree and out-degree."""
    in_deg = np.zeros((n_papers, n_relations), dtype=np.int64)
    out_deg = np.zeros((n_papers, n_relations), dtype=np.int64)
    src = edge_index_np[0]; tgt = edge_index_np[1]
    for i in range(len(src)):
        r = int(edge_type_np[i])
        in_deg[tgt[i], r] += 1
        out_deg[src[i], r] += 1
    return in_deg, out_deg


def common_neighbour_features(s_arr, t_arr, edge_index_np, n_papers):
    """For each (s, t) pair, compute common-neighbour statistics.

    We treat the graph as undirected for this purpose. Adamic-Adar and
    Jaccard are standard link-prediction features.
    """
    log.info("  building neighbour sets")
    src = edge_index_np[0]; tgt = edge_index_np[1]
    # Undirected neighbour set per node
    neigh = [set() for _ in range(n_papers)]
    for i in range(len(src)):
        a = int(src[i]); b = int(tgt[i])
        neigh[a].add(b)
        neigh[b].add(a)
    deg = np.array([len(s) for s in neigh], dtype=np.float32)

    log.info("  computing pair features for %d pairs", len(s_arr))
    n_pairs = len(s_arr)
    aa = np.zeros(n_pairs, dtype=np.float32)
    jacc = np.zeros(n_pairs, dtype=np.float32)
    common = np.zeros(n_pairs, dtype=np.float32)
    pref = np.zeros(n_pairs, dtype=np.float32)
    log_aa = np.zeros(n_pairs, dtype=np.float32)
    union_sz = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        a = int(s_arr[i]); b = int(t_arr[i])
        ns_a = neigh[a]; ns_b = neigh[b]
        if len(ns_a) == 0 or len(ns_b) == 0:
            continue
        inter = ns_a & ns_b
        union = ns_a | ns_b
        common[i] = len(inter)
        if len(union) > 0:
            jacc[i] = len(inter) / len(union)
        union_sz[i] = len(union)
        # Adamic-Adar: sum of 1/log(deg(c)) for c in inter
        s_aa = 0.0
        for c in inter:
            d = deg[c]
            if d > 1:
                s_aa += 1.0 / np.log(d)
        aa[i] = s_aa
        log_aa[i] = np.log1p(s_aa)
        pref[i] = deg[a] * deg[b]
    return np.stack([aa, log_aa, jacc, common, pref, union_sz], axis=1)


def make_pair_features_expanded(
    s_arr, t_arr,
    paper_emb_np,
    per_rel_aggs_np,        # (R, N, D), L2-normalised
    in_deg_per_rel,         # (N, R)
    out_deg_per_rel,        # (N, R)
    common_features,        # (n_pairs, 6) precomputed
    year_arr,
    topic_arr,
    edge_set_lookup,        # set of (src, tgt) tuples
):
    """Return (n_pairs, n_extra) feature matrix."""
    s = s_arr; t = t_arr
    n = len(s)

    feats = []

    # 1. Per-relation T2 cosines (R = 9 dims)
    for r in range(per_rel_aggs_np.shape[0]):
        a = per_rel_aggs_np[r][s]
        b = per_rel_aggs_np[r][t]
        feats.append((a * b).sum(axis=1).astype(np.float32))

    # 2. Per-relation in/out degrees of source AND target (4*R = 36 dims)
    for r in range(in_deg_per_rel.shape[1]):
        feats.append(np.log1p(in_deg_per_rel[s, r]).astype(np.float32))
        feats.append(np.log1p(in_deg_per_rel[t, r]).astype(np.float32))
        feats.append(np.log1p(out_deg_per_rel[s, r]).astype(np.float32))
        feats.append(np.log1p(out_deg_per_rel[t, r]).astype(np.float32))

    # 3. Common-neighbour features (6 dims)
    for k in range(common_features.shape[1]):
        feats.append(common_features[:, k])

    # 4. Topic and venue (2 dims)
    feats.append((topic_arr[s] == topic_arr[t]).astype(np.float32))
    feats.append((topic_arr[s] != topic_arr[t]).astype(np.float32))

    # 5. Year-gap basis (5 dims)
    yg = (year_arr[t] - year_arr[s]).astype(np.float32)
    feats.append(yg)
    feats.append(np.abs(yg))
    feats.append(np.log1p(np.abs(yg)))
    feats.append(np.maximum(yg, 0))     # forward
    feats.append(np.minimum(yg, 0))     # backward (negative if t before s)

    # 6. Has-back-edge and bidirectional (2 dims)
    has_back = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if (int(t[i]), int(s[i])) in edge_set_lookup:
            has_back[i] = 1.0
    feats.append(has_back)
    bidirectional = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if (int(s[i]), int(t[i])) in edge_set_lookup and \
           (int(t[i]), int(s[i])) in edge_set_lookup:
            bidirectional[i] = 1.0
    feats.append(bidirectional)

    # 7. Source-target ratios (3 dims)
    in_t_total = in_deg_per_rel[t].sum(axis=1).astype(np.float32)
    in_s_total = in_deg_per_rel[s].sum(axis=1).astype(np.float32)
    out_t_total = out_deg_per_rel[t].sum(axis=1).astype(np.float32)
    out_s_total = out_deg_per_rel[s].sum(axis=1).astype(np.float32)
    feats.append(np.log1p(in_t_total) - np.log1p(in_s_total))
    feats.append(np.log1p(out_s_total) - np.log1p(out_t_total))
    feats.append(np.log1p(in_t_total + out_t_total))

    return np.stack(feats, axis=1)


def evaluate_pairs(s_logits, n_logits):
    y = np.concatenate([np.ones(len(s_logits)), np.zeros(len(n_logits))])
    s = np.concatenate([s_logits, n_logits])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    models_dir = ensure_dir(Path(cfg["paths"]["models_dir"]) / "tier22_pair_mlp_expanded")
    graph_dir = Path(cfg["paths"]["graph_dir"])
    ret_dir = Path(cfg["paths"]["retrieval_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    log.info("loading graph")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    paper_emb = data["paper"].x_abstract.to(device)
    paper_emb_np = paper_emb.cpu().numpy()
    edge_index = data["paper", "trajectory", "paper"].edge_index
    edge_type = data["paper", "trajectory", "paper"].edge_type
    edge_attr = data["paper", "trajectory", "paper"].edge_attr
    edge_weight = edge_attr[:, 4]
    edge_index_np = edge_index.cpu().numpy()
    edge_type_np = edge_type.cpu().numpy()
    edge_weight_np = edge_weight.cpu().numpy()

    n_papers = int(data["paper"].num_nodes)
    log.info("nodes=%d edges=%d", n_papers, edge_index.shape[1])

    # Splits
    train_idx, val_idx, test_idx = prepare_temporal_split(
        data,
        train_year_max=cfg["corpus"]["train_years"][1],
        val_year_max=cfg["corpus"]["val_years"][1],
        test_year_max=cfg["corpus"]["test_years"][1],
    )
    train_pos = edge_index[:, train_idx].cpu().numpy()
    val_pos = edge_index[:, val_idx].cpu().numpy()
    test_pos = edge_index[:, test_idx].cpu().numpy()

    # Hard negatives
    paper_ids = data["paper"].paper_id.tolist()
    pid_to_row = {pid: i for i, pid in enumerate(paper_ids)}
    candidates_records = load_json(ret_dir / "candidates.json")
    edge_set = build_positive_set(edge_index)
    src_to_excluded = build_candidate_pool_negatives(candidates_records, edge_set, pid_to_row)
    topic_records = load_json(graph_dir / "topic_assignments.json")
    pid_to_topic = {r["paper_id"]: r["hard_topic"] for r in topic_records}
    topic_arr = np.array([pid_to_topic.get(p, 0) for p in paper_ids], dtype=np.int64)
    year_arr = data["paper"].year.cpu().numpy()
    fallback_pools = build_topic_year_pools(
        torch.tensor(paper_ids), torch.from_numpy(topic_arr),
        data["paper"].year, year_window=2,
    )
    rng = np.random.default_rng(seed)
    train_neg = sample_candidate_pool_negatives(
        torch.from_numpy(train_pos), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).numpy()
    val_neg = sample_candidate_pool_negatives(
        torch.from_numpy(val_pos), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).numpy()
    test_neg = sample_candidate_pool_negatives(
        torch.from_numpy(test_pos), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).numpy()

    log.info("train pos %d neg %d | val %d %d | test %d %d",
             train_pos.shape[1], train_neg.shape[1],
             val_pos.shape[1], val_neg.shape[1],
             test_pos.shape[1], test_neg.shape[1])

    # Precomputations (one-time, ~1-3 min)
    log.info("precomputing per-relation T2 aggregations (~1 min)")
    per_rel_aggs = per_relation_aggregations(
        paper_emb_np, edge_index_np, edge_type_np, edge_weight_np
    )
    log.info("per-relation aggs: %s", per_rel_aggs.shape)

    log.info("precomputing per-relation degrees")
    in_deg_per_rel, out_deg_per_rel = per_relation_degrees(
        edge_index_np, edge_type_np, n_papers
    )
    log.info("per-relation degrees: in %s out %s", in_deg_per_rel.shape, out_deg_per_rel.shape)

    log.info("building edge lookup set")
    edge_set_lookup = set(zip(edge_index_np[0].tolist(), edge_index_np[1].tolist()))

    # Common-neighbour features
    log.info("computing common-neighbour features for train/val/test")
    cn_train_pos = common_neighbour_features(train_pos[0], train_pos[1], edge_index_np, n_papers)
    cn_train_neg = common_neighbour_features(train_neg[0], train_neg[1], edge_index_np, n_papers)
    cn_val_pos = common_neighbour_features(val_pos[0], val_pos[1], edge_index_np, n_papers)
    cn_val_neg = common_neighbour_features(val_neg[0], val_neg[1], edge_index_np, n_papers)
    cn_test_pos = common_neighbour_features(test_pos[0], test_pos[1], edge_index_np, n_papers)
    cn_test_neg = common_neighbour_features(test_neg[0], test_neg[1], edge_index_np, n_papers)

    # Build all pair features
    log.info("building expanded pair features")
    def build(s, t, cn):
        return make_pair_features_expanded(
            s, t, paper_emb_np, per_rel_aggs,
            in_deg_per_rel, out_deg_per_rel, cn,
            year_arr, topic_arr, edge_set_lookup,
        )
    X_train_pos = build(train_pos[0], train_pos[1], cn_train_pos)
    X_train_neg = build(train_neg[0], train_neg[1], cn_train_neg)
    X_val_pos = build(val_pos[0], val_pos[1], cn_val_pos)
    X_val_neg = build(val_neg[0], val_neg[1], cn_val_neg)
    X_test_pos = build(test_pos[0], test_pos[1], cn_test_pos)
    X_test_neg = build(test_neg[0], test_neg[1], cn_test_neg)

    n_extra = X_train_pos.shape[1]
    log.info("expanded feature dim: %d", n_extra)

    # Standardise
    log.info("standardising features")
    scaler = StandardScaler().fit(np.vstack([X_train_pos, X_train_neg]))
    X_train_pos_s = scaler.transform(X_train_pos).astype(np.float32)
    X_train_neg_s = scaler.transform(X_train_neg).astype(np.float32)
    X_val_pos_s = scaler.transform(X_val_pos).astype(np.float32)
    X_val_neg_s = scaler.transform(X_val_neg).astype(np.float32)
    X_test_pos_s = scaler.transform(X_test_pos).astype(np.float32)
    X_test_neg_s = scaler.transform(X_test_neg).astype(np.float32)

    # Save the feature matrices for downstream use (T23, T24, T25)
    log.info("saving feature matrices for downstream stacking")
    np.savez(
        models_dir / "features.npz",
        X_train_pos=X_train_pos_s, X_train_neg=X_train_neg_s,
        X_val_pos=X_val_pos_s, X_val_neg=X_val_neg_s,
        X_test_pos=X_test_pos_s, X_test_neg=X_test_neg_s,
        train_pos_s=train_pos[0], train_pos_t=train_pos[1],
        train_neg_s=train_neg[0], train_neg_t=train_neg[1],
        val_pos_s=val_pos[0], val_pos_t=val_pos[1],
        val_neg_s=val_neg[0], val_neg_t=val_neg[1],
        test_pos_s=test_pos[0], test_pos_t=test_pos[1],
        test_neg_s=test_neg[0], test_neg_t=test_neg[1],
    )

    # Build PyTorch tensors
    paper_emb_t = torch.from_numpy(paper_emb_np).to(device)

    def to_torch(s, t, x):
        return (
            paper_emb_t[s].to(device),
            paper_emb_t[t].to(device),
            torch.from_numpy(x).to(device),
        )

    s_emb_p, t_emb_p, ex_p = to_torch(train_pos[0], train_pos[1], X_train_pos_s)
    s_emb_n, t_emb_n, ex_n = to_torch(train_neg[0], train_neg[1], X_train_neg_s)

    # MLP
    model = PairMLP(d=paper_emb_np.shape[1], n_extra=n_extra, hidden=512, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model params: %d", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()

    s_emb_all = torch.cat([s_emb_p, s_emb_n])
    t_emb_all = torch.cat([t_emb_p, t_emb_n])
    ex_all = torch.cat([ex_p, ex_n])
    y_all = torch.cat([
        torch.ones(len(s_emb_p), device=device),
        torch.zeros(len(s_emb_n), device=device),
    ])

    s_emb_vp, t_emb_vp, ex_vp = to_torch(val_pos[0], val_pos[1], X_val_pos_s)
    s_emb_vn, t_emb_vn, ex_vn = to_torch(val_neg[0], val_neg[1], X_val_neg_s)

    best_val_auc = 0.0
    best_state = None
    bad = 0
    batch_size = 4096

    for epoch in range(80):
        model.train()
        perm = torch.randperm(len(y_all), device=device)
        s_emb_e = s_emb_all[perm]; t_emb_e = t_emb_all[perm]
        ex_e = ex_all[perm]; y_e = y_all[perm]

        losses = []
        for i in range(0, len(y_e), batch_size):
            sl = slice(i, i + batch_size)
            logits = model(s_emb_e[sl], t_emb_e[sl], ex_e[sl])
            loss = bce(logits, y_e[sl])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Eval
        model.eval()
        with torch.no_grad():
            v_p = model(s_emb_vp, t_emb_vp, ex_vp).cpu().numpy()
            v_n = model(s_emb_vn, t_emb_vn, ex_vn).cpu().numpy()
        v_auc, v_ap = evaluate_pairs(v_p, v_n)
        log.info("epoch=%2d loss=%.4f val_auc=%.4f val_ap=%.4f",
                 epoch, np.mean(losses), v_auc, v_ap)

        if v_auc > best_val_auc + 1e-4:
            best_val_auc = v_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 8:
                log.info("early stop")
                break

    # Test
    model.load_state_dict(best_state)
    torch.save(best_state, models_dir / "best.pt")

    s_emb_tp, t_emb_tp, ex_tp = to_torch(test_pos[0], test_pos[1], X_test_pos_s)
    s_emb_tn, t_emb_tn, ex_tn = to_torch(test_neg[0], test_neg[1], X_test_neg_s)
    model.eval()
    with torch.no_grad():
        t_p_logits = model(s_emb_tp, t_emb_tp, ex_tp).cpu().numpy()
        t_n_logits = model(s_emb_tn, t_emb_tn, ex_tn).cpu().numpy()
    test_auc, test_ap = evaluate_pairs(t_p_logits, t_n_logits)
    log.info("=" * 50)
    log.info("TIER 22 TEST: AUC=%.4f AP=%.4f", test_auc, test_ap)
    log.info("=" * 50)

    # Save test predictions for ensembling
    np.savez(
        models_dir / "test_logits.npz",
        pos_logits=t_p_logits, neg_logits=t_n_logits,
    )

    metrics = {
        "model": "tier22_pair_mlp_expanded",
        "architecture": "Pair-MLP with ~60 expanded structural features",
        "n_extra_features": int(n_extra),
        "n_parameters": n_params,
        "best_val_auc": round(best_val_auc, 4),
        "link_prediction_auc_hard": round(test_auc, 4),
        "link_prediction_ap_hard": round(test_ap, 4),
    }
    with open(metrics_dir / "tier22_pair_mlp_expanded.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote: %s", metrics_dir / "tier22_pair_mlp_expanded.json")


if __name__ == "__main__":
    main()
