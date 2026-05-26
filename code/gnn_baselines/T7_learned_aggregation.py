"""
Tier 7 — Learned-weight Edge Aggregation (LwEdge).

T2 computes  e' = alpha * e + (1-alpha) * mean_{j in N(i)} w_ij * e_j
with alpha = 0.5 fixed. T7 generalises:

    e' = alpha * e + sum_{r in relations} beta_r * mean_{j in N_r(i)} w_ij * e_j
    s.t.  alpha + sum_r beta_r = 1   (softmax-parameterised)

where beta_r is a learned per-relation aggregation weight, and the
output stays in 768 dimensions. We optimise alpha and {beta_r} jointly
on training-set link prediction with a small validation-set early stop.

This avoids the 128-dim bottleneck that destroyed T3-T6's retrieval
performance, while still being a learned model. If T7 beats T2, the
negative result in §7 is refined to "no GNN with 128-dim bottleneck
improves on T2"; if T7 ties T2, the result is sharper.

Output: outputs/metrics/tier7_learned_aggregation.json
        models/tier7_learned_aggregation/embeddings.npy
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

log = get_logger("tier7_learned_agg")


N_RELATIONS = 9


class LearnedAggregation(nn.Module):
    """
    e'_i = a * e_i + sum_r b_r * mean_{j in N_r(i)} w_ij * e_j
    with a, b_0, ..., b_{R-1} >= 0 and a + sum b_r = 1.
    """
    def __init__(self, n_relations: int = N_RELATIONS):
        super().__init__()
        # Pre-softmax logits for [self, rel_0, rel_1, ..., rel_{R-1}]
        self.weight_logits = nn.Parameter(torch.zeros(n_relations + 1))

    def get_weights(self):
        return F.softmax(self.weight_logits, dim=0)

    def forward(self, paper_emb, edge_index, edge_type, edge_weight):
        """
        paper_emb:    (N, D)     SPECTER2 features
        edge_index:   (2, E)
        edge_type:    (E,)       int64 in [0, R-1]
        edge_weight:  (E,)       float
        Returns: (N, D) aggregated embeddings.
        """
        n, d = paper_emb.shape
        weights = self.get_weights()
        a = weights[0]
        b = weights[1:]      # (R,)

        out = a * paper_emb

        # For each relation, compute neighbour-mean and add b_r * mean
        for r in range(N_RELATIONS):
            mask = (edge_type == r)
            if mask.sum() == 0:
                continue
            src = edge_index[0, mask]
            tgt = edge_index[1, mask]
            w = edge_weight[mask]
            # weighted message: w_ij * e_j flowing src -> tgt
            messages = paper_emb[src] * w.unsqueeze(1)
            # mean per target via index_add
            agg = torch.zeros_like(paper_emb)
            count = torch.zeros(n, device=paper_emb.device)
            agg.index_add_(0, tgt, messages)
            count.index_add_(0, tgt, w)
            mask_has = count > 0
            agg[mask_has] = agg[mask_has] / count[mask_has].unsqueeze(1)
            out = out + b[r] * agg

        return out


def link_score(emb, src, tgt):
    a = F.normalize(emb[src], dim=1)
    b = F.normalize(emb[tgt], dim=1)
    return (a * b).sum(dim=1)


def evaluate(emb_np, pos, neg):
    pos_s = (emb_np[pos[0]] * emb_np[pos[1]]).sum(axis=1)
    neg_s = (emb_np[neg[0]] * emb_np[neg[1]]).sum(axis=1)
    # cosine: normalise first
    pos_n = (
        emb_np[pos[0]] / (np.linalg.norm(emb_np[pos[0]], axis=1, keepdims=True) + 1e-8)
    )
    pos_t = (
        emb_np[pos[1]] / (np.linalg.norm(emb_np[pos[1]], axis=1, keepdims=True) + 1e-8)
    )
    pos_s = (pos_n * pos_t).sum(axis=1)
    neg_n = (
        emb_np[neg[0]] / (np.linalg.norm(emb_np[neg[0]], axis=1, keepdims=True) + 1e-8)
    )
    neg_t = (
        emb_np[neg[1]] / (np.linalg.norm(emb_np[neg[1]], axis=1, keepdims=True) + 1e-8)
    )
    neg_s = (neg_n * neg_t).sum(axis=1)

    y_true = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    y_score = np.concatenate([pos_s, neg_s])
    return float(roc_auc_score(y_true, y_score)), float(
        average_precision_score(y_true, y_score)
    )


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    models_dir = ensure_dir(Path(cfg["paths"]["models_dir"]) / "tier7_learned_aggregation")
    graph_dir = Path(cfg["paths"]["graph_dir"])
    ret_dir = Path(cfg["paths"]["retrieval_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    log.info("loading graph")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    paper_emb = data["paper"].x_abstract.to(device)
    edge_index = data["paper", "trajectory", "paper"].edge_index.to(device)
    edge_type = data["paper", "trajectory", "paper"].edge_type.to(device)
    edge_attr = data["paper", "trajectory", "paper"].edge_attr.to(device)
    edge_weight = edge_attr[:, 4]

    n_papers = int(data["paper"].num_nodes)
    log.info("nodes=%d  edges=%d  dim=%d", n_papers, edge_index.shape[1], paper_emb.shape[1])

    # Splits
    train_idx, val_idx, test_idx = prepare_temporal_split(
        data,
        train_year_max=cfg["corpus"]["train_years"][1],
        val_year_max=cfg["corpus"]["val_years"][1],
        test_year_max=cfg["corpus"]["test_years"][1],
    )
    train_pos = edge_index[:, train_idx]
    val_pos = edge_index[:, val_idx]
    test_pos = edge_index[:, test_idx]

    # Hard negatives
    paper_ids = data["paper"].paper_id.tolist()
    pid_to_row = {pid: i for i, pid in enumerate(paper_ids)}
    candidates_records = load_json(ret_dir / "candidates.json")
    edge_set = build_positive_set(edge_index)
    src_to_excluded = build_candidate_pool_negatives(
        candidates_records, edge_set, pid_to_row
    )
    topic_records = load_json(graph_dir / "topic_assignments.json")
    pid_to_topic = {r["paper_id"]: r["hard_topic"] for r in topic_records}
    topic_arr = np.array([pid_to_topic.get(p, 0) for p in paper_ids], dtype=np.int64)
    year_arr = data["paper"].year.cpu().numpy()
    fallback_pools = build_topic_year_pools(
        torch.tensor(paper_ids),
        torch.from_numpy(topic_arr),
        data["paper"].year,
        year_window=2,
    )
    rng = np.random.default_rng(seed)

    # Sample hard negatives for train, val, test
    train_neg = sample_candidate_pool_negatives(
        train_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr
    ).to(device)
    val_neg = sample_candidate_pool_negatives(
        val_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr
    ).to(device)
    test_neg = sample_candidate_pool_negatives(
        test_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr
    ).to(device)

    log.info("train pos %d neg %d | val pos %d neg %d | test pos %d neg %d",
             train_pos.shape[1], train_neg.shape[1],
             val_pos.shape[1], val_neg.shape[1],
             test_pos.shape[1], test_neg.shape[1])

    # Model
    model = LearnedAggregation(n_relations=N_RELATIONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Train: only training-time edges in the message graph (avoid leakage)
    train_edge_index = edge_index[:, train_idx]
    train_edge_type = edge_type[train_idx]
    train_edge_weight = edge_weight[train_idx]

    best_val_auc = 0.0
    best_emb = None
    best_weights = None
    patience = 6
    bad_epochs = 0

    for epoch in range(60):
        model.train()
        emb = model(paper_emb, train_edge_index, train_edge_type, train_edge_weight)

        pos_s = link_score(emb, train_pos[0], train_pos[1])
        neg_s = link_score(emb, train_neg[0], train_neg[1])
        # margin-style: encourage pos > neg + margin
        loss = -F.logsigmoid(pos_s - neg_s).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval
        model.eval()
        with torch.no_grad():
            emb_eval = model(paper_emb, edge_index, edge_type, edge_weight)
        emb_np = emb_eval.cpu().numpy()
        val_pos_np = val_pos.cpu().numpy()
        val_neg_np = val_neg.cpu().numpy()
        val_auc, val_ap = evaluate(emb_np, val_pos_np, val_neg_np)
        weights = model.get_weights().detach().cpu().numpy()

        log.info(
            "epoch=%2d loss=%.4f  val_auc=%.4f  alpha=%.3f  top_betas=%s",
            epoch, loss.item(), val_auc, weights[0],
            np.round(weights[1:5], 3).tolist(),
        )

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_emb = emb_np
            best_weights = weights.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log.info("early stop at epoch %d", epoch)
                break

    log.info("best val AUC: %.4f", best_val_auc)
    log.info("learned weights: alpha=%.4f", best_weights[0])
    edge_type_names = [
        "direct_extension", "future_realized", "limit_addressed",
        "causal_extension", "performance_successor", "method_reuse",
        "temporal_semantic", "related_work", "dispute",
    ]
    for i, name in enumerate(edge_type_names):
        log.info("  beta[%s] = %.4f", name, best_weights[1 + i])

    # Test
    test_pos_np = test_pos.cpu().numpy()
    test_neg_np = test_neg.cpu().numpy()
    test_auc, test_ap = evaluate(best_emb, test_pos_np, test_neg_np)
    log.info("=" * 50)
    log.info("TIER 7 TEST: AUC_hard=%.4f  AP_hard=%.4f", test_auc, test_ap)
    log.info("=" * 50)

    # Save
    np.save(models_dir / "embeddings.npy", best_emb)
    metrics = {
        "model": "tier7_learned_aggregation",
        "architecture": "Learned per-relation aggregation, 768-d output",
        "n_papers": n_papers,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "best_val_auc": round(best_val_auc, 4),
        "link_prediction_auc_hard": round(test_auc, 4),
        "link_prediction_ap_hard": round(test_ap, 4),
        "learned_alpha": float(best_weights[0]),
        "learned_betas": {
            name: float(best_weights[1 + i])
            for i, name in enumerate(edge_type_names)
        },
        "embedding_dim": int(best_emb.shape[1]),
    }
    with open(metrics_dir / "tier7_learned_aggregation.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote: %s", metrics_dir / "tier7_learned_aggregation.json")


if __name__ == "__main__":
    main()
