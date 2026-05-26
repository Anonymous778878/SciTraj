"""
Tier 8 — Multi-hop Edge Aggregation.

T2 averages 1-hop neighbours. T8 extends to 2-hop:

    e' = a * e + b1 * mean_{j in N1(i)} w_ij e_j
              + b2 * mean_{k in N2(i)} v_ik e_k

where N1 and N2 are 1- and 2-hop neighbours. The 2-hop term captures
the transitive structure that motif analysis (§6) showed is heavily
over-represented (transitive triangle z = 1792). We compute the 2-hop
aggregator via a single sparse matrix multiplication on top of the
1-hop result, staying in 768 dimensions throughout.

a, b1, b2 are learned subject to a + b1 + b2 = 1 (softmax).

Output: outputs/metrics/tier8_multihop.json
        models/tier8_multihop/embeddings.npy
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

log = get_logger("tier8_multihop")


class MultiHopAgg(nn.Module):
    def __init__(self):
        super().__init__()
        # Logits for [self, hop1, hop2]
        self.logits = nn.Parameter(torch.zeros(3))

    def get_weights(self):
        return F.softmax(self.logits, dim=0)

    def forward(self, paper_emb, edge_index, edge_weight):
        n, d = paper_emb.shape
        weights = self.get_weights()
        a, b1, b2 = weights[0], weights[1], weights[2]

        src = edge_index[0]
        tgt = edge_index[1]
        # 1-hop weighted aggregation
        hop1 = torch.zeros_like(paper_emb)
        count = torch.zeros(n, device=paper_emb.device)
        msg = paper_emb[src] * edge_weight.unsqueeze(1)
        hop1.index_add_(0, tgt, msg)
        count.index_add_(0, tgt, edge_weight)
        mask = count > 0
        hop1[mask] = hop1[mask] / count[mask].unsqueeze(1)

        # 2-hop: aggregate hop1 again over the same edges
        hop2 = torch.zeros_like(paper_emb)
        count2 = torch.zeros(n, device=paper_emb.device)
        msg2 = hop1[src] * edge_weight.unsqueeze(1)
        hop2.index_add_(0, tgt, msg2)
        count2.index_add_(0, tgt, edge_weight)
        mask2 = count2 > 0
        hop2[mask2] = hop2[mask2] / count2[mask2].unsqueeze(1)

        out = a * paper_emb + b1 * hop1 + b2 * hop2
        return out


def cos_score(emb, src, tgt):
    a = F.normalize(emb[src], dim=1)
    b = F.normalize(emb[tgt], dim=1)
    return (a * b).sum(dim=1)


def evaluate(emb_np, pos, neg):
    a = emb_np[pos[0]] / (np.linalg.norm(emb_np[pos[0]], axis=1, keepdims=True) + 1e-8)
    b = emb_np[pos[1]] / (np.linalg.norm(emb_np[pos[1]], axis=1, keepdims=True) + 1e-8)
    pos_s = (a * b).sum(axis=1)
    a = emb_np[neg[0]] / (np.linalg.norm(emb_np[neg[0]], axis=1, keepdims=True) + 1e-8)
    b = emb_np[neg[1]] / (np.linalg.norm(emb_np[neg[1]], axis=1, keepdims=True) + 1e-8)
    neg_s = (a * b).sum(axis=1)
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
    models_dir = ensure_dir(Path(cfg["paths"]["models_dir"]) / "tier8_multihop")
    graph_dir = Path(cfg["paths"]["graph_dir"])
    ret_dir = Path(cfg["paths"]["retrieval_dir"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("loading graph")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    paper_emb = data["paper"].x_abstract.to(device)
    edge_index = data["paper", "trajectory", "paper"].edge_index.to(device)
    edge_attr = data["paper", "trajectory", "paper"].edge_attr.to(device)
    edge_weight = edge_attr[:, 4]

    n_papers = int(data["paper"].num_nodes)
    log.info("nodes=%d  edges=%d", n_papers, edge_index.shape[1])

    train_idx, val_idx, test_idx = prepare_temporal_split(
        data,
        train_year_max=cfg["corpus"]["train_years"][1],
        val_year_max=cfg["corpus"]["val_years"][1],
        test_year_max=cfg["corpus"]["test_years"][1],
    )
    train_pos = edge_index[:, train_idx]
    val_pos = edge_index[:, val_idx]
    test_pos = edge_index[:, test_idx]

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
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr,
        year_arr=data["paper"].year.cpu().numpy()
    ).to(device)
    val_neg = sample_candidate_pool_negatives(
        val_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr,
        year_arr=data["paper"].year.cpu().numpy()
    ).to(device)
    test_neg = sample_candidate_pool_negatives(
        test_pos.cpu(), src_to_excluded, rng, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr,
        year_arr=data["paper"].year.cpu().numpy()
    ).to(device)

    train_edge_index = edge_index[:, train_idx]
    train_edge_weight = edge_weight[train_idx]

    model = MultiHopAgg().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    best_val_auc = 0.0
    best_emb = None
    best_weights = None
    patience = 6
    bad_epochs = 0

    for epoch in range(60):
        model.train()
        emb = model(paper_emb, train_edge_index, train_edge_weight)
        pos_s = cos_score(emb, train_pos[0], train_pos[1])
        neg_s = cos_score(emb, train_neg[0], train_neg[1])
        loss = -F.logsigmoid(pos_s - neg_s).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            emb_full = model(paper_emb, edge_index, edge_weight)
        emb_np = emb_full.cpu().numpy()
        val_auc, _ = evaluate(emb_np, val_pos.cpu().numpy(), val_neg.cpu().numpy())
        weights = model.get_weights().detach().cpu().numpy()

        log.info("epoch=%2d loss=%.4f  val_auc=%.4f  a=%.3f b1=%.3f b2=%.3f",
                 epoch, loss.item(), val_auc, *weights)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc
            best_emb = emb_np
            best_weights = weights.copy()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    test_auc, test_ap = evaluate(best_emb, test_pos.cpu().numpy(), test_neg.cpu().numpy())
    log.info("=" * 50)
    log.info("TIER 8 TEST: AUC_hard=%.4f  AP_hard=%.4f", test_auc, test_ap)
    log.info("=" * 50)

    np.save(models_dir / "embeddings.npy", best_emb)
    metrics = {
        "model": "tier8_multihop",
        "architecture": "Multi-hop edge aggregation, learned hop weights, 768-d",
        "n_papers": n_papers,
        "best_val_auc": round(best_val_auc, 4),
        "link_prediction_auc_hard": round(test_auc, 4),
        "link_prediction_ap_hard": round(test_ap, 4),
        "learned_a": float(best_weights[0]),
        "learned_b1": float(best_weights[1]),
        "learned_b2": float(best_weights[2]),
    }
    with open(metrics_dir / "tier8_multihop.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("wrote: %s", metrics_dir / "tier8_multihop.json")


if __name__ == "__main__":
    main()
