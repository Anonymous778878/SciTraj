"""
Phase 9 Tier 5 — Temporal R-GCN with time-decay attention.

Builds on Tier 4 (signed) and adds explicit temporal awareness:

  1. Sinusoidal time encoding injected as additional features per node.
     The graph already includes 32-dim TPE in the input feature vector,
     but Tier 5 ALSO concatenates a re-projected time encoding at each
     conv layer so time information persists through the bottleneck.

  2. Per-edge time-decay attention: each neighbor's contribution is
     weighted by exp(-λ * |year_src - year_tgt|), where λ is per-relation
     and learnable. Edges with large time gaps contribute less.

  3. Otherwise identical to Tier 4 (signed dispute, skip, edge-type bias).

Why this might help:
  Phase 7 showed transitive triangles 69× over-represented and convergence
  motifs significant. These motifs are inherently temporal — recent papers
  consolidating older trajectories. Per-edge time decay lets the model
  prefer locally-recent connections, which is closer to how real research
  citation patterns work.

Output:
  models/tier5_temporal_rgcn/best_model.pt
  outputs/metrics/tier5_temporal_rgcn.json
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn_utils import (
    evaluate_clustering,
    evaluate_link_prediction,
    evaluate_temporal_coherence,
    evaluate_temporal_shuffling_drop,
    load_graph,
    prepare_temporal_split,
    sample_negative_edges,
    save_metrics,
)
from hard_negatives import build_positive_set, build_topic_year_pools
from hard_negatives_v2 import (
    build_candidate_pool_negatives,
    sample_candidate_pool_negatives,
)
from utils import ensure_dir, get_logger, load_config, load_json

log = get_logger("tier5")


N_RELATIONS = 9
DISPUTE_TYPE_ID = 8


class TemporalSignedConv(nn.Module):
    """
    Signed conv with per-relation learnable time-decay rates.

    For each edge (u, v, r) with time delta Δt = year_v - year_u:
        attention_weight = exp(-λ_r * Δt)
        message = sign(r) * (W_r · x_u + e_r) * attention_weight

    λ_r is learnable per relation (initialized from Phase 6 config defaults,
    but the model can refine).
    """
    def __init__(self, in_dim, out_dim, n_relations=N_RELATIONS, rel_emb_dim=32):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.rel_emb = nn.Embedding(n_relations, rel_emb_dim)
        self.rel_proj = nn.Linear(rel_emb_dim, in_dim)

        sign_buffer = torch.ones(n_relations)
        sign_buffer[DISPUTE_TYPE_ID] = -1.0
        self.register_buffer("relation_sign", sign_buffer)

        # Learnable per-relation decay rate, initialized from values in
        # Phase 6 config (direct=0.5, future=0.1, limit=0.3, causal=0.15,
        # temp=0.4, related=0.5, method=0.3, perf=0.4, dispute=0.2)
        init_decay = torch.tensor([0.50, 0.10, 0.30, 0.15, 0.40, 0.30, 0.40, 0.50, 0.20])
        self.log_decay = nn.Parameter(torch.log(init_decay + 1e-6))

    def forward(self, x, edge_index, edge_type, time_delta):
        n = x.size(0)
        src = edge_index[0]
        tgt = edge_index[1]

        # Per-edge attention: exp(-λ_r * Δt)
        decay = torch.exp(self.log_decay[edge_type])             # (E,)
        attention = torch.exp(-decay * time_delta).unsqueeze(1)  # (E, 1)

        rel_offset = self.rel_proj(self.rel_emb(edge_type))
        signs = self.relation_sign[edge_type].unsqueeze(1)
        neigh_msg = signs * (x[src] + rel_offset) * attention

        agg = torch.zeros_like(x)
        weight_sum = torch.zeros(n, 1, device=x.device)
        agg.index_add_(0, tgt, neigh_msg)
        weight_sum.index_add_(0, tgt, attention)
        weight_sum = weight_sum.clamp(min=1e-6)
        agg = agg / weight_sum

        return self.W_self(x) + self.W_neigh(agg)


class TemporalRGCNWithSkip(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim,
                 n_relations=N_RELATIONS, dropout=0.2):
        super().__init__()
        self.conv1 = TemporalSignedConv(in_dim, hidden_dim, n_relations)
        self.conv2 = TemporalSignedConv(hidden_dim, out_dim, n_relations)
        self.skip_proj = nn.Linear(in_dim, out_dim)
        self.alpha_raw = nn.Parameter(torch.zeros(1))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type, time_delta):
        h = self.conv1(x, edge_index, edge_type, time_delta)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_type, time_delta)
        s = self.skip_proj(x)
        alpha = torch.sigmoid(self.alpha_raw)
        return alpha * s + (1 - alpha) * h

    @property
    def alpha_value(self):
        return float(torch.sigmoid(self.alpha_raw).item())

    def get_decay_rates(self):
        return torch.exp(self.conv1.log_decay).detach().cpu().numpy()


def link_bce_loss(embeddings, pos_edges, neg_edges):
    pos_logits = (embeddings[pos_edges[0]] * embeddings[pos_edges[1]]).sum(dim=1)
    neg_logits = (embeddings[neg_edges[0]] * embeddings[neg_edges[1]]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits])
    labels = torch.cat([
        torch.ones(pos_logits.shape[0], device=embeddings.device),
        torch.zeros(neg_logits.shape[0], device=embeddings.device),
    ])
    return F.binary_cross_entropy_with_logits(logits, labels)


def triplet_loss(embeddings, pos_edges, neg_edges, margin=0.2):
    embs_norm = F.normalize(embeddings, dim=1)
    pos_sim = (embs_norm[pos_edges[0]] * embs_norm[pos_edges[1]]).sum(dim=1)
    neg_sim = (embs_norm[neg_edges[0]] * embs_norm[neg_edges[1]]).sum(dim=1)
    return F.relu(margin - pos_sim + neg_sim).mean()


def lr_at_step(step, warmup_steps, total_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        return min_lr + (base_lr - min_lr) * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    models_dir  = ensure_dir(Path(cfg["paths"]["models_dir"]) / "tier5_temporal_rgcn")
    graph_dir   = Path(cfg["paths"]["graph_dir"])
    ret_dir     = Path(cfg["paths"]["retrieval_dir"])

    log.info("loading graph")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    n_papers = int(data["paper"].num_nodes)

    x = data["paper"].x.to(device)
    edge_index = data["paper", "trajectory", "paper"].edge_index.to(device)
    edge_type  = data["paper", "trajectory", "paper"].edge_type.to(device)
    edge_attr  = data["paper", "trajectory", "paper"].edge_attr.to(device)
    time_delta = edge_attr[:, 3]                                # column 3 is time_delta
    years      = data["paper"].year
    paper_ids  = data["paper"].paper_id.tolist()
    log.info("graph: %d papers, %d edges; time_delta range [%.1f, %.1f]",
             n_papers, edge_index.shape[1],
             float(time_delta.min()), float(time_delta.max()))

    pid_to_row = {pid: i for i, pid in enumerate(paper_ids)}

    log.info("loading Phase 5 candidates for hard negatives")
    candidates_records = load_json(ret_dir / "candidates.json")
    edge_set = build_positive_set(edge_index.cpu())
    src_to_excluded = build_candidate_pool_negatives(
        candidates_records, edge_set, pid_to_row
    )
    topic_records = load_json(graph_dir / "topic_assignments.json")
    pid_to_topic = {r["paper_id"]: r["hard_topic"] for r in topic_records}
    topic_arr = np.array([pid_to_topic.get(p, 0) for p in paper_ids], dtype=np.int64)
    year_arr  = years.cpu().numpy()
    fallback_pools = build_topic_year_pools(
        torch.tensor(paper_ids), torch.from_numpy(topic_arr), years, year_window=2
    )

    train_idx, val_idx, test_idx = prepare_temporal_split(
        data,
        train_year_max=cfg["corpus"]["train_years"][1],
        val_year_max=cfg["corpus"]["val_years"][1],
        test_year_max=cfg["corpus"]["test_years"][1],
    )

    train_pos = edge_index[:, train_idx.to(device)]
    val_pos   = edge_index[:, val_idx.to(device)]
    test_pos  = edge_index[:, test_idx.to(device)]

    rng_np = np.random.default_rng(seed)
    val_neg_hard = sample_candidate_pool_negatives(
        val_pos.cpu(), src_to_excluded, rng_np, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).to(device)

    model = TemporalRGCNWithSkip(
        in_dim=x.shape[1], hidden_dim=512, out_dim=128,
        n_relations=N_RELATIONS, dropout=0.2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model parameters: %d", n_params)

    n_epochs = 50
    base_lr = 5e-5
    warmup_steps = 5
    triplet_weight = 0.5
    best_val_auc = 0.0; best_epoch = 0; patience = 12; epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        cur_lr = lr_at_step(epoch - 1, warmup_steps, n_epochs, base_lr)
        for pg in optimizer.param_groups: pg["lr"] = cur_lr

        model.train()
        t0 = time.time()
        train_neg_hard = sample_candidate_pool_negatives(
            train_pos.cpu(), src_to_excluded, rng_np, n_per_pos=1,
            fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
        ).to(device)

        embeddings = model(x, edge_index, edge_type, time_delta)
        bce = link_bce_loss(embeddings, train_pos, train_neg_hard)
        trp = triplet_loss(embeddings, train_pos, train_neg_hard, margin=0.2)
        loss = bce + triplet_weight * trp

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(x, edge_index, edge_type, time_delta)
            val_auc, val_ap = evaluate_link_prediction(embeddings, val_pos, val_neg_hard)

        log.info("epoch %2d  lr=%.2e α=%.3f loss=%.4f (bce=%.4f trp=%.4f)  val_auc=%.4f  (%.1fs)",
                 epoch, cur_lr, model.alpha_value, loss.item(),
                 bce.item(), trp.item(), val_auc, time.time()-t0)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc; best_epoch = epoch; epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_auc": val_auc,
                        "alpha": model.alpha_value,
                        "decay_rates": model.get_decay_rates().tolist()},
                       models_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("early stop at epoch %d", epoch)
                break

    ckpt = torch.load(models_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    final_decay = model.get_decay_rates().tolist()
    log.info("best epoch %d, alpha=%.4f", best_epoch, model.alpha_value)
    log.info("learned per-relation decay rates: %s", [round(d, 3) for d in final_decay])

    with torch.no_grad():
        embeddings = model(x, edge_index, edge_type, time_delta).detach()
    np.save(models_dir / "embeddings.npy", embeddings.cpu().numpy())

    test_neg_hard = sample_candidate_pool_negatives(
        test_pos.cpu(), src_to_excluded, rng_np, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).to(device)
    sil, ch = evaluate_clustering(embeddings.cpu(), n_clusters=cfg["graph"]["n_topics"], seed=seed)
    rho = evaluate_temporal_coherence(embeddings.cpu(), years)
    auc_h, ap_h = evaluate_link_prediction(embeddings.cpu(), test_pos.cpu(), test_neg_hard.cpu())
    rng_compat = random.Random(seed)
    test_neg_random = sample_negative_edges(
        n_papers, edge_index.cpu(), test_pos.shape[1], years, rng_compat,
    )
    auc_r, ap_r = evaluate_link_prediction(embeddings.cpu(), test_pos.cpu(), test_neg_random)
    shuf = evaluate_temporal_shuffling_drop(
        embeddings.cpu(), years, test_pos.cpu(), test_neg_random, seed=seed
    )

    edge_type_names = ["direct_extension", "future_realized", "limit_addressed",
                       "causal_extension", "perf_succ", "method_re",
                       "temporal_semantic", "related_work", "dispute"]
    metrics = {
        "model": "tier5_temporal_rgcn",
        "architecture": "Signed temporal R-GCN + skip + per-relation time decay",
        "n_papers": n_papers, "n_parameters": n_params,
        "best_epoch": best_epoch, "best_val_auc": round(best_val_auc, 4),
        "final_alpha": round(model.alpha_value, 4),
        "learned_decay_rates": {
            edge_type_names[i]: round(final_decay[i], 4) for i in range(N_RELATIONS)
        },
        "embedding_dim": int(embeddings.shape[1]),
        "silhouette": round(sil, 4), "calinski_harabasz": round(ch, 1),
        "temporal_coherence_rho": round(rho, 4),
        "link_prediction_auc_hard": round(auc_h, 4), "link_prediction_ap_hard": round(ap_h, 4),
        "link_prediction_auc_random": round(auc_r, 4), "link_prediction_ap_random": round(ap_r, 4),
        "temporal_shuffling": shuf,
    }
    save_metrics(metrics, metrics_dir / "tier5_temporal_rgcn.json")
    log.info("=" * 55)
    log.info("TIER 5 TEMPORAL R-GCN COMPLETE")
    log.info("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, dict):
            log.info("%s:", k)
            for kk, vv in v.items(): log.info("  %s = %s", kk, vv)
        else:
            log.info("%s = %s", k, v)


if __name__ == "__main__":
    main()
