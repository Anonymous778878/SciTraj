"""
Phase 9 Tier 6 — Heterogeneous multi-task flagship.

Builds on Tier 5 and adds two final pieces:

  1. HETEROGENEOUS GRAPH. Adds topic nodes with paper-topic edges
     (paper belongs to topic with weight = soft membership). The model
     processes both relation types: paper-trajectory-paper (typed signed
     temporal, from Tiers 4+5) and paper-belongs_to-topic.

  2. MULTI-TASK LEARNING. Three simultaneous objectives:
        L = α_link · L_link + α_type · L_edge_type + α_topic · L_topic
     where:
       L_link    : link prediction (BCE + triplet, as before)
       L_edge_type: predict edge type given (src, tgt) embeddings
                    (9-class cross-entropy on positive edges only)
       L_topic   : reconstruct soft topic membership from paper embedding
                    (KL divergence between predicted and target distribution)

  This is the architecture closest to the original SciTraj-V2 design
  proposed in the project planning. It is also the most expensive.

Output:
  models/tier6_hetero_multitask/best_model.pt
  outputs/metrics/tier6_hetero_multitask.json
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

log = get_logger("tier6")


N_RELATIONS_PAPER = 9
DISPUTE_TYPE_ID = 8


class TemporalSignedConv(nn.Module):
    """Same as Tier 5 — signed + temporal + edge-type bias."""
    def __init__(self, in_dim, out_dim, n_relations=N_RELATIONS_PAPER, rel_emb_dim=32):
        super().__init__()
        self.W_self  = nn.Linear(in_dim, out_dim)
        self.W_neigh = nn.Linear(in_dim, out_dim)
        self.rel_emb = nn.Embedding(n_relations, rel_emb_dim)
        self.rel_proj = nn.Linear(rel_emb_dim, in_dim)

        sign_buffer = torch.ones(n_relations)
        sign_buffer[DISPUTE_TYPE_ID] = -1.0
        self.register_buffer("relation_sign", sign_buffer)

        init_decay = torch.tensor([0.50, 0.10, 0.30, 0.15, 0.40, 0.30, 0.40, 0.50, 0.20])
        self.log_decay = nn.Parameter(torch.log(init_decay + 1e-6))

    def forward(self, x, edge_index, edge_type, time_delta):
        n = x.size(0)
        src = edge_index[0]; tgt = edge_index[1]

        decay = torch.exp(self.log_decay[edge_type])
        attention = torch.exp(-decay * time_delta).unsqueeze(1)

        rel_offset = self.rel_proj(self.rel_emb(edge_type))
        signs = self.relation_sign[edge_type].unsqueeze(1)
        neigh_msg = signs * (x[src] + rel_offset) * attention

        agg = torch.zeros_like(x)
        weight_sum = torch.zeros(n, 1, device=x.device)
        agg.index_add_(0, tgt, neigh_msg)
        weight_sum.index_add_(0, tgt, attention)
        weight_sum = weight_sum.clamp(min=1e-6)
        return self.W_self(x) + self.W_neigh(agg / weight_sum)


class HeteroMultitaskModel(nn.Module):
    """
    Two-layer encoder that processes paper-paper trajectory edges,
    plus a topic-membership cross-attention mechanism that lets papers
    incorporate topic embeddings.

    Uses skip connection (from v4).
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_topics, topic_dim=768,
                 n_relations=N_RELATIONS_PAPER, dropout=0.2):
        super().__init__()
        self.conv1 = TemporalSignedConv(in_dim, hidden_dim, n_relations)
        self.conv2 = TemporalSignedConv(hidden_dim, out_dim, n_relations)

        # Skip connection
        self.skip_proj = nn.Linear(in_dim, out_dim)
        self.alpha_raw = nn.Parameter(torch.zeros(1))

        # Topic projection: project topic centroids to hidden_dim for attention
        self.topic_proj = nn.Linear(topic_dim, hidden_dim)

        # Topic-paper attention: each paper attends to all topics
        self.topic_attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.topic_attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.topic_attn_v = nn.Linear(hidden_dim, out_dim)

        # Multi-task heads
        # 1) edge-type classification head: takes [emb_src, emb_tgt, emb_src*emb_tgt] -> 9 classes
        self.edge_type_head = nn.Sequential(
            nn.Linear(out_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_relations),
        )
        # 2) topic membership head: project paper emb back to topic distribution
        self.topic_head = nn.Linear(out_dim, n_topics)

        self.dropout = dropout

    def forward(self, x_paper, edge_index, edge_type, time_delta, x_topic):
        # Paper-paper conv stack
        h1 = self.conv1(x_paper, edge_index, edge_type, time_delta)
        h1 = F.relu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Topic-attention on h1: each paper attends to projected topic centroids
        topic_h = self.topic_proj(x_topic)                       # (n_topics, hidden_dim)
        Q = self.topic_attn_q(h1)                                # (n_papers, hidden_dim)
        K = self.topic_attn_k(topic_h)                           # (n_topics, hidden_dim)
        V = self.topic_attn_v(topic_h)                           # (n_topics, out_dim)
        attn = torch.softmax(Q @ K.t() / math.sqrt(K.size(1)), dim=1)  # (n_papers, n_topics)
        topic_contribution = attn @ V                            # (n_papers, out_dim)

        # Continue with conv2 in hidden space for paper-paper
        h2 = self.conv2(h1, edge_index, edge_type, time_delta)

        # Skip path
        s = self.skip_proj(x_paper)
        alpha = torch.sigmoid(self.alpha_raw)

        # Final embedding: weighted combination
        emb = alpha * s + (1 - alpha) * (0.7 * h2 + 0.3 * topic_contribution)
        return emb

    def predict_edge_type(self, embeddings, edge_index):
        """Predict edge type given source-target embeddings."""
        src_emb = embeddings[edge_index[0]]
        tgt_emb = embeddings[edge_index[1]]
        feat = torch.cat([src_emb, tgt_emb, src_emb * tgt_emb], dim=1)
        return self.edge_type_head(feat)

    def predict_topic_membership(self, embeddings):
        """Predict soft topic distribution per paper."""
        return F.softmax(self.topic_head(embeddings), dim=1)

    @property
    def alpha_value(self):
        return float(torch.sigmoid(self.alpha_raw).item())


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


def topic_kl_loss(predicted_topic_dist, target_topic_dist):
    """KL divergence between predicted soft topic distribution and target."""
    eps = 1e-8
    pred = predicted_topic_dist.clamp(min=eps)
    return F.kl_div(pred.log(), target_topic_dist, reduction="batchmean")


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
    models_dir  = ensure_dir(Path(cfg["paths"]["models_dir"]) / "tier6_hetero_multitask")
    graph_dir   = Path(cfg["paths"]["graph_dir"])
    ret_dir     = Path(cfg["paths"]["retrieval_dir"])

    log.info("loading graph and topic centroids")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    n_papers = int(data["paper"].num_nodes)

    x_paper = data["paper"].x.to(device)
    edge_index = data["paper", "trajectory", "paper"].edge_index.to(device)
    edge_type  = data["paper", "trajectory", "paper"].edge_type.to(device)
    edge_attr  = data["paper", "trajectory", "paper"].edge_attr.to(device)
    time_delta = edge_attr[:, 3]
    years      = data["paper"].year
    paper_ids  = data["paper"].paper_id.tolist()

    # Topic features
    x_topic = data["topic"].x.to(device)                          # (n_topics, 768)
    n_topics = int(data["topic"].num_nodes)

    # Soft topic distribution as a target for the topic head
    topic_membership = data["paper"].topic_membership.to(device)  # (n_papers, n_topics)
    log.info("graph: %d papers, %d edges, %d topics", n_papers, edge_index.shape[1], n_topics)

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
    train_edge_types = edge_type[train_idx.to(device)]

    rng_np = np.random.default_rng(seed)
    val_neg_hard = sample_candidate_pool_negatives(
        val_pos.cpu(), src_to_excluded, rng_np, n_per_pos=1,
        fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
    ).to(device)

    model = HeteroMultitaskModel(
        in_dim=x_paper.shape[1], hidden_dim=512, out_dim=128,
        n_topics=n_topics, topic_dim=x_topic.shape[1],
        n_relations=N_RELATIONS_PAPER, dropout=0.2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=1e-4)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("model parameters: %d", n_params)

    n_epochs = 50
    base_lr = 5e-5
    warmup_steps = 5
    triplet_weight = 0.3
    edge_type_weight = 0.2     # multi-task weights — link is primary
    topic_weight = 0.1
    # Allow env-var override (used by C5 loss ablation)
    import os as _os
    if _os.environ.get("SCITRAJ_LOSS_WEIGHT_TYPE") is not None:
        try:
            edge_type_weight = float(_os.environ["SCITRAJ_LOSS_WEIGHT_TYPE"])
            log.info("override edge_type_weight from env: %g", edge_type_weight)
        except ValueError:
            pass
    if _os.environ.get("SCITRAJ_LOSS_WEIGHT_TOPIC") is not None:
        try:
            topic_weight = float(_os.environ["SCITRAJ_LOSS_WEIGHT_TOPIC"])
            log.info("override topic_weight from env: %g", topic_weight)
        except ValueError:
            pass
    variant_name = _os.environ.get("SCITRAJ_TIER6_VARIANT", "default")
    best_val_auc = 0.0; best_epoch = 0; patience = 12; epochs_no_improve = 0

    log.info("training: %d epochs, multi-task weights link=1.0 + type=%g + topic=%g",
             n_epochs, edge_type_weight, topic_weight)

    for epoch in range(1, n_epochs + 1):
        cur_lr = lr_at_step(epoch - 1, warmup_steps, n_epochs, base_lr)
        for pg in optimizer.param_groups: pg["lr"] = cur_lr

        model.train()
        t0 = time.time()
        train_neg_hard = sample_candidate_pool_negatives(
            train_pos.cpu(), src_to_excluded, rng_np, n_per_pos=1,
            fallback_topic_pools=fallback_pools, topic_arr=topic_arr, year_arr=year_arr,
        ).to(device)

        embeddings = model(x_paper, edge_index, edge_type, time_delta, x_topic)

        # Link prediction loss
        bce = link_bce_loss(embeddings, train_pos, train_neg_hard)
        trp = triplet_loss(embeddings, train_pos, train_neg_hard, margin=0.2)
        l_link = bce + triplet_weight * trp

        # Edge-type classification loss (positive edges only)
        # Sub-sample edges to keep memory manageable
        n_subsample = min(20000, train_pos.shape[1])
        sub_idx = torch.randint(0, train_pos.shape[1], (n_subsample,), device=device)
        sub_edges = train_pos[:, sub_idx]
        sub_types = train_edge_types[sub_idx]
        type_logits = model.predict_edge_type(embeddings, sub_edges)
        l_type = F.cross_entropy(type_logits, sub_types)

        # Topic membership loss (sample papers)
        topic_idx = torch.randint(0, n_papers, (5000,), device=device)
        pred_topic = model.predict_topic_membership(embeddings[topic_idx])
        target_topic = topic_membership[topic_idx]
        # Some topic memberships sum to <1 (papers with no top-k mass left over);
        # renormalize for KL stability
        target_topic = target_topic / target_topic.sum(dim=1, keepdim=True).clamp(min=1e-8)
        l_topic = topic_kl_loss(pred_topic, target_topic)

        loss = l_link + edge_type_weight * l_type + topic_weight * l_topic

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(x_paper, edge_index, edge_type, time_delta, x_topic)
            val_auc, val_ap = evaluate_link_prediction(embeddings, val_pos, val_neg_hard)
            # Edge-type accuracy for monitoring
            sub_pred = model.predict_edge_type(embeddings, val_pos[:, :5000]).argmax(dim=1)
            sub_truth = edge_type[val_idx.to(device)[:5000]]
            type_acc = float((sub_pred == sub_truth).float().mean())

        log.info("epoch %2d  lr=%.2e α=%.3f  L=%.3f (link=%.3f type=%.3f topic=%.3f)  val_auc=%.4f  type_acc=%.3f  (%.1fs)",
                 epoch, cur_lr, model.alpha_value, loss.item(),
                 l_link.item(), l_type.item(), l_topic.item(),
                 val_auc, type_acc, time.time()-t0)

        if val_auc > best_val_auc + 1e-4:
            best_val_auc = val_auc; best_epoch = epoch; epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "val_auc": val_auc,
                        "alpha": model.alpha_value, "type_acc": type_acc},
                       models_dir / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("early stop at epoch %d", epoch)
                break

    ckpt = torch.load(models_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    log.info("best epoch %d, alpha=%.4f", best_epoch, model.alpha_value)

    with torch.no_grad():
        embeddings = model(x_paper, edge_index, edge_type, time_delta, x_topic).detach()
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

    # Test-set edge-type accuracy
    with torch.no_grad():
        test_type_pred = model.predict_edge_type(embeddings, test_pos).argmax(dim=1).cpu()
        test_type_truth = edge_type[test_idx].cpu()
        type_acc_test = float((test_type_pred == test_type_truth).float().mean())

    metrics = {
        "model": "tier6_hetero_multitask",
        "architecture": "Hetero (paper+topic) multi-task R-GCN + skip + temporal + signed",
        "n_papers": n_papers, "n_topics": n_topics, "n_parameters": n_params,
        "best_epoch": best_epoch, "best_val_auc": round(best_val_auc, 4),
        "final_alpha": round(model.alpha_value, 4),
        "embedding_dim": int(embeddings.shape[1]),
        "silhouette": round(sil, 4), "calinski_harabasz": round(ch, 1),
        "temporal_coherence_rho": round(rho, 4),
        "link_prediction_auc_hard": round(auc_h, 4), "link_prediction_ap_hard": round(ap_h, 4),
        "link_prediction_auc_random": round(auc_r, 4), "link_prediction_ap_random": round(ap_r, 4),
        "edge_type_classification_accuracy": round(type_acc_test, 4),
        "temporal_shuffling": shuf,
    }
    save_metrics(metrics, metrics_dir / "tier6_hetero_multitask.json")
    log.info("=" * 55)
    log.info("TIER 6 HETERO MULTI-TASK COMPLETE")
    log.info("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, dict):
            log.info("%s:", k)
            for kk, vv in v.items(): log.info("  %s = %s", kk, vv)
        else:
            log.info("%s = %s", k, v)


if __name__ == "__main__":
    main()
