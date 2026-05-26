"""
Phase 6.4 — Assemble PyTorch Geometric HeteroData graph.

Combines everything into a single torch.save-able object ready for GNN training.

Node types:
  paper      : the main entity, with rich feature vector
  topic      : 30 topic nodes, used for soft membership encoding

Node features for each paper:
  abstract_embedding  (768-dim) : SPECTER2 abstract embedding
  signal_embedding    (768-dim) : signal-aware embedding
  year                (1-dim)   : publication year (raw)
  year_norm           (1-dim)   : normalized to [0, 1] over corpus year range
  tpe                 (32-dim)  : sinusoidal temporal positional encoding
  venue_onehot        (3-dim)   : ACL / NeurIPS / CVPR
  quality_score       (1-dim)   : Phase 1.2 quality
  signal_richness     (1-dim)   : Phase 2 richness
  topic_membership    (n_topics): soft topic membership distribution
  has_limit           (1-dim)   : binary
  has_future          (1-dim)   : binary
  has_dispute         (1-dim)   : binary

Edge attributes:
  edge_type_id        (long)
  confidence          (float)   : signed for dispute edges (negative)
  abstract_sim        (float)
  topic_sim           (float)
  time_delta          (float)
  cross_venue         (long)
  shared_authors      (long)
  edge_weight         (float)   : confidence * temporal_decay(time_delta)

Output:
  data/graph/graph_data.pt          PyTorch HeteroData
  data/graph/graph_summary.json     Human-readable summary
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import HeteroData

from utils import ensure_dir, get_logger, load_config, load_json, save_json

log = get_logger("phase6.4")


VENUE_TO_IDX = {"ACL": 0, "NeurIPS": 1, "CVPR": 2}


def temporal_positional_encoding(year_norm: np.ndarray, dim: int = 32) -> np.ndarray:
    """
    Sinusoidal positional encoding over publication year, analogous to
    transformer position encodings but over time instead of token position.
    year_norm: (N,) values in [0, 1].
    Returns: (N, dim).
    """
    n = year_norm.shape[0]
    out = np.zeros((n, dim), dtype=np.float32)
    pos = year_norm[:, None]                          # (N, 1)
    # Frequencies log-spaced for the dim-dimensional encoding
    div = np.exp(-np.log(10000.0) * (2 * np.arange(dim // 2) / dim))
    out[:, 0::2] = np.sin(pos * div)
    out[:, 1::2] = np.cos(pos * div)
    return out


def main():
    cfg = load_config()
    graph_dir = Path(cfg["paths"]["graph_dir"])
    emb_dir   = Path(cfg["paths"]["embeddings_dir"])
    val_dir   = Path(cfg["paths"]["validated_dir"])

    n_topics = cfg["graph"]["n_topics"]
    decay    = cfg["graph"]["temporal_decay"]
    year_min = cfg["corpus"]["year_min"]
    year_max = cfg["corpus"]["year_max"]

    # Load papers (validated signals = our authoritative paper records)
    papers = load_json(val_dir / "signals_with_reliability.json")
    papers.sort(key=lambda p: p["paper_id"])
    paper_ids = [p["paper_id"] for p in papers]

    # Verify Phase 4 row order
    saved_order = load_json(emb_dir / "paper_id_order.json")
    if saved_order != paper_ids:
        raise RuntimeError("paper_id row order mismatch with Phase 4")

    # Load embeddings
    log.info("loading embeddings")
    abstract_embs = np.load(emb_dir / "abstract_embs_norm.npy").astype(np.float32)
    signal_embs   = np.load(emb_dir / "signal_embs_norm.npy").astype(np.float32)

    # Load topic assignments
    topic_assignments = load_json(graph_dir / "topic_assignments.json")
    topic_assignments.sort(key=lambda t: t["paper_id"])

    # Load MMR-selected edges
    edges = load_json(graph_dir / "edges_mmr.json")

    # ───────────────────────── Build node features ─────────────────────────

    n = len(papers)
    log.info("building node features for %d papers", n)

    # Year and year-normalized
    years = np.array([p["year"] for p in papers], dtype=np.float32)
    year_norm = (years - year_min) / max(year_max - year_min, 1)

    # Temporal positional encoding (32-dim)
    tpe = temporal_positional_encoding(year_norm, dim=32)

    # Venue one-hot
    venue_onehot = np.zeros((n, 3), dtype=np.float32)
    for i, p in enumerate(papers):
        v = VENUE_TO_IDX.get(p["venue"], 0)
        venue_onehot[i, v] = 1.0

    # Quality and richness
    quality = np.array([p.get("quality_score", 0) for p in papers], dtype=np.float32).reshape(-1, 1)
    richness = np.array([p.get("signal_richness", 0) for p in papers], dtype=np.float32).reshape(-1, 1)

    # Soft topic membership (dense, n × n_topics)
    topic_membership = np.zeros((n, n_topics), dtype=np.float32)
    for i, ta in enumerate(topic_assignments):
        for t, w in ta["soft_topics"]:
            if 0 <= t < n_topics:
                topic_membership[i, t] = w

    # Presence flags
    has_limit   = np.array([1 if (p.get("limit_text", "") or "").strip()  else 0 for p in papers], dtype=np.float32).reshape(-1, 1)
    has_future  = np.array([1 if (p.get("future_text", "") or "").strip() else 0 for p in papers], dtype=np.float32).reshape(-1, 1)
    has_dispute = np.array([1 if (p.get("dispute_text", "") or "").strip()else 0 for p in papers], dtype=np.float32).reshape(-1, 1)

    # ───────────────────────── Build edges ─────────────────────────

    log.info("building %d edges", len(edges))

    # Build paper_id -> row index
    pid_to_row = {pid: i for i, pid in enumerate(paper_ids)}

    src_idx = np.array([pid_to_row[e["src"]] for e in edges], dtype=np.int64)
    tgt_idx = np.array([pid_to_row[e["tgt"]] for e in edges], dtype=np.int64)

    edge_type_id     = np.array([e["edge_type_id"] for e in edges], dtype=np.int64)
    confidence       = np.array([e["confidence"] for e in edges], dtype=np.float32)
    abstract_sim     = np.array([e["abstract_sim"] for e in edges], dtype=np.float32)
    topic_sim        = np.array([e["topic_sim"] for e in edges], dtype=np.float32)
    time_delta       = np.array([e["time_delta"] for e in edges], dtype=np.float32)
    cross_venue_arr  = np.array([1 if e["cross_venue"] else 0 for e in edges], dtype=np.int64)
    shared_authors   = np.array([e["shared_authors"] for e in edges], dtype=np.int64)

    # Edge weight: similarity * exp(-lambda * time_delta) per edge type
    type_id_to_name = {0: "direct_extension", 1: "future_realized", 2: "limit_addressed",
                       3: "causal_extension", 4: "performance_successor", 5: "method_reuse",
                       6: "temporal_semantic", 7: "related_work", 8: "dispute"}
    decay_per_edge = np.array([decay.get(type_id_to_name[t], 0.3) for t in edge_type_id], dtype=np.float32)
    edge_weight = np.abs(confidence) * np.exp(-decay_per_edge * time_delta)

    # ───────────────────────── Assemble HeteroData ─────────────────────────

    log.info("assembling PyTorch Geometric HeteroData")
    data = HeteroData()

    # Per-feature tensors on paper node
    data["paper"].x_abstract       = torch.from_numpy(abstract_embs)         # (N, 768)
    data["paper"].x_signal         = torch.from_numpy(signal_embs)           # (N, 768)
    data["paper"].year             = torch.from_numpy(years)                 # (N,)
    data["paper"].year_norm        = torch.from_numpy(year_norm)             # (N,)
    data["paper"].tpe              = torch.from_numpy(tpe)                   # (N, 32)
    data["paper"].venue_onehot     = torch.from_numpy(venue_onehot)          # (N, 3)
    data["paper"].quality_score    = torch.from_numpy(quality)               # (N, 1)
    data["paper"].signal_richness  = torch.from_numpy(richness)              # (N, 1)
    data["paper"].topic_membership = torch.from_numpy(topic_membership)      # (N, n_topics)
    data["paper"].has_limit        = torch.from_numpy(has_limit)             # (N, 1)
    data["paper"].has_future       = torch.from_numpy(has_future)            # (N, 1)
    data["paper"].has_dispute      = torch.from_numpy(has_dispute)           # (N, 1)
    data["paper"].paper_id         = torch.tensor(paper_ids, dtype=torch.long)
    data["paper"].num_nodes        = n

    # Concatenate the most useful features into a single x for convenience
    # GNNs that don't want to manage per-attribute features can use this directly.
    x_all = np.concatenate([
        abstract_embs,                           # 768
        tpe,                                     # 32
        venue_onehot,                            # 3
        quality / 100.0,                         # 1 (normalize 0-100 -> 0-1)
        np.tanh(richness / 5.0),                 # 1 (compress)
        topic_membership,                        # n_topics
        has_limit, has_future, has_dispute,      # 3
    ], axis=1).astype(np.float32)
    data["paper"].x = torch.from_numpy(x_all)
    log.info("  paper.x shape: %s (concatenated feature)", tuple(data["paper"].x.shape))

    # Edges — store all edge types in a single relation 'paper -> trajectory -> paper'
    # Edge type is stored as an attribute, so R-GCN can route in Phase 9.
    edge_index = torch.from_numpy(np.vstack([src_idx, tgt_idx]))             # (2, E)
    data["paper", "trajectory", "paper"].edge_index = edge_index
    data["paper", "trajectory", "paper"].edge_type  = torch.from_numpy(edge_type_id)
    data["paper", "trajectory", "paper"].edge_attr  = torch.from_numpy(np.stack([
        confidence, abstract_sim, topic_sim, time_delta, edge_weight,
    ], axis=1).astype(np.float32))
    data["paper", "trajectory", "paper"].cross_venue    = torch.from_numpy(cross_venue_arr)
    data["paper", "trajectory", "paper"].shared_authors = torch.from_numpy(shared_authors)
    log.info("  edge_index shape: %s", tuple(edge_index.shape))

    # Topic node block (small, useful for downstream tasks)
    topic_centroids = np.load(graph_dir / "topic_centroids.npy").astype(np.float32)
    data["topic"].x = torch.from_numpy(topic_centroids)                      # (n_topics, 768)
    data["topic"].num_nodes = n_topics

    # Save
    out_path = graph_dir / "graph_data.pt"
    torch.save(data, out_path)
    log.info("wrote %s", out_path)

    # Human-readable summary
    summary = {
        "num_papers":          n,
        "num_topics":          n_topics,
        "num_edges":           len(edges),
        "feature_dim":         int(data["paper"].x.shape[1]),
        "avg_out_degree":      round(len(edges) / n, 2),
        "by_edge_type":        {type_id_to_name[t]: int((edge_type_id == t).sum()) for t in range(9)},
        "cross_venue_edges":   int(cross_venue_arr.sum()),
        "author_cont_edges":   int((shared_authors > 0).sum()),
        "year_range":          [int(years.min()), int(years.max())],
        "venue_counts": {
            "ACL":     int((venue_onehot[:, 0] == 1).sum()),
            "NeurIPS": int((venue_onehot[:, 1] == 1).sum()),
            "CVPR":    int((venue_onehot[:, 2] == 1).sum()),
        },
    }
    save_json(summary, graph_dir / "graph_summary.json")

    log.info("=" * 55)
    log.info("PHASE 6 COMPLETE — GRAPH ASSEMBLED")
    log.info("=" * 55)
    log.info("nodes:   %d papers, %d topics", n, n_topics)
    log.info("edges:   %d  (avg out-degree %.1f)", len(edges), summary["avg_out_degree"])
    log.info("feature dim: %d", summary["feature_dim"])
    log.info("by edge type:")
    for et, c in summary["by_edge_type"].items():
        if c > 0:
            log.info("  %-22s  %7d", et, c)


if __name__ == "__main__":
    main()
