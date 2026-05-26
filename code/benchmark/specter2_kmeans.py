"""
Phase 9 Tier 1 — SPECTER2 + KMeans baseline.

No graph. No training. Just take the abstract embedding and cluster it.
This is the lower bound — every GNN tier must beat this on at least temporal
coherence and link prediction. If a GNN tier does NOT beat this, the graph
adds nothing.

Why this matters for the paper:
  Reviewers will ask "but does the graph actually help, vs. just better
  embeddings?" This baseline answers that question by removing the graph
  entirely.

Output:
  outputs/metrics/tier1_baseline.json
"""
import random
from pathlib import Path

import torch

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
from utils import ensure_dir, get_logger, load_config

log = get_logger("tier1")


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    graph_path = Path(cfg["paths"]["graph_dir"]) / "graph_data.pt"

    log.info("loading graph from %s", graph_path)
    data = load_graph(str(graph_path))
    n_papers = int(data["paper"].num_nodes)
    log.info("graph: %d papers", n_papers)

    # Use the abstract embedding directly — no transformation
    embeddings = data["paper"].x_abstract            # (N, 768)
    log.info("using x_abstract embeddings (shape=%s)", tuple(embeddings.shape))

    # Build temporal edge split
    train_idx, val_idx, test_idx = prepare_temporal_split(
        data,
        train_year_max=cfg["corpus"]["train_years"][1],
        val_year_max=cfg["corpus"]["val_years"][1],
        test_year_max=cfg["corpus"]["test_years"][1],
        seed=seed,
    )
    edge_index = data["paper", "trajectory", "paper"].edge_index
    log.info("split: train=%d, val=%d, test=%d edges",
             train_idx.numel(), val_idx.numel(), test_idx.numel())

    # Sample negatives for evaluation
    rng = random.Random(seed)
    n_test = test_idx.numel()
    test_pos = edge_index[:, test_idx]

    log.info("sampling %d negative edges for test", n_test)
    test_neg = sample_negative_edges(
        n_papers,
        positive_edges=edge_index,
        n_negatives=n_test,
        years=data["paper"].year,
        rng=rng,
    )

    # ── Evaluate ──
    log.info("evaluating clustering")
    sil, ch = evaluate_clustering(embeddings, n_clusters=cfg["graph"]["n_topics"], seed=seed)

    log.info("evaluating temporal coherence")
    rho = evaluate_temporal_coherence(embeddings, data["paper"].year)

    log.info("evaluating link prediction")
    auc, ap = evaluate_link_prediction(embeddings, test_pos, test_neg)

    log.info("evaluating temporal shuffling drop")
    shuf = evaluate_temporal_shuffling_drop(
        embeddings, data["paper"].year, test_pos, test_neg, seed=seed
    )

    metrics = {
        "model":              "tier1_specter2_kmeans",
        "embedding_dim":      int(embeddings.shape[1]),
        "n_papers":           n_papers,
        "silhouette":         round(sil, 4),
        "calinski_harabasz":  round(ch, 1),
        "temporal_coherence_rho": round(rho, 4),
        "link_prediction_auc": round(auc, 4),
        "link_prediction_ap":  round(ap, 4),
        "temporal_shuffling": shuf,
        "n_train_edges":      int(train_idx.numel()),
        "n_val_edges":        int(val_idx.numel()),
        "n_test_edges":       int(test_idx.numel()),
    }

    save_metrics(metrics, metrics_dir / "tier1_baseline.json")
    log.info("=" * 55)
    log.info("TIER 1 BASELINE COMPLETE")
    log.info("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, dict):
            log.info("%s:", k)
            for kk, vv in v.items():
                log.info("  %s = %s", kk, vv)
        else:
            log.info("%s = %s", k, v)


if __name__ == "__main__":
    main()
