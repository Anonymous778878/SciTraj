"""
Phase 9 Tier 2 — Edge aggregation baseline.

For each paper, compute a new embedding as:
  emb_aggregated = alpha * own_embedding + (1-alpha) * weighted_avg(neighbor_embeddings)

Where neighbor weight = edge_weight (similarity * temporal decay) from the graph.

This isolates the contribution of GRAPH STRUCTURE (vs. just SPECTER2 embeddings)
without using a neural network. If Tier 2 beats Tier 1, the graph helps even
without learning. If Tier 3 (R-GCN) beats Tier 2, the neural component helps
beyond what simple aggregation provides.

Why this matters for the paper:
  Three-way comparison (Tier 1 vs Tier 2 vs Tier 3) cleanly separates:
    Tier 1 -> Tier 2 : the value of graph structure
    Tier 2 -> Tier 3 : the value of neural transformation

Output: outputs/metrics/tier2_aggregation.json
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

log = get_logger("tier2")

ALPHA = 0.5  # weight of original embedding vs. neighbor average


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    graph_path = Path(cfg["paths"]["graph_dir"]) / "graph_data.pt"

    log.info("loading graph from %s", graph_path)
    data = load_graph(str(graph_path))
    n_papers = int(data["paper"].num_nodes)
    log.info("graph: %d papers", n_papers)

    # ── Aggregate ──
    abstract_embs = data["paper"].x_abstract            # (N, 768)
    edge_index = data["paper", "trajectory", "paper"].edge_index   # (2, E)
    edge_attr  = data["paper", "trajectory", "paper"].edge_attr    # (E, 5)
    edge_weight = edge_attr[:, 4]                       # column 4 is our weight

    log.info("aggregating %d edges into neighbor sums", edge_index.shape[1])

    # For each target paper, sum incoming weighted neighbor embeddings.
    # The semantics here: a paper's representation incorporates its
    # incoming citations / influences (papers that point TO it).
    #
    # Implementation: for edge (src, tgt), add weight*embs[src] to
    # accumulator[tgt], plus track total weight per tgt for normalization.
    accumulator = torch.zeros_like(abstract_embs)
    weight_sum = torch.zeros(n_papers)

    src = edge_index[0]
    tgt = edge_index[1]
    weighted = abstract_embs[src] * edge_weight.unsqueeze(1)

    accumulator.index_add_(0, tgt, weighted)
    weight_sum.index_add_(0, tgt, edge_weight)

    # Normalize. Papers with no incoming edges get just their original embedding.
    has_neighbors = weight_sum > 0
    neighbor_avg = torch.zeros_like(abstract_embs)
    neighbor_avg[has_neighbors] = accumulator[has_neighbors] / weight_sum[has_neighbors].unsqueeze(1)

    aggregated = ALPHA * abstract_embs + (1 - ALPHA) * neighbor_avg

    # For papers with no incoming edges (likely source-only nodes from latest years),
    # fall back to the original embedding.
    no_neighbors_mask = ~has_neighbors
    aggregated[no_neighbors_mask] = abstract_embs[no_neighbors_mask]

    log.info("aggregated embeddings: %d / %d papers had neighbors",
             int(has_neighbors.sum()), n_papers)

    # ── Evaluate ──
    train_idx, val_idx, test_idx = prepare_temporal_split(
        data,
        train_year_max=cfg["corpus"]["train_years"][1],
        val_year_max=cfg["corpus"]["val_years"][1],
        test_year_max=cfg["corpus"]["test_years"][1],
        seed=seed,
    )
    log.info("split: train=%d, val=%d, test=%d edges",
             train_idx.numel(), val_idx.numel(), test_idx.numel())

    test_pos = edge_index[:, test_idx]
    rng = random.Random(seed)
    log.info("sampling %d negative edges", test_pos.shape[1])
    test_neg = sample_negative_edges(
        n_papers, positive_edges=edge_index,
        n_negatives=test_pos.shape[1],
        years=data["paper"].year, rng=rng,
    )

    log.info("evaluating clustering")
    sil, ch = evaluate_clustering(aggregated, n_clusters=cfg["graph"]["n_topics"], seed=seed)

    log.info("evaluating temporal coherence")
    rho = evaluate_temporal_coherence(aggregated, data["paper"].year)

    log.info("evaluating link prediction")
    auc, ap = evaluate_link_prediction(aggregated, test_pos, test_neg)

    log.info("evaluating temporal shuffling drop")
    shuf = evaluate_temporal_shuffling_drop(
        aggregated, data["paper"].year, test_pos, test_neg, seed=seed
    )

    metrics = {
        "model":              "tier2_edge_aggregation",
        "alpha":              ALPHA,
        "n_papers":           n_papers,
        "n_papers_with_neighbors": int(has_neighbors.sum()),
        "silhouette":         round(sil, 4),
        "calinski_harabasz":  round(ch, 1),
        "temporal_coherence_rho": round(rho, 4),
        "link_prediction_auc": round(auc, 4),
        "link_prediction_ap":  round(ap, 4),
        "temporal_shuffling": shuf,
    }

    save_metrics(metrics, metrics_dir / "tier2_aggregation.json")
    log.info("=" * 55)
    log.info("TIER 2 AGGREGATION COMPLETE")
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
