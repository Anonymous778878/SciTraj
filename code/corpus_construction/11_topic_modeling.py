"""
Phase 6.1 — Topic partitioning with soft membership.

We use a simple but effective approach:
  1. KMeans clustering on the abstract embeddings (already L2-normalized)
  2. Compute soft membership as 1 / distance-to-each-cluster, normalized

Why not BERTopic? BERTopic is great but adds heavy dependencies (UMAP, HDBSCAN)
and its discovered topic count is unstable across runs. KMeans gives us:
  - Deterministic results given a fixed seed
  - Exactly the configured number of topics
  - Cosine-distance-based clustering (since embeddings are L2-normalized)

The soft membership is what enables cross-topic edges in Phase 6.3.

Output:
  data/graph/topic_assignments.json    — paper_id -> hard topic + top-k soft
  data/graph/topic_centroids.npy       — (n_topics, dim) — for inspection
  data/graph/topic_descriptors.json    — top representative papers per topic
"""
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from utils import ensure_dir, get_logger, load_config, load_json, save_json

log = get_logger("phase6.1")


def soft_membership(embs: np.ndarray, centroids: np.ndarray, top_k: int = 5) -> list:
    """
    Compute soft topic membership for every paper.
    Returns a list of [(topic_id, weight), ...] per paper, top-k entries.

    Weight is computed as softmax over similarities so it sums to 1.0.
    """
    # Cosine similarity = dot product because both are L2-normalized
    sims = embs @ centroids.T          # (N, n_topics)

    # Softmax-style normalization with temperature (sharper than raw similarity)
    temperature = 5.0                  # higher = sharper
    exp_sims = np.exp(temperature * sims)
    weights = exp_sims / exp_sims.sum(axis=1, keepdims=True)

    out = []
    for i in range(embs.shape[0]):
        # Top-k topics for this paper
        top_idx = np.argsort(-weights[i])[:top_k]
        out.append([(int(t), float(weights[i, t])) for t in top_idx])
    return out


def main():
    cfg = load_config()
    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    graph_dir = ensure_dir(cfg["paths"]["graph_dir"])

    n_topics = cfg["graph"]["n_topics"]
    seed = cfg["project"]["seed"]

    # Load papers and abstract embeddings (deterministic order from Phase 4)
    papers = load_json(Path(cfg["paths"]["validated_dir"]) / "signals_with_reliability.json")
    papers.sort(key=lambda p: p["paper_id"])
    paper_ids = [p["paper_id"] for p in papers]

    saved_order = load_json(emb_dir / "paper_id_order.json")
    if saved_order != paper_ids:
        raise RuntimeError("paper_id order mismatch between Phase 3 and Phase 4")

    embs = np.load(emb_dir / "abstract_embs_norm.npy").astype(np.float32)
    log.info("loaded %d abstract embeddings (dim=%d)", embs.shape[0], embs.shape[1])

    # Run KMeans
    log.info("clustering into %d topics (seed=%d)", n_topics, seed)
    km = KMeans(n_clusters=n_topics, random_state=seed, n_init=10, verbose=0)
    hard_labels = km.fit_predict(embs)

    # Re-normalize centroids so cosine similarity holds
    centroids = km.cluster_centers_
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_norm = centroids / np.maximum(centroid_norms, 1e-12)

    # Soft membership
    log.info("computing soft topic membership")
    soft = soft_membership(embs, centroids_norm.astype(np.float32), top_k=5)

    # Per-topic top representative papers (closest to centroid)
    log.info("identifying topic descriptors")
    descriptors = {}
    for t in range(n_topics):
        mask = hard_labels == t
        if mask.sum() == 0:
            descriptors[t] = {"size": 0, "representatives": []}
            continue
        member_ids = np.where(mask)[0]
        member_embs = embs[member_ids]
        sims_to_centroid = member_embs @ centroids_norm[t]
        top = member_ids[np.argsort(-sims_to_centroid)[:5]]
        descriptors[t] = {
            "size": int(mask.sum()),
            "representatives": [
                {
                    "paper_id": int(papers[i]["paper_id"]),
                    "title":    papers[i]["title"][:80],
                    "year":     papers[i]["year"],
                    "venue":    papers[i]["venue"],
                }
                for i in top
            ],
        }

    # Persist
    assignments = []
    for i, p in enumerate(papers):
        assignments.append({
            "paper_id":     p["paper_id"],
            "hard_topic":   int(hard_labels[i]),
            "soft_topics":  soft[i],          # list of (topic_id, weight)
        })
    save_json(assignments, graph_dir / "topic_assignments.json")
    save_json(descriptors, graph_dir / "topic_descriptors.json")
    np.save(graph_dir / "topic_centroids.npy", centroids_norm)

    log.info("=" * 55)
    log.info("PHASE 6.1 COMPLETE")
    log.info("=" * 55)
    log.info("topics produced:")
    sizes = sorted([(t, descriptors[t]["size"]) for t in range(n_topics)],
                   key=lambda x: -x[1])
    for t, sz in sizes[:10]:
        title = descriptors[t]["representatives"][0]["title"] if descriptors[t]["representatives"] else "?"
        log.info("  topic %2d  size=%4d  e.g. \"%s\"", t, sz, title[:60])
    log.info("...")
    log.info("largest:  %d papers", sizes[0][1])
    log.info("smallest: %d papers", sizes[-1][1])


if __name__ == "__main__":
    main()
