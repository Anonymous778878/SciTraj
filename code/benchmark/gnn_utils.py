"""
Shared utilities for GNN training (Phase 9).

What this module provides:
  - load_graph()              : load HeteroData from Phase 6.4
  - prepare_temporal_split()  : build train/val/test edge masks with strict temporal ordering
  - sample_negative_edges()   : random negative pairs respecting temporal constraint
  - evaluate_link_prediction(): AUC + AP for binary link prediction
  - evaluate_clustering()     : silhouette + Calinski-Harabasz on learned embeddings
  - evaluate_temporal_coherence(): does node ordering reflect publication time?

These are reused across every GNN tier so all results are comparable.
"""
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    roc_auc_score,
    silhouette_score,
)


def load_graph(graph_path: str) -> 'torch_geometric.data.HeteroData':
    """Load the Phase 6.4 HeteroData object."""
    return torch.load(graph_path, weights_only=False)


def prepare_temporal_split(
    data,
    train_year_max: int,
    val_year_max: int,
    test_year_max: int,
    seed: int = 42,
):
    """
    Build edge masks for temporal train/val/test split.
      train: edges where source year <= train_year_max (e.g. 2020)
      val:   edges where source year in (train_year_max, val_year_max] (e.g. 2021-2022)
      test:  edges where source year in (val_year_max, test_year_max] (e.g. 2023-2024)

    A test edge is a (src, tgt) pair where the model must predict whether they
    are connected — the source paper exists at test time, the target may be
    older but the link prediction still tests learned relational structure.

    Returns: (train_idx, val_idx, test_idx) as LongTensor of edge indices.
    """
    edge_index = data["paper", "trajectory", "paper"].edge_index
    years = data["paper"].year

    src_years = years[edge_index[0]]
    train_mask = src_years <= train_year_max
    val_mask   = (src_years > train_year_max) & (src_years <= val_year_max)
    test_mask  = (src_years > val_year_max) & (src_years <= test_year_max)

    train_idx = train_mask.nonzero(as_tuple=False).squeeze()
    val_idx   = val_mask.nonzero(as_tuple=False).squeeze()
    test_idx  = test_mask.nonzero(as_tuple=False).squeeze()

    return train_idx, val_idx, test_idx


def sample_negative_edges(
    n_papers: int,
    positive_edges: torch.Tensor,    # (2, E)
    n_negatives: int,
    years: torch.Tensor,              # (N,)
    rng: random.Random,
):
    """
    Sample random (src, tgt) pairs that are NOT in positive_edges and where
    year(tgt) > year(src). Returns LongTensor of shape (2, n_negatives).
    """
    pos_set = set()
    for i in range(positive_edges.shape[1]):
        pos_set.add((int(positive_edges[0, i]), int(positive_edges[1, i])))

    negatives = []
    attempts = 0
    max_attempts = n_negatives * 10
    while len(negatives) < n_negatives and attempts < max_attempts:
        attempts += 1
        s = rng.randrange(n_papers)
        t = rng.randrange(n_papers)
        if s == t:
            continue
        if int(years[s]) >= int(years[t]):
            continue
        if (s, t) in pos_set:
            continue
        negatives.append((s, t))

    if len(negatives) < n_negatives:
        # Pad with random-without-temporal-check pairs if we couldn't sample enough
        for _ in range(n_negatives - len(negatives)):
            negatives.append((rng.randrange(n_papers), rng.randrange(n_papers)))

    arr = np.array(negatives, dtype=np.int64).T
    return torch.from_numpy(arr)


def evaluate_link_prediction(
    embeddings: torch.Tensor,         # (N, D)
    pos_edges: torch.Tensor,          # (2, E_pos)
    neg_edges: torch.Tensor,          # (2, E_neg)
):
    """
    Score each edge by dot product of its endpoint embeddings.
    Compute ROC-AUC and Average Precision against (positive, negative) labels.
    """
    embs = F.normalize(embeddings, dim=1)

    pos_scores = (embs[pos_edges[0]] * embs[pos_edges[1]]).sum(dim=1)
    neg_scores = (embs[neg_edges[0]] * embs[neg_edges[1]]).sum(dim=1)

    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = np.concatenate([
        np.ones(pos_scores.shape[0]),
        np.zeros(neg_scores.shape[0]),
    ])

    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)
    return float(auc), float(ap)


def evaluate_clustering(
    embeddings: torch.Tensor,
    n_clusters: int = 30,
    seed: int = 42,
    sample_size: int = 5000,
):
    """
    Cluster embeddings with KMeans, compute silhouette and Calinski-Harabasz.
    Subsamples for silhouette since it is O(N^2). CH index is full-population.
    """
    embs = embeddings.cpu().numpy()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5)
    labels = km.fit_predict(embs)

    n = embs.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n, sample_size, replace=False)
        sil_embs = embs[sample_idx]
        sil_labels = labels[sample_idx]
    else:
        sil_embs = embs
        sil_labels = labels

    sil = float(silhouette_score(sil_embs, sil_labels, metric="cosine"))
    ch  = float(calinski_harabasz_score(embs, labels))
    return sil, ch


def evaluate_temporal_coherence(
    embeddings: torch.Tensor,
    years: torch.Tensor,
    n_pairs: int = 50_000,
    seed: int = 42,
):
    """
    Temporal coherence: for random (paper_a, paper_b) pairs sampled from
    different topics, the model embedding distance should correlate with
    year difference (further apart in time -> further apart in embedding).

    Returns: spearman_rho (float in [-1, 1])
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    a = rng.integers(0, n, size=n_pairs)
    b = rng.integers(0, n, size=n_pairs)

    # Filter out self-pairs
    keep = a != b
    a, b = a[keep], b[keep]

    embs = F.normalize(embeddings, dim=1).cpu().numpy()
    cos_sim  = (embs[a] * embs[b]).sum(axis=1)
    cos_dist = 1.0 - cos_sim

    year_arr = years.cpu().numpy()
    year_diff = np.abs(year_arr[a] - year_arr[b])

    rho, _ = spearmanr(year_diff, cos_dist)
    return float(rho)


def evaluate_temporal_shuffling_drop(
    embeddings: torch.Tensor,
    years: torch.Tensor,
    pos_edges: torch.Tensor,
    neg_edges: torch.Tensor,
    seed: int = 42,
) -> dict:
    """
    Compare link prediction performance on:
      A) original embeddings against original edges
      B) original embeddings against TEMPORALLY-SHUFFLED positives (same edges,
         but year of source paper randomly reassigned)

    Strong temporal models drop more than weak ones — proves the model uses
    time, not just topical similarity.

    Returns dict with auc_normal, auc_shuffled, drop_auc.
    """
    auc_normal, ap_normal = evaluate_link_prediction(embeddings, pos_edges, neg_edges)

    # Shuffle: replace each positive edge's source with a random source whose
    # year is at least 1 year before the target's year. This breaks the
    # genuine temporal structure but keeps year-validity.
    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    year_arr = years.cpu().numpy()

    new_src = []
    for i in range(pos_edges.shape[1]):
        tgt = int(pos_edges[1, i])
        valid_sources = np.where(year_arr < year_arr[tgt])[0]
        if len(valid_sources) == 0:
            new_src.append(int(pos_edges[0, i]))
        else:
            new_src.append(int(rng.choice(valid_sources)))
    new_src = torch.tensor(new_src, dtype=torch.long)

    shuffled_pos = torch.stack([new_src, pos_edges[1]], dim=0)
    auc_shuffled, _ = evaluate_link_prediction(embeddings, shuffled_pos, neg_edges)

    return {
        "auc_normal":   round(auc_normal, 4),
        "auc_shuffled": round(auc_shuffled, 4),
        "drop_auc":     round(auc_normal - auc_shuffled, 4),
    }


def save_metrics(metrics: dict, path):
    """Save metrics dict to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
