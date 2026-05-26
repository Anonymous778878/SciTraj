"""
t22_features.py (v6.1)

Exact mirror of the feature extractor in src/67_tier22_pair_mlp_expanded.py.

CRITICAL DIFFERENCES FROM v6:
  1. Caller must pass `paper_emb` from `graph['paper'].x_abstract` (not
     abstract_embs.npy — they may differ).
  2. Caller must pass `edge_weight` from
     `graph[('paper','trajectory','paper')].edge_attr[:, 4]` (not None).
  3. Features must be standardized at inference using a StandardScaler
     fit on training pair features. Use `fit_scaler_from_train_pairs()`.

Use:
    extractor = T22FeatureExtractor.from_data(
        paper_emb=paper_emb_from_graph,           # graph['paper'].x_abstract
        edge_index=edge_index_np,
        edge_type=edge_type_np,
        year_arr=year_arr,
        topic_arr=topic_arr,
        edge_weight=edge_weight_np,               # edge_attr[:, 4]
    )

    # Fit scaler on the training pairs (mirror of training-time scaler)
    scaler = extractor.fit_scaler_from_train_pairs(
        train_pos_s, train_pos_t, train_neg_s, train_neg_t
    )

    # At inference, transform features through the scaler
    raw_feats = extractor.featurize_pairs(s_arr, t_arr)
    feats = scaler.transform(raw_feats).astype(np.float32)
"""

import time

import numpy as np


N_RELATIONS = 9


def per_relation_aggregations(paper_emb_np, edge_index_np, edge_type_np,
                              edge_weight_np, n_relations=N_RELATIONS):
    """One T2-style L2-normalised aggregated embedding per relation.

    Returns array (R, N, D).
    """
    n, d = paper_emb_np.shape
    out = []
    src = edge_index_np[0]
    tgt = edge_index_np[1]
    for r in range(n_relations):
        mask = (edge_type_np == r)
        agg = np.zeros((n, d), dtype=np.float32)
        count = np.zeros(n, dtype=np.float32)
        if mask.sum() > 0:
            sr = src[mask]
            tg = tgt[mask]
            w = edge_weight_np[mask]
            for i in range(len(sr)):
                agg[tg[i]] += paper_emb_np[sr[i]] * w[i]
                count[tg[i]] += w[i]
            has = count > 0
            agg[has] /= count[has, None]
            agg[~has] = paper_emb_np[~has]
        else:
            agg = paper_emb_np.copy()
        norm = np.linalg.norm(agg, axis=1, keepdims=True) + 1e-8
        out.append(agg / norm)
    return np.stack(out, axis=0)


def per_relation_degrees(edge_index_np, edge_type_np, n_papers,
                          n_relations=N_RELATIONS):
    in_deg = np.zeros((n_papers, n_relations), dtype=np.int64)
    out_deg = np.zeros((n_papers, n_relations), dtype=np.int64)
    src = edge_index_np[0]
    tgt = edge_index_np[1]
    for i in range(len(src)):
        r = int(edge_type_np[i])
        if 0 <= r < n_relations:
            in_deg[tgt[i], r] += 1
            out_deg[src[i], r] += 1
    return in_deg, out_deg


def build_neighbour_sets(edge_index_np, n_papers):
    src = edge_index_np[0]
    tgt = edge_index_np[1]
    neigh = [set() for _ in range(n_papers)]
    for i in range(len(src)):
        a = int(src[i])
        b = int(tgt[i])
        neigh[a].add(b)
        neigh[b].add(a)
    deg = np.array([len(s) for s in neigh], dtype=np.float32)
    return neigh, deg


def common_neighbour_features(s_arr, t_arr, neigh, deg):
    n_pairs = len(s_arr)
    aa = np.zeros(n_pairs, dtype=np.float32)
    log_aa = np.zeros(n_pairs, dtype=np.float32)
    jacc = np.zeros(n_pairs, dtype=np.float32)
    common = np.zeros(n_pairs, dtype=np.float32)
    pref = np.zeros(n_pairs, dtype=np.float32)
    union_sz = np.zeros(n_pairs, dtype=np.float32)

    for i in range(n_pairs):
        a = int(s_arr[i])
        b = int(t_arr[i])
        ns_a = neigh[a]
        ns_b = neigh[b]
        if len(ns_a) == 0 or len(ns_b) == 0:
            continue
        inter = ns_a & ns_b
        union = ns_a | ns_b
        common[i] = len(inter)
        if len(union) > 0:
            jacc[i] = len(inter) / len(union)
        union_sz[i] = len(union)
        s_aa = 0.0
        for c in inter:
            d = deg[c]
            if d > 1:
                s_aa += 1.0 / np.log(d)
        aa[i] = s_aa
        log_aa[i] = np.log1p(s_aa)
        pref[i] = deg[a] * deg[b]
    return np.stack([aa, log_aa, jacc, common, pref, union_sz], axis=1)


class T22FeatureExtractor:
    def __init__(self, paper_emb, per_rel_aggs, in_deg_per_rel,
                 out_deg_per_rel, year_arr, topic_arr, neigh, deg,
                 edge_set_lookup):
        self.paper_emb = paper_emb
        self.per_rel_aggs = per_rel_aggs
        self.in_deg = in_deg_per_rel
        self.out_deg = out_deg_per_rel
        self.year_arr = year_arr
        self.topic_arr = topic_arr
        self.neigh = neigh
        self.deg = deg
        self.edge_set = edge_set_lookup

    @classmethod
    def from_data(cls, paper_emb, edge_index, edge_type, year_arr, topic_arr,
                   edge_weight=None, verbose=True):
        n_papers = paper_emb.shape[0]
        n_edges = edge_index.shape[1]
        if edge_weight is None:
            print("  WARNING: edge_weight is None; defaulting to ones. "
                  "Per-relation aggregations may not match training!")
            edge_weight = np.ones(n_edges, dtype=np.float32)

        if verbose:
            print(f"[T22FeatureExtractor] precomputing ({n_papers} nodes, "
                  f"{n_edges} edges)...")
            tic = time.time()

        if verbose:
            print(f"  per-relation aggregations (with edge weights)...")
        per_rel_aggs = per_relation_aggregations(
            paper_emb, edge_index, edge_type, edge_weight
        )

        if verbose:
            print(f"  per-relation degrees...")
        in_deg_per_rel, out_deg_per_rel = per_relation_degrees(
            edge_index, edge_type, n_papers
        )

        if verbose:
            print(f"  neighbour sets...")
        neigh, deg = build_neighbour_sets(edge_index, n_papers)

        if verbose:
            print(f"  edge-set lookup...")
        edge_set = set()
        for i in range(n_edges):
            edge_set.add((int(edge_index[0, i]), int(edge_index[1, i])))

        if verbose:
            print(f"  done ({time.time()-tic:.1f}s)")

        return cls(
            paper_emb, per_rel_aggs, in_deg_per_rel, out_deg_per_rel,
            year_arr, topic_arr, neigh, deg, edge_set
        )

    def featurize_pairs(self, s_arr, t_arr):
        """Returns RAW (unstandardized) 63-dim features."""
        s = np.asarray(s_arr, dtype=np.int64)
        t = np.asarray(t_arr, dtype=np.int64)
        n = len(s)

        feats = []

        # Group 1: per-relation T2 cosines (9 dims)
        for r in range(self.per_rel_aggs.shape[0]):
            a = self.per_rel_aggs[r][s]
            b = self.per_rel_aggs[r][t]
            feats.append((a * b).sum(axis=1).astype(np.float32))

        # Group 2: per-relation degrees (4*9 = 36 dims)
        for r in range(self.in_deg.shape[1]):
            feats.append(np.log1p(self.in_deg[s, r]).astype(np.float32))
            feats.append(np.log1p(self.in_deg[t, r]).astype(np.float32))
            feats.append(np.log1p(self.out_deg[s, r]).astype(np.float32))
            feats.append(np.log1p(self.out_deg[t, r]).astype(np.float32))

        # Group 3: common-neighbour features (6 dims)
        cn = common_neighbour_features(s, t, self.neigh, self.deg)
        for k in range(cn.shape[1]):
            feats.append(cn[:, k])

        # Group 4: topic match (2 dims)
        feats.append((self.topic_arr[s] == self.topic_arr[t]).astype(np.float32))
        feats.append((self.topic_arr[s] != self.topic_arr[t]).astype(np.float32))

        # Group 5: year-gap basis (5 dims)
        yg = (self.year_arr[t] - self.year_arr[s]).astype(np.float32)
        feats.append(yg)
        feats.append(np.abs(yg))
        feats.append(np.log1p(np.abs(yg)))
        feats.append(np.maximum(yg, 0))
        feats.append(np.minimum(yg, 0))

        # Group 6: back-edge and bidirectional (2 dims)
        has_back = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if (int(t[i]), int(s[i])) in self.edge_set:
                has_back[i] = 1.0
        feats.append(has_back)
        bidirectional = np.zeros(n, dtype=np.float32)
        for i in range(n):
            if ((int(s[i]), int(t[i])) in self.edge_set and
                    (int(t[i]), int(s[i])) in self.edge_set):
                bidirectional[i] = 1.0
        feats.append(bidirectional)

        # Group 7: source-target ratios (3 dims)
        in_t_total = self.in_deg[t].sum(axis=1).astype(np.float32)
        in_s_total = self.in_deg[s].sum(axis=1).astype(np.float32)
        out_t_total = self.out_deg[t].sum(axis=1).astype(np.float32)
        out_s_total = self.out_deg[s].sum(axis=1).astype(np.float32)
        feats.append(np.log1p(in_t_total) - np.log1p(in_s_total))
        feats.append(np.log1p(out_s_total) - np.log1p(out_t_total))
        feats.append(np.log1p(in_t_total + out_t_total))

        return np.stack(feats, axis=1)

    def fit_scaler_from_train_pairs(self, train_pos_s, train_pos_t,
                                      train_neg_s, train_neg_t,
                                      verbose=True):
        """Fit a StandardScaler on training pair features.

        This mirrors the training-time scaler. The model was trained on
        scaled features, so inference must apply the same scaler.

        Returns sklearn StandardScaler instance.
        """
        from sklearn.preprocessing import StandardScaler

        if verbose:
            print(f"  Fitting StandardScaler on "
                  f"{len(train_pos_s) + len(train_neg_s)} training pairs...")
            tic = time.time()

        X_pos = self.featurize_pairs(train_pos_s, train_pos_t)
        X_neg = self.featurize_pairs(train_neg_s, train_neg_t)
        scaler = StandardScaler().fit(np.vstack([X_pos, X_neg]))

        if verbose:
            print(f"    done ({time.time()-tic:.0f}s)")
            print(f"    Scaler mean range: [{scaler.mean_.min():.3f}, "
                  f"{scaler.mean_.max():.3f}]")
            print(f"    Scaler scale range: [{scaler.scale_.min():.3f}, "
                  f"{scaler.scale_.max():.3f}]")

        return scaler
