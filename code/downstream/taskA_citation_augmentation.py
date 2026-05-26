"""
91_e10a_citation_augmentation.py (v6.1)

Task A: Citation Augmentation with corrected feature extraction.

Critical fixes from v6:
  1. paper_emb loaded from graph['paper'].x_abstract (not abstract_embs.npy)
  2. edge_weight loaded from edge_attr[:, 4] (not all-ones)
  3. StandardScaler fit on training pairs and applied to inference features
     (matches training-time normalisation)

Sanity check should now pass: re-featurize 200 test_pos pairs, score with
T22, compare to saved test_logits.npz.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_all, load_t22_model, EDGE_TYPE_NAMES
from t22_features import T22FeatureExtractor


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_VALUES = [5, 10, 20, 50, 100]


def venue_to_subfield(venue):
    v = (venue or "").lower()
    if any(s in v for s in ["cvpr", "iccv", "wacv", "bmvc", "eccv"]):
        return "CV"
    if any(s in v for s in ["acl", "emnlp", "naacl", "eacl", "tacl", "coling"]):
        return "NLP"
    if any(s in v for s in ["neurips", "nips", "icml", "iclr", "aaai", "ijcai"]):
        return "ML"
    return "OTHER"


def recall_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def mrr(ranked, gold):
    for r, idx in enumerate(ranked, start=1):
        if idx in gold:
            return 1.0 / r
    return 0.0


def ndcg_at_k(ranked, gold, k):
    if not gold:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 1) for r, idx in enumerate(ranked[:k], 1)
              if idx in gold)
    n_rel = min(len(gold), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def load_topic_assignments(graph_dir, n_papers):
    path = graph_dir / "topic_assignments.json"
    with open(path) as f:
        recs = json.load(f)
    pid_to_topic = {r["paper_id"]: r["hard_topic"] for r in recs}
    return np.array([pid_to_topic.get(i, 0) for i in range(n_papers)],
                     dtype=np.int64)


def extract_real_paper_emb_and_edge_weight(graph):
    """Pull paper_emb and edge_weight from the actual HeteroData object.

    Mirrors training-script logic:
        paper_emb = data["paper"].x_abstract.cpu().numpy()
        edge_weight = data[edge_triple].edge_attr[:, 4].cpu().numpy()
    """
    # Find the paper node store
    if hasattr(graph, "node_types"):
        for nt in graph.node_types:
            if "paper" in str(nt).lower() or nt == "paper":
                store = graph[nt]
                if hasattr(store, "x_abstract"):
                    paper_emb = store.x_abstract.cpu().numpy()
                    print(f"  paper_emb from graph[{nt!r}].x_abstract: "
                          f"shape={paper_emb.shape}")
                    break
        else:
            raise RuntimeError("Could not find 'paper' node store in graph")
    else:
        raise RuntimeError("Graph is not HeteroData")

    # Find the edge store with the trajectory relation
    edge_weight = None
    if hasattr(graph, "edge_types"):
        for et in graph.edge_types:
            store = graph[et]
            if hasattr(store, "edge_attr") and store.edge_attr is not None:
                attr = store.edge_attr.cpu().numpy()
                if attr.ndim == 2 and attr.shape[1] >= 5:
                    edge_weight = attr[:, 4]
                    print(f"  edge_weight from graph[{et}].edge_attr[:, 4]: "
                          f"shape={edge_weight.shape}, "
                          f"range=[{edge_weight.min():.3f}, "
                          f"{edge_weight.max():.3f}]")
                    break

    if edge_weight is None:
        print("  WARNING: could not find edge_attr[:, 4]; defaulting to ones")
        for et in graph.edge_types:
            store = graph[et]
            if hasattr(store, "edge_index"):
                edge_weight = np.ones(store.edge_index.shape[1],
                                       dtype=np.float32)
                break

    return paper_emb, edge_weight


def score_t22_pairs(model, s_arr, t_arr, paper_emb, extras, batch=4096):
    n = len(s_arr)
    scores = np.zeros(n, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch):
            j = min(i + batch, n)
            s = torch.from_numpy(paper_emb[s_arr[i:j]]).float().to(DEVICE)
            t = torch.from_numpy(paper_emb[t_arr[i:j]]).float().to(DEVICE)
            ex = torch.from_numpy(extras[i:j]).float().to(DEVICE)
            scores[i:j] = model(s, t, ex).cpu().numpy()
    return scores


def score_specter2_pairs(s_arr, t_arr, paper_emb):
    s = paper_emb[s_arr]
    t = paper_emb[t_arr]
    s_norm = np.linalg.norm(s, axis=1) + 1e-9
    t_norm = np.linalg.norm(t, axis=1) + 1e-9
    return ((s * t).sum(axis=1) / (s_norm * t_norm)).astype(np.float32)


def sanity_check(extractor, scaler, t22_model, t22_feat_npz,
                  test_logits_path, paper_emb, n_check=200):
    """Featurize test_pos pairs, scale, score, compare to saved logits."""
    print("\n--- Sanity check: re-featurize and compare to saved logits ---")
    if not test_logits_path.exists():
        print(f"  test_logits.npz not found; skipping.")
        return None

    test_logits = np.load(test_logits_path)
    if "pos_logits" not in test_logits:
        print("  no pos_logits key; skipping.")
        return None

    saved = test_logits["pos_logits"][:n_check]
    test_pos_s = t22_feat_npz["test_pos_s"][:n_check].astype(np.int64)
    test_pos_t = t22_feat_npz["test_pos_t"][:n_check].astype(np.int64)

    print(f"  Featurizing {n_check} test pairs...")
    raw_feats = extractor.featurize_pairs(test_pos_s, test_pos_t)
    print(f"  Raw feature range: [{raw_feats.min():.3f}, {raw_feats.max():.3f}]")

    scaled_feats = scaler.transform(raw_feats).astype(np.float32)
    print(f"  Scaled feature range: [{scaled_feats.min():.3f}, "
          f"{scaled_feats.max():.3f}]")

    print(f"  Scoring with T22...")
    our_scores = score_t22_pairs(t22_model, test_pos_s, test_pos_t,
                                   paper_emb, scaled_feats)

    # Also compare scaled features to features.npz directly (best comparison)
    saved_scaled = t22_feat_npz["X_test_pos"][:n_check]
    feat_diff = np.abs(scaled_feats - saved_scaled)
    feat_pearson = np.array([
        np.corrcoef(scaled_feats[:, k], saved_scaled[:, k])[0, 1]
        for k in range(63)
    ])
    print(f"  Feature mean abs diff: {feat_diff.mean():.4f}")
    print(f"  Feature Pearson per dim: min={feat_pearson.min():.3f}, "
          f"mean={feat_pearson.mean():.3f}, max={feat_pearson.max():.3f}")

    diff = our_scores - saved
    abs_diff = np.abs(diff)
    print(f"  Saved logits range: [{saved.min():.4f}, {saved.max():.4f}]")
    print(f"  Our   logits range: [{our_scores.min():.4f}, {our_scores.max():.4f}]")
    print(f"  Mean abs diff: {abs_diff.mean():.4f}")
    print(f"  Pearson correlation: "
          f"{np.corrcoef(saved, our_scores)[0, 1]:.4f}")

    pearson = float(np.corrcoef(saved, our_scores)[0, 1])
    if abs_diff.mean() < 0.1 or pearson > 0.95:
        print(f"  ✓ Re-featurization matches saved logits within tolerance.")
        ok = True
    else:
        print(f"  ✗ Re-featurization disagrees with saved logits.")
        print(f"    First 5 saved:  {saved[:5]}")
        print(f"    First 5 ours:   {our_scores[:5]}")
        ok = False

    return {
        "n_check": int(n_check),
        "mean_abs_diff_logits": float(abs_diff.mean()),
        "pearson_logits": pearson,
        "feature_mean_pearson": float(feat_pearson.mean()),
        "passed": ok,
    }


def sample_year_stratified_hard_negatives(query_id, gold_t_ids, gold_t_years,
                                            paper_emb, edges_by_source,
                                            year_arr, n_papers, k_pool=200,
                                            k_neg=3, year_window=2, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    cited_set = set(t for t, _ in edges_by_source.get(query_id, []))
    cited_set.add(query_id)

    q_emb = paper_emb[query_id]
    sims = paper_emb @ q_emb / (
        (np.linalg.norm(paper_emb, axis=1) + 1e-9) *
        (np.linalg.norm(q_emb) + 1e-9)
    )
    top_pool = np.argsort(-sims)[:k_pool + len(cited_set) + 100]
    top_pool = np.array([p for p in top_pool if int(p) not in cited_set],
                        dtype=np.int64)
    if len(top_pool) == 0:
        return []
    pool_years = year_arr[top_pool]

    negatives = []
    used = set()
    for t_id, y_t in zip(gold_t_ids, gold_t_years):
        in_window = (pool_years >= y_t - year_window) & \
                    (pool_years <= y_t + year_window)
        cands = top_pool[in_window]
        cands = np.array([c for c in cands if int(c) not in used],
                          dtype=np.int64)
        if len(cands) == 0:
            continue
        n_take = min(k_neg, len(cands))
        chosen = rng.choice(len(cands), size=n_take, replace=False)
        for c in cands[chosen]:
            negatives.append((int(c), -1))
            used.add(int(c))

    return negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--query-year-min", type=int, default=2022)
    parser.add_argument("--k-pool", type=int, default=200)
    parser.add_argument("--k-neg-per-gold", type=int, default=3)
    parser.add_argument("--year-window", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-sanity-check", action="store_true")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    data = load_all(load_t22_features=True)
    paths = data["paths"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    corpus = data["corpus"]
    n_papers = data["n_papers"]
    t22_feat_npz = data["t22_features"]
    graph = data["graph"]

    if t22_feat_npz is None:
        print("ERROR: features.npz required.")
        return

    # CRITICAL FIX: load paper_emb and edge_weight from the graph itself,
    # not from .npy files
    print("\n--- Extracting paper_emb and edge_weight from graph (training-time source) ---")
    paper_emb, edge_weight = extract_real_paper_emb_and_edge_weight(graph)

    OUTPUT_PATH = paths["outputs"] / "e10a_citation_augmentation.json"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\nBuilding year array from corpus...")
    year_arr = np.array([corpus[i].get("year", 2020) for i in range(n_papers)],
                        dtype=np.int64)

    print("Loading topic assignments...")
    graph_dir = paths["project_root"] / "data/graph"
    topic_arr = load_topic_assignments(graph_dir, n_papers)
    print(f"  topic_arr: {len(np.unique(topic_arr))} unique topics")

    print("\nBuilding T22FeatureExtractor with REAL edge weights...")
    extractor = T22FeatureExtractor.from_data(
        paper_emb=paper_emb,
        edge_index=edge_index,
        edge_type=edge_type,
        year_arr=year_arr,
        topic_arr=topic_arr,
        edge_weight=edge_weight,    # CRITICAL: real edge weights from graph
    )

    print("\nFitting StandardScaler on training pairs...")
    scaler = extractor.fit_scaler_from_train_pairs(
        t22_feat_npz["train_pos_s"], t22_feat_npz["train_pos_t"],
        t22_feat_npz["train_neg_s"], t22_feat_npz["train_neg_t"],
    )

    print("\nLoading T22 model...")
    t22_model, _ = load_t22_model(paths["t22_model"], DEVICE)

    sanity_result = None
    if not args.skip_sanity_check:
        test_logits_path = (paths["project_root"] /
                            "models/tier22_pair_mlp_expanded/test_logits.npz")
        sanity_result = sanity_check(
            extractor, scaler, t22_model, t22_feat_npz, test_logits_path,
            paper_emb, n_check=200
        )
        if sanity_result and not sanity_result["passed"]:
            print("\n✗ Sanity check FAILED. Aborting.")
            return

    edges_by_source = defaultdict(list)
    for i in range(edge_index.shape[1]):
        s = int(edge_index[0, i])
        t = int(edge_index[1, i])
        et = int(edge_type[i])
        edges_by_source[s].append((t, et))

    rng = np.random.default_rng(args.seed)
    eligible = []
    for p in corpus:
        pid = p["paper_id"]
        if (p.get("year", 0) >= args.query_year_min and
                len(edges_by_source.get(pid, [])) >= 3):
            eligible.append(pid)
    print(f"\nEligible queries (year>={args.query_year_min}, >=3 outgoing): "
          f"{len(eligible)}")

    n = 5 if args.dry_run else min(args.n_queries, len(eligible))
    test_queries = rng.choice(eligible, size=n, replace=False).tolist()
    print(f"Selected {n} test queries (dry_run={args.dry_run})")

    results = {
        "config": {
            "n_queries": n,
            "k_pool": args.k_pool,
            "k_neg_per_gold": args.k_neg_per_gold,
            "year_window": args.year_window,
            "k_values": K_VALUES,
            "seed": args.seed,
            "dry_run": args.dry_run,
            "evaluation_setup": (
                "Year-stratified hard negatives. Features extracted with "
                "training-time edge weights from edge_attr[:, 4] and paper "
                "embeddings from graph['paper'].x_abstract; standardized "
                "via scaler fit on training pairs."
            ),
        },
        "sanity_check": sanity_result,
        "modes": {}
    }

    for mode in ["baseline_specter2_only", "t22"]:
        print(f"\n=== Evaluating mode: {mode} ===")
        per_q = []
        tic = time.time()

        for i, q in enumerate(test_queries):
            if i % 50 == 0 and i > 0:
                el = time.time() - tic
                eta = el / i * (n - i)
                print(f"  Query {i}/{n} ({el:.0f}s, ~{eta:.0f}s remaining)")

            cited = edges_by_source.get(q, [])
            gold_ids = [c[0] for c in cited]
            gold_types = {c[0]: c[1] for c in cited}
            gold_years = [year_arr[g] for g in gold_ids]
            if not gold_ids:
                continue

            negs = sample_year_stratified_hard_negatives(
                q, gold_ids, gold_years, paper_emb, edges_by_source,
                year_arr, n_papers, k_pool=args.k_pool,
                k_neg=args.k_neg_per_gold, year_window=args.year_window,
                rng=rng
            )
            if not negs:
                continue

            cand_t = list(gold_ids) + [n for n, _ in negs]
            s_arr = np.full(len(cand_t), q, dtype=np.int64)
            t_arr = np.array(cand_t, dtype=np.int64)

            if mode == "baseline_specter2_only":
                scores = score_specter2_pairs(s_arr, t_arr, paper_emb)
            else:
                raw_feats = extractor.featurize_pairs(s_arr, t_arr)
                scaled = scaler.transform(raw_feats).astype(np.float32)
                scores = score_t22_pairs(t22_model, s_arr, t_arr,
                                          paper_emb, scaled)

            order = np.argsort(-scores)
            ranked = [cand_t[k] for k in order]
            gold_set = set(gold_ids)

            metric = {
                "n_gold": len(gold_set),
                "n_negatives": len(negs),
                "n_candidates": len(cand_t),
                "mrr": mrr(ranked, gold_set),
            }
            for k in K_VALUES:
                metric[f"recall_at_{k}"] = recall_at_k(ranked, gold_set, k)
                if k <= 50:
                    metric[f"ndcg_at_{k}"] = ndcg_at_k(ranked, gold_set, k)

            for type_id, type_name in EDGE_TYPE_NAMES.items():
                type_gold = {g for g in gold_set
                             if gold_types.get(g) == type_id}
                if type_gold:
                    metric[f"recall_at_10_{type_name}"] = recall_at_k(
                        ranked, type_gold, 10
                    )

            metric["q_subfield"] = venue_to_subfield(corpus[q].get("venue", ""))
            metric["query_id"] = int(q)
            per_q.append(metric)

        if not per_q:
            print("  No queries evaluated.")
            continue

        agg = {}
        keys = set()
        for m in per_q:
            keys.update(m.keys())
        keys -= {"n_gold", "n_negatives", "n_candidates",
                 "q_subfield", "query_id"}
        for k in keys:
            if k.startswith("recall") or k.startswith("ndcg") or k == "mrr":
                vs = [m[k] for m in per_q if k in m and m[k] is not None]
                if vs:
                    agg[f"mean_{k}"] = float(np.mean(vs))
                    agg[f"std_{k}"] = float(np.std(vs))
                    agg[f"n_{k}"] = len(vs)
        agg["n_queries"] = len(per_q)

        by_sub = defaultdict(list)
        for m in per_q:
            by_sub[m.get("q_subfield", "OTHER")].append(m)
        agg["per_subfield"] = {}
        for sub, ms in by_sub.items():
            agg["per_subfield"][sub] = {
                "n": len(ms),
                "mean_recall_at_10": float(np.mean(
                    [m.get("recall_at_10", 0) for m in ms]
                )),
                "mean_mrr": float(np.mean([m.get("mrr", 0) for m in ms])),
            }

        results["modes"][mode] = {
            "aggregate": agg,
            "per_query_sample": per_q[:5],
        }
        print(f"  {mode}: R@10={agg.get('mean_recall_at_10', 0):.4f}, "
              f"MRR={agg.get('mean_mrr', 0):.4f}")

    if "t22" in results["modes"] and "baseline_specter2_only" in results["modes"]:
        b = results["modes"]["baseline_specter2_only"]["aggregate"]
        t = results["modes"]["t22"]["aggregate"]
        lift = {}
        for metric in ["mean_recall_at_10", "mean_mrr",
                       "mean_recall_at_20", "mean_ndcg_at_10"]:
            if metric in b and metric in t:
                lift[metric] = {
                    "baseline": b[metric],
                    "t22": t[metric],
                    "absolute_delta": t[metric] - b[metric],
                    "relative_delta": (t[metric] - b[metric]) /
                                      max(b[metric], 1e-9),
                }
        results["lift_t22_over_baseline"] = lift

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
