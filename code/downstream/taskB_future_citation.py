"""
92_e10b_future_citation_prediction.py (v2)

Task B: Future Citation Prediction (temporal split).

For each test year Y in {2021, 2022, 2023, 2024}, train on edges where the
citing paper has year < Y, and evaluate on edges where citing-year == Y.

Compares:
  - SPECTER2 cosine baseline (year-agnostic)
  - T22 retrained on the temporally-restricted training graph

Output:
  outputs/metrics/e10b_future_citation_prediction.json
  models/tier22_temporal_split/{Y}/best.pt

Wall-clock: ~45-90 min on A100.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_all, EDGE_TYPE_NAMES


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PairMLP(nn.Module):
    def __init__(self, in_dim=63, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_degree_cache(edge_index, edge_type, n_nodes, mask=None):
    """Compute per-relation in/out degrees.

    If mask is given (boolean array of length E), only edges where mask[i] is
    True contribute. This is how we enforce the temporal restriction.
    """
    in_deg = np.zeros((n_nodes, 9), dtype=np.float32)
    out_deg = np.zeros((n_nodes, 9), dtype=np.float32)
    if mask is None:
        idx = range(edge_index.shape[1])
    else:
        idx = np.where(mask)[0]
    for i in idx:
        s = int(edge_index[0, i])
        t = int(edge_index[1, i])
        et = int(edge_type[i])
        if 0 <= et < 9 and 0 <= s < n_nodes and 0 <= t < n_nodes:
            out_deg[s, et] += 1
            in_deg[t, et] += 1
    return in_deg, out_deg


def venue_to_subfield(venue):
    v = (venue or "").lower()
    if any(s in v for s in ["cvpr", "iccv", "wacv", "bmvc", "eccv"]):
        return "CV"
    if any(s in v for s in ["acl", "emnlp", "naacl", "eacl", "tacl", "coling"]):
        return "NLP"
    if any(s in v for s in ["neurips", "nips", "icml", "iclr", "aaai", "ijcai"]):
        return "ML"
    return "OTHER"


def featurize_pair(s, t, abstract_embs, corpus, in_deg, out_deg):
    """63-dim T22 features (same layout as Task A)."""
    f = np.zeros(63, dtype=np.float32)
    s_emb = abstract_embs[s]
    t_emb = abstract_embs[t]
    cos = float((s_emb @ t_emb) /
                ((np.linalg.norm(s_emb) + 1e-9) *
                 (np.linalg.norm(t_emb) + 1e-9)))
    f[0:9] = cos
    f[9:18] = in_deg[s]
    f[18:27] = out_deg[s]
    f[27:36] = in_deg[t]
    f[36:45] = out_deg[t]
    s_year = corpus[s].get("year", 2020)
    t_year = corpus[t].get("year", 2020)
    s_sub = venue_to_subfield(corpus[s].get("venue", ""))
    t_sub = venue_to_subfield(corpus[t].get("venue", ""))
    f[51] = 1.0 if s_sub == t_sub else 0.0
    f[52] = 1.0 if (s_year // 5) == (t_year // 5) else 0.0
    gap = s_year - t_year
    for j, mu in enumerate([0, 1, 2, 5, 10]):
        f[53 + j] = np.exp(-((gap - mu) ** 2) / 4.0)
    f[60] = in_deg[t].sum()
    f[61] = out_deg[t].sum()
    f[62] = f[60] + f[61]
    return f


def build_features_batch(pairs, abstract_embs, corpus, in_deg, out_deg):
    feats = np.zeros((len(pairs), 63), dtype=np.float32)
    for i, (s, t) in enumerate(pairs):
        feats[i] = featurize_pair(s, t, abstract_embs, corpus, in_deg, out_deg)
    return feats


def make_temporal_split(edge_index, edge_type, corpus, test_year, n_papers, rng):
    """
    Given the flat edge arrays and a test year Y, build:
      - train_pos: edges with citing-year < Y
      - test_pos: edges with citing-year == Y

    We need to determine the edge convention (does edge go cited→citing or
    citing→cited?). We check by looking at year stats.
    """
    n = edge_index.shape[1]
    src_years = np.array([corpus[int(edge_index[0, i])].get("year", 2020)
                          for i in range(min(2000, n))])
    tgt_years = np.array([corpus[int(edge_index[1, i])].get("year", 2020)
                          for i in range(min(2000, n))])
    src_is_citing = src_years.mean() > tgt_years.mean()
    print(f"  Edge convention: src.mean_year={src_years.mean():.1f}, "
          f"tgt.mean_year={tgt_years.mean():.1f}; "
          f"src=citing? {src_is_citing}")

    # Get citing year for every edge
    train_pos, test_pos = [], []
    train_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        s = int(edge_index[0, i])
        t = int(edge_index[1, i])
        et = int(edge_type[i])
        citing_id = s if src_is_citing else t
        cited_id = t if src_is_citing else s
        citing_year = corpus[citing_id].get("year", 2020)
        if citing_year < test_year:
            train_pos.append((cited_id, citing_id, et))
            train_mask[i] = True
        elif citing_year == test_year:
            test_pos.append((cited_id, citing_id, et))

    print(f"  Train edges (citing year < {test_year}): {len(train_pos)}")
    print(f"  Test edges (citing year == {test_year}): {len(test_pos)}")

    if len(test_pos) == 0:
        return None

    # Negative sampling: 1:1 ratio
    pos_set = set((s, t) for s, t, _ in train_pos + test_pos)

    def sample_negs(pos_list, k=1):
        negs = []
        for s, t, et in pos_list:
            for _ in range(k):
                tries = 0
                while tries < 100:
                    cand = int(rng.integers(0, n_papers))
                    if cand != s and (s, cand) not in pos_set:
                        negs.append((s, cand, et))
                        break
                    tries += 1
        return negs

    train_neg = sample_negs(train_pos)
    test_neg = sample_negs(test_pos)

    return {
        "train_pos": train_pos,
        "train_neg": train_neg,
        "test_pos": test_pos,
        "test_neg": test_neg,
        "train_edge_mask": train_mask,
    }


def train_split(split, abstract_embs, corpus, n_papers, edge_index, edge_type,
                 epochs=20, batch=512):
    """Train T22 from scratch on the temporally-restricted edges."""
    in_deg, out_deg = build_degree_cache(
        edge_index, edge_type, n_papers, mask=split["train_edge_mask"]
    )

    train_pairs = ([(s, t) for s, t, _ in split["train_pos"]] +
                   [(s, t) for s, t, _ in split["train_neg"]])
    train_y = np.concatenate([
        np.ones(len(split["train_pos"])),
        np.zeros(len(split["train_neg"]))
    ])
    print(f"  Featurizing {len(train_pairs)} training pairs...")
    tic = time.time()
    train_X = build_features_batch(train_pairs, abstract_embs, corpus,
                                     in_deg, out_deg)
    print(f"    done ({time.time()-tic:.0f}s)")

    tr_ds = TensorDataset(torch.from_numpy(train_X).float(),
                          torch.from_numpy(train_y).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch, shuffle=True)

    model = PairMLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(X)
            loss = bce(logits, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(X)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    Epoch {ep+1}/{epochs}: loss={ep_loss/len(train_X):.4f}")

    return model, in_deg, out_deg


def evaluate_split(model, split, abstract_embs, corpus, in_deg, out_deg):
    test_pairs = ([(s, t) for s, t, _ in split["test_pos"]] +
                  [(s, t) for s, t, _ in split["test_neg"]])
    test_y = np.concatenate([
        np.ones(len(split["test_pos"])),
        np.zeros(len(split["test_neg"]))
    ])
    test_types = [et for _, _, et in split["test_pos"] + split["test_neg"]]

    print(f"  Featurizing {len(test_pairs)} test pairs...")
    test_X = build_features_batch(test_pairs, abstract_embs, corpus,
                                    in_deg, out_deg)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_X).float().to(DEVICE))
        scores = torch.sigmoid(logits).cpu().numpy()

    auc = roc_auc_score(test_y, scores)
    ap = average_precision_score(test_y, scores)

    per_rel = {}
    for type_id, type_name in EDGE_TYPE_NAMES.items():
        mask = np.array([t == type_id for t in test_types])
        type_y = test_y[mask]
        type_s = scores[mask]
        if len(np.unique(type_y)) > 1:
            per_rel[type_name] = {
                "auc": float(roc_auc_score(type_y, type_s)),
                "ap": float(average_precision_score(type_y, type_s)),
                "n": int(mask.sum())
            }

    return {
        "auc": float(auc), "ap": float(ap),
        "n_test": len(test_y),
        "n_pos": int(test_y.sum()),
        "per_relation": per_rel,
    }


def evaluate_specter2_baseline(split, abstract_embs):
    test_pairs = ([(s, t) for s, t, _ in split["test_pos"]] +
                  [(s, t) for s, t, _ in split["test_neg"]])
    test_y = np.concatenate([
        np.ones(len(split["test_pos"])),
        np.zeros(len(split["test_neg"]))
    ])
    scores = []
    for s, t in test_pairs:
        cs = float((abstract_embs[s] @ abstract_embs[t]) /
                   ((np.linalg.norm(abstract_embs[s]) + 1e-9) *
                    (np.linalg.norm(abstract_embs[t]) + 1e-9)))
        scores.append(cs)
    scores = np.array(scores)
    return {
        "auc": float(roc_auc_score(test_y, scores)),
        "ap": float(average_precision_score(test_y, scores)),
        "n_test": len(test_y)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--years", nargs="+", type=int,
                        default=[2021, 2022, 2023, 2024])
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    data = load_all(load_t22_features=False)
    paths = data["paths"]
    abstract_embs = data["abstract_embs"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    corpus = data["corpus"]
    n_papers = data["n_papers"]

    OUTPUT_PATH = paths["outputs"] / "e10b_future_citation_prediction.json"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRAINED_DIR = paths["project_root"] / "models/tier22_temporal_split"
    RETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    test_years = [2022] if args.dry_run else args.years
    epochs = 3 if args.dry_run else args.epochs

    results = {
        "config": {
            "test_years": test_years,
            "epochs": epochs,
            "dry_run": args.dry_run
        },
        "results_by_year": {}
    }

    rng = np.random.default_rng(42)

    for Y in test_years:
        print(f"\n{'='*60}")
        print(f"=== Test year {Y} ===")
        print(f"{'='*60}")
        split = make_temporal_split(edge_index, edge_type, corpus, Y,
                                     n_papers, rng)
        if split is None:
            print(f"  No test edges for year {Y}, skipping.")
            continue

        print(f"\n  Running SPECTER2 baseline...")
        baseline = evaluate_specter2_baseline(split, abstract_embs)
        print(f"  SPECTER2: AUC={baseline['auc']:.4f}, AP={baseline['ap']:.4f}")

        print(f"\n  Training T22 on edges < year {Y}...")
        tic = time.time()
        model, in_deg, out_deg = train_split(
            split, abstract_embs, corpus, n_papers, edge_index, edge_type,
            epochs=epochs
        )
        train_time = time.time() - tic

        ckpt_path = RETRAINED_DIR / f"{Y}/best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)

        print(f"\n  Evaluating retrained T22 on year {Y}...")
        t22_results = evaluate_split(model, split, abstract_embs, corpus,
                                       in_deg, out_deg)
        print(f"  T22: AUC={t22_results['auc']:.4f}, AP={t22_results['ap']:.4f}")

        results["results_by_year"][str(Y)] = {
            "specter2_baseline": baseline,
            "t22_retrained": t22_results,
            "train_time_seconds": train_time,
            "n_train_edges": len(split["train_pos"]),
            "n_test_edges": len(split["test_pos"])
        }

        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n\nFinal results saved to {OUTPUT_PATH}")
    print("\nSummary:")
    print(f"{'Year':<8} {'SPECTER2 AUC':<15} {'T22 AUC':<15} {'Δ AUC':<10}")
    for Y, r in results["results_by_year"].items():
        b = r["specter2_baseline"]["auc"]
        t = r["t22_retrained"]["auc"]
        print(f"{Y:<8} {b:<15.4f} {t:<15.4f} {t-b:+.4f}")


if __name__ == "__main__":
    main()
