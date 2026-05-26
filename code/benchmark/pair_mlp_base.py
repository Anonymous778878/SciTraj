"""
run_w12_multiseed_pair_mlp_base_standalone.py

Self-contained version of W12 — builds Pair-MLP-Base features from
scratch from typed_edges.json + SPECTER2 embeddings.

Pair-MLP-Base feature set (per §4.2 of the paper):
  - |e_s - e_t|  (768 dims)
  - e_s * e_t   (768 dims, element-wise product)
  - Small structural extras: in_deg(s), out_deg(s), in_deg(t),
    out_deg(t), common_neighbors(s, t), year_gap
  
Total: 768 + 768 + 6 = 1542 dims.

For training, we use 1 hard negative per positive: same year ±2,
not connected to s.

Run:
    python3 run_w12_multiseed_pair_mlp_base_standalone.py
"""
import argparse
import json
import time
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

SEEDS = [42, 17, 2024, 511, 7]
TRAIN_YEAR_MAX = 2020
VAL_YEARS = {2021, 2022}
TEST_YEARS = {2023, 2024}

# Pair-MLP-Base config
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 20
BATCH_SIZE = 256


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------

def load_data(corpus_dir):
    corpus_dir = Path(corpus_dir)
    print("Loading typed_edges.json...")
    edges = json.load(open(corpus_dir / "data/graph/typed_edges.json"))
    print(f"  {len(edges):,} total edges")
    
    print("Loading paper years...")
    for p in [corpus_dir / "data/filtered/corpus.json",
              corpus_dir / "data/standardized/corpus.json"]:
        if p.exists():
            papers = json.load(open(p))
            break
    years = {int(p["paper_id"]): p.get("year") for p in papers}
    
    print("Loading SPECTER2 embeddings...")
    embeddings = np.load(corpus_dir / "data/embeddings/abstract_embs.npy")
    print(f"  shape={embeddings.shape}")
    
    return edges, years, embeddings


def build_graph(edges, years):
    """Build out/in adjacency for structural features."""
    out_adj = defaultdict(set)
    in_adj = defaultdict(set)
    train_edges = []
    val_edges = []
    test_edges = []
    
    for e in edges:
        src = int(e["src"])
        tgt = int(e["tgt"])
        if src not in years or tgt not in years:
            continue
        
        citing_year = years[src]
        
        # All edges visible during training go into the adjacency
        # (in time-truncated fashion, but simplified here for Pair-MLP-Base)
        if citing_year <= TRAIN_YEAR_MAX:
            train_edges.append((src, tgt))
            out_adj[src].add(tgt)
            in_adj[tgt].add(src)
        elif citing_year in VAL_YEARS:
            val_edges.append((src, tgt))
        elif citing_year in TEST_YEARS:
            test_edges.append((src, tgt))
    
    return train_edges, val_edges, test_edges, out_adj, in_adj


def sample_negatives(positives, all_pos_set, years, num_nodes,
                     n_per_pos=1, max_attempts=50, seed=42):
    """Hard negatives: same year ±2, not a positive edge of s."""
    rng = np.random.RandomState(seed)
    negs = []
    for src, tgt in positives:
        src_year = years.get(src)
        if src_year is None:
            continue
        for _ in range(n_per_pos):
            for _attempt in range(max_attempts):
                cand = int(rng.randint(0, num_nodes))
                if cand == src or cand == tgt:
                    continue
                cand_year = years.get(cand)
                if cand_year is None or abs(cand_year - src_year) > 2:
                    continue
                if (src, cand) in all_pos_set:
                    continue
                negs.append((src, cand))
                break
    return negs


def build_features(pos_edges, neg_edges, embeddings, years, out_adj, in_adj):
    """Build Pair-MLP-Base features for positive + negative edges."""
    pairs = [(s, t, 1) for s, t in pos_edges] + [(s, t, 0) for s, t in neg_edges]
    n = len(pairs)
    d_emb = embeddings.shape[1]
    
    X = np.zeros((n, 2 * d_emb + 6), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)
    
    for i, (s, t, label) in enumerate(pairs):
        if s >= embeddings.shape[0] or t >= embeddings.shape[0]:
            continue
        es = embeddings[s]
        et = embeddings[t]
        X[i, :d_emb] = np.abs(es - et)
        X[i, d_emb:2*d_emb] = es * et
        # Structural extras (6 dims)
        X[i, 2*d_emb + 0] = np.log1p(len(out_adj.get(s, [])))
        X[i, 2*d_emb + 1] = np.log1p(len(in_adj.get(s, [])))
        X[i, 2*d_emb + 2] = np.log1p(len(out_adj.get(t, [])))
        X[i, 2*d_emb + 3] = np.log1p(len(in_adj.get(t, [])))
        X[i, 2*d_emb + 4] = np.log1p(
            len(out_adj.get(s, set()) & in_adj.get(t, set())))
        X[i, 2*d_emb + 5] = years.get(s, 2015) - years.get(t, 2015)
        y[i] = label
    
    return X, y


# ----------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------

class PairMLPBase(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers = []
        d = input_dim
        for h in HIDDEN_DIMS:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(DROPOUT))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ----------------------------------------------------------------------
# TRAIN ONE SEED
# ----------------------------------------------------------------------

def train_one_seed(seed, X_train, y_train, X_val, y_val, X_test, y_test,
                   device="cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    
    X_train_t = torch.from_numpy(X_train_s).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_val_t = torch.from_numpy(X_val_s).to(device)
    X_test_t = torch.from_numpy(X_test_s).to(device)
    
    model = PairMLPBase(X_train_s.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0.0
    best_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train_t))
        total_loss = 0.0
        n_batches = 0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(X_train_t[idx])
            loss = criterion(logits, y_train_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).cpu().numpy()
        val_auc = roc_auc_score(y_val, val_logits)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t).cpu().numpy()
    test_auc = roc_auc_score(y_test, test_logits)
    test_ap = average_precision_score(y_test, test_logits)
    
    return {
        "seed": seed,
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "best_val_auc": float(best_val_auc),
    }


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default=".")
    parser.add_argument("--output_dir", default="outputs/metrics")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    
    corpus_dir = Path(args.corpus_dir).resolve()
    out_dir = corpus_dir / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("W12: Multi-Seed Pair-MLP-Base (Standalone) on Task 1")
    print("=" * 70)
    
    edges, years, embeddings = load_data(corpus_dir)
    num_nodes = embeddings.shape[0]
    
    print("\nBuilding graph and splits...")
    train_pos, val_pos, test_pos, out_adj, in_adj = build_graph(edges, years)
    print(f"  train_pos={len(train_pos):,}, val_pos={len(val_pos):,}, "
          f"test_pos={len(test_pos):,}")
    print(f"  out_adj source nodes: {len(out_adj):,}")
    
    all_pos = set(train_pos) | set(val_pos) | set(test_pos)
    
    print("\nSampling hard negatives...")
    t0 = time.time()
    train_neg = sample_negatives(train_pos, all_pos, years, num_nodes,
                                  n_per_pos=1, seed=42)
    val_neg = sample_negatives(val_pos, all_pos, years, num_nodes,
                                n_per_pos=1, seed=43)
    test_neg = sample_negatives(test_pos, all_pos, years, num_nodes,
                                 n_per_pos=1, seed=44)
    print(f"  train_neg={len(train_neg):,}, val_neg={len(val_neg):,}, "
          f"test_neg={len(test_neg):,}  ({time.time()-t0:.1f}s)")
    
    print("\nBuilding features...")
    t0 = time.time()
    X_train, y_train = build_features(train_pos, train_neg, embeddings,
                                       years, out_adj, in_adj)
    X_val, y_val = build_features(val_pos, val_neg, embeddings,
                                   years, out_adj, in_adj)
    X_test, y_test = build_features(test_pos, test_neg, embeddings,
                                     years, out_adj, in_adj)
    print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"  feature build took {time.time()-t0:.1f}s")
    
    # Run all seeds
    results = []
    for seed in args.seeds:
        print(f"\n[Seed {seed}]")
        t0 = time.time()
        r = train_one_seed(seed, X_train, y_train, X_val, y_val,
                            X_test, y_test, device=args.device)
        elapsed = time.time() - t0
        print(f"  test_auc = {r['test_auc']:.4f}, test_ap = {r['test_ap']:.4f}, "
              f"best_val_auc = {r['best_val_auc']:.4f}  [{elapsed:.1f}s]")
        results.append(r)
    
    test_aucs = np.array([r["test_auc"] for r in results])
    test_aps = np.array([r["test_ap"] for r in results])
    
    summary = {
        "model": "Pair-MLP-Base",
        "seeds": args.seeds,
        "n_seeds": len(results),
        "test_auc_mean": float(test_aucs.mean()),
        "test_auc_std": float(test_aucs.std(ddof=1)),
        "test_auc_min": float(test_aucs.min()),
        "test_auc_max": float(test_aucs.max()),
        "test_ap_mean": float(test_aps.mean()),
        "test_ap_std": float(test_aps.std(ddof=1)),
        "per_seed": results,
    }
    
    out_path = out_dir / "w12_pair_mlp_base_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Wrote {out_path}")
    
    print("\n" + "=" * 70)
    print("HEADLINE FOR PAPER:")
    print("=" * 70)
    print(f"  Pair-MLP-Base (5 seeds):")
    print(f"    AUC:  {test_aucs.mean():.4f} ± {test_aucs.std(ddof=1):.4f}")
    print(f"    AP:   {test_aps.mean():.4f} ± {test_aps.std(ddof=1):.4f}")
    print(f"    Range: [{test_aucs.min():.4f}, {test_aucs.max():.4f}]")
    print()
    print(f"  Reported single-seed AUC: 0.908")
    print(f"  SciTraj-Pair (5 seeds): 0.9137 ± 0.0051")


if __name__ == "__main__":
    main()
