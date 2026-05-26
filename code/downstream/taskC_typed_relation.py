"""
93_e10c_typed_relation_classification.py (v2)

Task C: Typed Relation Classification.

Setting: Given a verified positive edge (s, t), predict its relation type
from 9 categories. We use only positive edges (negatives have no relation
type to predict).

We use the precomputed T22 pair features from features.npz directly
(X_train_pos, X_val_pos, X_test_pos), so this script doesn't need to
re-featurize anything. We compare:

  - Random class prior baseline
  - SPECTER2-only MLP: uses concat(abstract_embs[s], abstract_embs[t])
  - T22 features classifier: uses the precomputed 63-dim features

For each positive pair (s, t), the gold class is the edge_type of the
graph edge from s to t (looked up via the graph).

Output:
  outputs/metrics/e10c_typed_relation_classification.json
  models/relation_classifier_t22/best.pt
  models/relation_classifier_specter2/best.pt

Wall-clock: ~10-20 min on A100.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix
)

# Local import (data_loader.py must be in same directory)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_all, EDGE_TYPE_NAMES


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class T22ClassificationHead(nn.Module):
    def __init__(self, in_dim=63, hidden=(128, 64), n_classes=9, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SPECTER2BaselineMLP(nn.Module):
    def __init__(self, in_dim=1536, hidden=(256, 128), n_classes=9, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def lookup_edge_types(s_arr, t_arr, edge_index, edge_type, n_papers):
    """
    For each (s, t) pair, look up the edge_type in the graph.

    Builds a sparse dict (s, t) -> edge_type. Falls back to type 7 (related_work)
    for pairs not found (defensive; should not occur if features.npz pairs
    were derived from the graph).
    """
    print("  Building (s, t) -> edge_type lookup...")
    pair_to_type = {}
    for i in range(edge_index.shape[1]):
        s = int(edge_index[0, i])
        t = int(edge_index[1, i])
        pair_to_type[(s, t)] = int(edge_type[i])
    print(f"  Lookup size: {len(pair_to_type)}")

    n_missing = 0
    types = np.zeros(len(s_arr), dtype=np.int64)
    for i, (s, t) in enumerate(zip(s_arr, t_arr)):
        s, t = int(s), int(t)
        if (s, t) in pair_to_type:
            types[i] = pair_to_type[(s, t)]
        elif (t, s) in pair_to_type:
            types[i] = pair_to_type[(t, s)]
        else:
            types[i] = 7  # default to "related_work"
            n_missing += 1
    if n_missing > 0:
        print(f"  WARNING: {n_missing}/{len(s_arr)} pairs not found in graph")
    return types


def build_specter2_features(s_arr, t_arr, abstract_embs):
    """Concatenated SPECTER2 embeddings, [N, 1536]."""
    return np.concatenate(
        [abstract_embs[s_arr], abstract_embs[t_arr]], axis=1
    )


def train_classifier(model, train_loader, val_loader, epochs, lr=1e-3,
                     class_weights=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if class_weights is not None:
        cw = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
        ce = nn.CrossEntropyLoss(weight=cw)
    else:
        ce = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits = model(X)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_loader.dataset)

        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                logits = model(X)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_pred.extend(preds.tolist())
                all_true.extend(y.numpy().tolist())
        val_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    Epoch {ep+1}/{epochs}: train_loss={train_loss:.4f} "
                  f"val_macro_f1={val_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_f1


def evaluate_classifier(model, test_loader):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(y.numpy().tolist())

    macro_f1 = float(f1_score(all_true, all_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_true, all_pred, average="weighted", zero_division=0))
    p, r, f, support = precision_recall_fscore_support(
        all_true, all_pred, labels=list(range(9)), zero_division=0
    )
    cm = confusion_matrix(all_true, all_pred, labels=list(range(9)))

    per_class = {}
    for i in range(9):
        per_class[EDGE_TYPE_NAMES[i]] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(support[i])
        }

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [EDGE_TYPE_NAMES[i] for i in range(9)],
        "n_test": len(all_true)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    data = load_all()
    paths = data["paths"]
    abstract_embs = data["abstract_embs"]
    edge_index = data["edge_index"]
    edge_type = data["edge_type"]
    n_papers = data["n_papers"]
    t22_feat = data["t22_features"]

    if t22_feat is None:
        print("ERROR: features.npz not loaded. Cannot proceed with Task C.")
        return

    OUTPUT_PATH = paths["outputs"] / "e10c_typed_relation_classification.json"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    T22_CLF_DIR = paths["project_root"] / "models/relation_classifier_t22"
    S2_CLF_DIR = paths["project_root"] / "models/relation_classifier_specter2"
    T22_CLF_DIR.mkdir(parents=True, exist_ok=True)
    S2_CLF_DIR.mkdir(parents=True, exist_ok=True)

    # Look up edge_type for each positive train/val/test pair
    print("\nLooking up edge types for positive pairs...")
    y_train = lookup_edge_types(
        t22_feat["train_pos_s"], t22_feat["train_pos_t"],
        edge_index, edge_type, n_papers
    )
    y_val = lookup_edge_types(
        t22_feat["val_pos_s"], t22_feat["val_pos_t"],
        edge_index, edge_type, n_papers
    )
    y_test = lookup_edge_types(
        t22_feat["test_pos_s"], t22_feat["test_pos_t"],
        edge_index, edge_type, n_papers
    )

    # Class distribution
    train_dist = Counter(y_train.tolist())
    print("\nTraining class distribution:")
    total_train = sum(train_dist.values())
    for i in range(9):
        c = train_dist.get(i, 0)
        print(f"  {i}: {EDGE_TYPE_NAMES[i]:<25} {c:>8} ({100*c/total_train:5.2f}%)")

    # Subsample for dry-run
    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        n_train_sub = min(5000, len(y_train))
        n_val_sub = min(1000, len(y_val))
        n_test_sub = min(1000, len(y_test))
        train_idx = rng.choice(len(y_train), size=n_train_sub, replace=False)
        val_idx = rng.choice(len(y_val), size=n_val_sub, replace=False)
        test_idx = rng.choice(len(y_test), size=n_test_sub, replace=False)
        print(f"\nDry-run subsample: train={n_train_sub}, val={n_val_sub}, "
              f"test={n_test_sub}")
    else:
        train_idx = np.arange(len(y_train))
        val_idx = np.arange(len(y_val))
        test_idx = np.arange(len(y_test))
        print(f"\nFull set: train={len(train_idx)}, val={len(val_idx)}, "
              f"test={len(test_idx)}")

    # ----- T22 features (precomputed) -----
    X_train_t22 = t22_feat["X_train_pos"][train_idx]
    X_val_t22 = t22_feat["X_val_pos"][val_idx]
    X_test_t22 = t22_feat["X_test_pos"][test_idx]
    y_train_sub = y_train[train_idx]
    y_val_sub = y_val[val_idx]
    y_test_sub = y_test[test_idx]

    # ----- SPECTER2 features (compute now) -----
    print("\nBuilding SPECTER2-concat features...")
    X_train_s2 = build_specter2_features(
        t22_feat["train_pos_s"][train_idx],
        t22_feat["train_pos_t"][train_idx],
        abstract_embs
    )
    X_val_s2 = build_specter2_features(
        t22_feat["val_pos_s"][val_idx],
        t22_feat["val_pos_t"][val_idx],
        abstract_embs
    )
    X_test_s2 = build_specter2_features(
        t22_feat["test_pos_s"][test_idx],
        t22_feat["test_pos_t"][test_idx],
        abstract_embs
    )

    # Class weights from training
    train_dist_sub = Counter(y_train_sub.tolist())
    class_weights = np.array([
        1.0 / max(train_dist_sub.get(i, 1), 1) for i in range(9)
    ])
    class_weights = class_weights / class_weights.sum() * 9

    epochs = 3 if args.dry_run else args.epochs

    results = {
        "config": {
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "n_classes": 9,
            "class_distribution": {
                EDGE_TYPE_NAMES[i]: int(train_dist_sub.get(i, 0))
                for i in range(9)
            },
            "epochs": epochs,
            "dry_run": args.dry_run,
            "seed": args.seed
        },
        "results": {}
    }

    def make_loaders(X_tr, X_va, X_te, y_tr, y_va, y_te, batch=512):
        tr = TensorDataset(torch.from_numpy(X_tr).float(),
                           torch.from_numpy(y_tr).long())
        va = TensorDataset(torch.from_numpy(X_va).float(),
                           torch.from_numpy(y_va).long())
        te = TensorDataset(torch.from_numpy(X_te).float(),
                           torch.from_numpy(y_te).long())
        return (DataLoader(tr, batch_size=batch, shuffle=True),
                DataLoader(va, batch_size=batch, shuffle=False),
                DataLoader(te, batch_size=batch, shuffle=False))

    # ----- Random baseline -----
    print("\n=== Random baseline ===")
    priors = np.array([train_dist_sub.get(i, 0) / max(sum(train_dist_sub.values()), 1)
                       for i in range(9)])
    rng2 = np.random.default_rng(0)
    rand_pred = rng2.choice(9, size=len(y_test_sub), p=priors)
    rand_macro_f1 = float(f1_score(y_test_sub, rand_pred, average="macro",
                                   zero_division=0))
    print(f"  Random macro-F1: {rand_macro_f1:.4f}")
    results["results"]["random_class_prior"] = {
        "macro_f1": rand_macro_f1,
        "method": "random sampling weighted by training class priors"
    }

    # ----- SPECTER2 baseline -----
    print("\n=== SPECTER2-only MLP baseline ===")
    s2_tr_loader, s2_va_loader, s2_te_loader = make_loaders(
        X_train_s2, X_val_s2, X_test_s2, y_train_sub, y_val_sub, y_test_sub
    )
    s2_model = SPECTER2BaselineMLP().to(DEVICE)
    print(f"  Training ({epochs} epochs)...")
    tic = time.time()
    s2_model, s2_val_f1 = train_classifier(
        s2_model, s2_tr_loader, s2_va_loader, epochs=epochs,
        class_weights=class_weights
    )
    s2_train_time = time.time() - tic
    print(f"  Evaluating on test set...")
    s2_test = evaluate_classifier(s2_model, s2_te_loader)
    s2_test["best_val_macro_f1"] = s2_val_f1
    s2_test["train_time_seconds"] = s2_train_time
    print(f"  SPECTER2 macro-F1: {s2_test['macro_f1']:.4f}")
    results["results"]["specter2_only_mlp"] = s2_test
    torch.save(s2_model.state_dict(), S2_CLF_DIR / "best.pt")

    # ----- T22 classifier -----
    print("\n=== T22 (63 engineered features) classifier ===")
    t22_tr_loader, t22_va_loader, t22_te_loader = make_loaders(
        X_train_t22, X_val_t22, X_test_t22, y_train_sub, y_val_sub, y_test_sub
    )
    t22_model = T22ClassificationHead().to(DEVICE)
    print(f"  Training ({epochs} epochs)...")
    tic = time.time()
    t22_model, t22_val_f1 = train_classifier(
        t22_model, t22_tr_loader, t22_va_loader, epochs=epochs,
        class_weights=class_weights
    )
    t22_train_time = time.time() - tic
    print(f"  Evaluating on test set...")
    t22_test = evaluate_classifier(t22_model, t22_te_loader)
    t22_test["best_val_macro_f1"] = t22_val_f1
    t22_test["train_time_seconds"] = t22_train_time
    print(f"  T22 macro-F1: {t22_test['macro_f1']:.4f}")
    results["results"]["t22_engineered_features"] = t22_test
    torch.save(t22_model.state_dict(), T22_CLF_DIR / "best.pt")

    # Lift
    results["lift_t22_over_specter2"] = {
        "macro_f1_delta": t22_test["macro_f1"] - s2_test["macro_f1"],
        "macro_f1_relative_delta": (
            (t22_test["macro_f1"] - s2_test["macro_f1"]) /
            max(s2_test["macro_f1"], 1e-9)
        )
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")
    print("\nSummary:")
    print(f"  Random:    macro-F1 = {rand_macro_f1:.4f}")
    print(f"  SPECTER2:  macro-F1 = {s2_test['macro_f1']:.4f}")
    print(f"  T22:       macro-F1 = {t22_test['macro_f1']:.4f}")
    print(f"  Lift T22 over SPECTER2: "
          f"{t22_test['macro_f1'] - s2_test['macro_f1']:+.4f}")


if __name__ == "__main__":
    main()
