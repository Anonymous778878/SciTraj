"""
E9: T22 feature-group ablation.

T22 has 63 expanded pair features. This script attributes T22's AUC win
to feature groups by dropping each group and retraining:

  Group A — Per-relation cosines (typed-relation signal),  cols  [0:9]   (9 dims)
  Group B — Per-relation degrees (structural),              cols  [9:45]  (36 dims)
  Group C — Common-neighbour features (structural),         cols [45:51]  (6 dims)
  Group D — Topic match flags,                              cols [51:53]  (2 dims)
  Group E — Year-gap basis (TEMPORAL),                      cols [53:58]  (5 dims)
  Group F — Back-edge and bidirectional indicators,         cols [58:60]  (2 dims)
  Group G — Source-target popularity ratios,                cols [60:63]  (3 dims)

Each group is dropped one at a time. The MLP is retrained from scratch
(same hyperparameters and 5-seed protocol). Reports mean ± std AUC for
each ablation, plus a contribution table.

Output:
  outputs/metrics/tier22_feature_ablation.json
  outputs/metrics/tier22_feature_ablation.md

Wall-clock: ~30-45 min (7 groups × 5 seeds × ~1 min/training).

Addresses: gcwq Concern 3 (ablation experiments) — direct mechanism
explanation for why T22 wins. Also gives reviewers attribution of
contributions to typed-relation, temporal, and structural sources.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from gnn_utils import load_graph
from utils import ensure_dir, get_logger, load_config

log = get_logger("t22_ablation")


# Feature-group map (start, end) inclusive-exclusive, names
GROUPS = {
    "A_per_rel_cosines":  (0, 9,   "Per-relation cosine similarities",   "typed-relation"),
    "B_per_rel_degrees":  (9, 45,  "Per-relation in/out degrees",         "structural"),
    "C_common_neighbour": (45, 51, "Common-neighbour features",           "structural"),
    "D_topic_match":      (51, 53, "Topic match flags",                   "topic"),
    "E_year_gap":         (53, 58, "Year-gap basis (5-dim)",              "temporal"),
    "F_back_bidirec":     (58, 60, "Back-edge / bidirectional indicators", "structural"),
    "G_pop_ratios":       (60, 63, "Source-target popularity ratios",     "structural"),
}

SEEDS = [42, 17, 2024, 511, 7]


class PairMLP(nn.Module):
    def __init__(self, d, n_extra, hidden=512, dropout=0.3):
        super().__init__()
        in_dim = 2 * d + n_extra
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4), nn.ReLU(),
            nn.Linear(hidden // 4, 1),
        )

    def forward(self, s, t, extra):
        f = torch.cat([torch.abs(s - t), s * t, extra], dim=1)
        return self.net(f).squeeze(-1)


def auc_ap(pos, neg):
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def train_eval(seed, X_tr_pos, X_tr_neg, X_va_pos, X_va_neg, X_te_pos, X_te_neg,
               paper_emb,
               tr_pos_s, tr_pos_t, tr_neg_s, tr_neg_t,
               va_pos_s, va_pos_t, va_neg_s, va_neg_t,
               te_pos_s, te_pos_t, te_neg_s, te_neg_t,
               device, n_epochs=30, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_extra = X_tr_pos.shape[1]
    model = PairMLP(paper_emb.shape[1], n_extra).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    X_tr = np.vstack([X_tr_pos, X_tr_neg])
    y_tr = np.concatenate([np.ones(len(X_tr_pos)), np.zeros(len(X_tr_neg))])
    s_tr = np.concatenate([tr_pos_s, tr_neg_s])
    t_tr = np.concatenate([tr_pos_t, tr_neg_t])

    X_tr_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    s_tr_t = torch.from_numpy(paper_emb[s_tr]).to(device)
    t_tr_t = torch.from_numpy(paper_emb[t_tr]).to(device)

    rng = np.random.default_rng(seed)
    n = len(X_tr); bs = 1024
    best_val = 0.0; best_state = None
    for epoch in range(n_epochs):
        model.train()
        idx = rng.permutation(n)
        for b0 in range(0, n, bs):
            b = idx[b0:b0+bs]
            opt.zero_grad()
            logits = model(s_tr_t[b], t_tr_t[b], X_tr_t[b])
            F.binary_cross_entropy_with_logits(logits, y_tr_t[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v_p = model(torch.from_numpy(paper_emb[va_pos_s]).to(device),
                        torch.from_numpy(paper_emb[va_pos_t]).to(device),
                        torch.from_numpy(X_va_pos.astype(np.float32)).to(device)).cpu().numpy()
            v_n = model(torch.from_numpy(paper_emb[va_neg_s]).to(device),
                        torch.from_numpy(paper_emb[va_neg_t]).to(device),
                        torch.from_numpy(X_va_neg.astype(np.float32)).to(device)).cpu().numpy()
        v_auc, _ = auc_ap(v_p, v_n)
        if v_auc > best_val:
            best_val = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        t_p = model(torch.from_numpy(paper_emb[te_pos_s]).to(device),
                    torch.from_numpy(paper_emb[te_pos_t]).to(device),
                    torch.from_numpy(X_te_pos.astype(np.float32)).to(device)).cpu().numpy()
        t_n = model(torch.from_numpy(paper_emb[te_neg_s]).to(device),
                    torch.from_numpy(paper_emb[te_neg_t]).to(device),
                    torch.from_numpy(X_te_neg.astype(np.float32)).to(device)).cpu().numpy()
    auc, ap = auc_ap(t_p, t_n)
    return auc, ap, best_val


def drop_columns(X, start, end):
    """Replace columns [start:end] with zeros (preserves dim, drops signal)."""
    X2 = X.copy()
    X2[:, start:end] = 0.0
    return X2


def main():
    cfg = load_config()
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    models_root = Path(cfg["paths"]["models_dir"])
    graph_dir = Path(cfg["paths"]["graph_dir"])

    log.info("loading graph + features")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    paper_emb = data["paper"].x_abstract.cpu().numpy().astype(np.float32)

    feat_path = models_root / "tier22_pair_mlp_expanded" / "features.npz"
    if not feat_path.exists():
        log.error("features.npz missing; run script 67 first")
        return
    z = np.load(feat_path)
    X_tr_pos = z["X_train_pos"]; X_tr_neg = z["X_train_neg"]
    X_va_pos = z["X_val_pos"];   X_va_neg = z["X_val_neg"]
    X_te_pos = z["X_test_pos"];  X_te_neg = z["X_test_neg"]
    tr_pos_s = z["train_pos_s"]; tr_pos_t = z["train_pos_t"]
    tr_neg_s = z["train_neg_s"]; tr_neg_t = z["train_neg_t"]
    va_pos_s = z["val_pos_s"];   va_pos_t = z["val_pos_t"]
    va_neg_s = z["val_neg_s"];   va_neg_t = z["val_neg_t"]
    te_pos_s = z["test_pos_s"];  te_pos_t = z["test_pos_t"]
    te_neg_s = z["test_neg_s"];  te_neg_t = z["test_neg_t"]
    n_features = X_tr_pos.shape[1]
    log.info("feature dim: %d", n_features)
    if n_features != 63:
        log.warning("expected 63 features, got %d — group indices may be off", n_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Full T22 baseline (5 seeds)
    log.info("=" * 60)
    log.info("BASELINE T22 (full features, 5 seeds)")
    log.info("=" * 60)
    base_aucs, base_aps = [], []
    for seed in SEEDS:
        t0 = time.time()
        auc, ap, _ = train_eval(
            seed, X_tr_pos, X_tr_neg, X_va_pos, X_va_neg, X_te_pos, X_te_neg,
            paper_emb,
            tr_pos_s, tr_pos_t, tr_neg_s, tr_neg_t,
            va_pos_s, va_pos_t, va_neg_s, va_neg_t,
            te_pos_s, te_pos_t, te_neg_s, te_neg_t,
            device,
        )
        base_aucs.append(auc); base_aps.append(ap)
        log.info("  seed %d: AUC=%.4f AP=%.4f (%.1fs)", seed, auc, ap, time.time() - t0)
    base_auc_mean = np.mean(base_aucs); base_auc_std = np.std(base_aucs)
    base_ap_mean = np.mean(base_aps);   base_ap_std = np.std(base_aps)
    log.info("BASELINE: AUC=%.4f±%.4f AP=%.4f±%.4f",
             base_auc_mean, base_auc_std, base_ap_mean, base_ap_std)

    # Per-group ablations
    group_results = {}
    for group_id, (start, end, name, category) in GROUPS.items():
        log.info("=" * 60)
        log.info("Ablating group %s (%s, cols %d:%d, %d dims)",
                 group_id, name, start, end, end - start)
        log.info("=" * 60)
        Xtp = drop_columns(X_tr_pos, start, end); Xtn = drop_columns(X_tr_neg, start, end)
        Xvp = drop_columns(X_va_pos, start, end); Xvn = drop_columns(X_va_neg, start, end)
        Xep = drop_columns(X_te_pos, start, end); Xen = drop_columns(X_te_neg, start, end)

        ablation_aucs, ablation_aps = [], []
        for seed in SEEDS:
            t0 = time.time()
            auc, ap, _ = train_eval(
                seed, Xtp, Xtn, Xvp, Xvn, Xep, Xen, paper_emb,
                tr_pos_s, tr_pos_t, tr_neg_s, tr_neg_t,
                va_pos_s, va_pos_t, va_neg_s, va_neg_t,
                te_pos_s, te_pos_t, te_neg_s, te_neg_t,
                device,
            )
            ablation_aucs.append(auc); ablation_aps.append(ap)
            log.info("  seed %d: AUC=%.4f AP=%.4f (%.1fs)", seed, auc, ap, time.time() - t0)

        rec = {
            "group_id": group_id, "group_name": name, "category": category,
            "n_dims_dropped": end - start,
            "auc_mean": round(float(np.mean(ablation_aucs)), 4),
            "auc_std":  round(float(np.std(ablation_aucs)),  4),
            "ap_mean":  round(float(np.mean(ablation_aps)),  4),
            "ap_std":   round(float(np.std(ablation_aps)),   4),
            "delta_auc_vs_full": round(float(np.mean(ablation_aucs)) - base_auc_mean, 4),
            "delta_ap_vs_full":  round(float(np.mean(ablation_aps))  - base_ap_mean,  4),
        }
        group_results[group_id] = rec
        log.info("group %s: ΔAUC = %+.4f", group_id, rec["delta_auc_vs_full"])

    # Save
    out = {
        "task": "T22 feature-group ablation",
        "seeds": SEEDS,
        "baseline_full": {
            "auc_mean": round(base_auc_mean, 4), "auc_std": round(base_auc_std, 4),
            "ap_mean":  round(base_ap_mean, 4),  "ap_std":  round(base_ap_std, 4),
        },
        "groups": group_results,
        "n_test_positive": int(len(te_pos_s)),
        "n_test_negative": int(len(te_neg_s)),
    }
    out_path = metrics_dir / "tier22_feature_ablation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote: %s", out_path)

    md = ["# T22 Feature-Group Ablation\n\n"]
    md.append(f"5 seeds, baseline AUC = {base_auc_mean:.4f} ± {base_auc_std:.4f}\n\n")
    md.append("## Contribution per feature group\n\n")
    md.append("Lower AUC after dropping = group contributed more to the win.\n\n")
    md.append("| Group | Category | Dims | AUC after drop | Δ AUC vs full |\n")
    md.append("|-------|----------|------|----------------|---------------|\n")
    sorted_groups = sorted(group_results.values(), key=lambda r: r["delta_auc_vs_full"])
    for r in sorted_groups:
        md.append(
            f"| {r['group_id']} ({r['group_name']}) | {r['category']} | "
            f"{r['n_dims_dropped']} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} | "
            f"{r['delta_auc_vs_full']:+.4f} |\n"
        )

    md.append("\n## Per-category attribution\n\n")
    cat_totals = {}
    for r in group_results.values():
        cat_totals.setdefault(r["category"], 0.0)
        cat_totals[r["category"]] += abs(r["delta_auc_vs_full"])
    md.append("| Category | Total |Δ AUC| (sum of magnitudes) |\n")
    md.append("|----------|--------------------------------------|\n")
    for cat, total in sorted(cat_totals.items(), key=lambda x: -x[1]):
        md.append(f"| {cat} | {total:.4f} |\n")

    md.append("\n## Reading\n\n")
    md.append("- Most-negative Δ AUC = the feature group whose removal hurt T22 most.\n")
    md.append("- The 'temporal' category answers: 'how much of T22's lift is year-driven?'\n")
    md.append("- The 'typed-relation' category answers: 'do the per-relation cosines\n")
    md.append("  add anything beyond raw SPECTER2?'\n")
    md.append("- The 'structural' category answers: 'how much is graph-structural?'\n")
    md.append("\n## Addresses Reviewer gcwq Concern 3 (ablation)\n\n")
    md.append("This is the systematic feature-level ablation the reviewer requested,\n")
    md.append("complementary to the year-shuffling causal control. Together:\n\n")
    md.append("- **Year shuffle** (script 70): permute years, hold features unchanged → \n")
    md.append("  attributes performance to year information.\n")
    md.append("- **This script**: drop feature groups, hold years unchanged → attributes\n")
    md.append("  performance to specific feature engineering.\n\n")
    md.append("Either alone would be incomplete. Both together fully decompose where\n")
    md.append("T22's win comes from.\n")

    out_md = metrics_dir / "tier22_feature_ablation.md"
    out_md.write_text("".join(md))
    log.info("wrote: %s", out_md)


if __name__ == "__main__":
    main()
