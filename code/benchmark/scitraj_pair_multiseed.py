"""
E7: T22 multi-seed evaluation + paired-t test vs T2.

Runs T22 (Pair-MLP expanded) 5 times with different random seeds and
computes mean ± std for AUC and AP. For each seed, also evaluates T2
(parameter-free aggregation) on the same hard-negative test split.
Then runs a paired t-test on per-pair logits to test:

    H0:  T22 score(s, t) does not predict positives better than T2 cosine(s, t)
    H1:  T22 scores discriminate positives better than T2 cosine

We use a paired bootstrap (10k resamples) on per-pair difference scores
because raw paired-t on AUC point estimates ignores the within-pair
correlation structure. The bootstrap test is more conservative and more
defensible.

Output:
  outputs/metrics/tier22_multiseed.json
  outputs/metrics/tier22_significance.md

Wall-clock: ~10-15 min (T22 trains in 1-2 min per seed; 5 seeds + bootstrap).
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gnn_utils import load_graph, prepare_temporal_split
from hard_negatives import build_positive_set, build_topic_year_pools
from hard_negatives_v2 import (
    build_candidate_pool_negatives,
    sample_candidate_pool_negatives,
)
from utils import ensure_dir, get_logger, load_config, load_json

log = get_logger("t22_multiseed")


SEEDS = [42, 17, 2024, 511, 7]
N_BOOTSTRAP = 10000


# ===================== T22 architecture (matches script 67) =====================

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


def cosine_similarity(emb, src, tgt):
    a = emb[src]; b = emb[tgt]
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a * b).sum(axis=1)


def auc_ap(pos_scores, neg_scores):
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def t2_aggregation(paper_emb, eidx_np, ew_np, n_papers):
    """T2 — parameter-free aggregation (replicates script 19)."""
    out = np.zeros_like(paper_emb)
    cnt = np.zeros(n_papers)
    for i in range(eidx_np.shape[1]):
        out[eidx_np[1, i]] += paper_emb[eidx_np[0, i]] * ew_np[i]
        cnt[eidx_np[1, i]] += ew_np[i]
    has = cnt > 0
    out[has] /= cnt[has, None]
    out[~has] = paper_emb[~has]
    return 0.5 * paper_emb + 0.5 * out


def train_one_seed(seed, X_tr_pos, X_tr_neg, X_va_pos, X_va_neg,
                   X_te_pos, X_te_neg, paper_emb,
                   tr_pos_s, tr_pos_t, tr_neg_s, tr_neg_t,
                   va_pos_s, va_pos_t, va_neg_s, va_neg_t,
                   te_pos_s, te_pos_t, te_neg_s, te_neg_t,
                   device, n_epochs=30, lr=1e-3):
    """Train T22 once with given seed; return test pos+neg logits."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_extra = X_tr_pos.shape[1]
    d = paper_emb.shape[1]
    model = PairMLP(d, n_extra).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Combine pos+neg train
    X_tr = np.vstack([X_tr_pos, X_tr_neg])
    y_tr = np.concatenate([np.ones(len(X_tr_pos)), np.zeros(len(X_tr_neg))])
    s_tr = np.concatenate([tr_pos_s, tr_neg_s])
    t_tr = np.concatenate([tr_pos_t, tr_neg_t])

    X_tr_t = torch.from_numpy(X_tr.astype(np.float32)).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    s_tr_t = torch.from_numpy(paper_emb[s_tr]).to(device)
    t_tr_t = torch.from_numpy(paper_emb[t_tr]).to(device)

    rng = np.random.default_rng(seed)
    n = len(X_tr)
    batch_size = 1024
    best_val_auc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        idx = rng.permutation(n)
        for b_start in range(0, n, batch_size):
            b = idx[b_start:b_start + batch_size]
            opt.zero_grad()
            logits = model(s_tr_t[b], t_tr_t[b], X_tr_t[b])
            loss = F.binary_cross_entropy_with_logits(logits, y_tr_t[b])
            loss.backward()
            opt.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_p = model(
                torch.from_numpy(paper_emb[va_pos_s]).to(device),
                torch.from_numpy(paper_emb[va_pos_t]).to(device),
                torch.from_numpy(X_va_pos.astype(np.float32)).to(device),
            ).cpu().numpy()
            v_n = model(
                torch.from_numpy(paper_emb[va_neg_s]).to(device),
                torch.from_numpy(paper_emb[va_neg_t]).to(device),
                torch.from_numpy(X_va_neg.astype(np.float32)).to(device),
            ).cpu().numpy()
        v_auc, _ = auc_ap(v_p, v_n)
        if v_auc > best_val_auc:
            best_val_auc = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        t_p = model(
            torch.from_numpy(paper_emb[te_pos_s]).to(device),
            torch.from_numpy(paper_emb[te_pos_t]).to(device),
            torch.from_numpy(X_te_pos.astype(np.float32)).to(device),
        ).cpu().numpy()
        t_n = model(
            torch.from_numpy(paper_emb[te_neg_s]).to(device),
            torch.from_numpy(paper_emb[te_neg_t]).to(device),
            torch.from_numpy(X_te_neg.astype(np.float32)).to(device),
        ).cpu().numpy()
    return t_p, t_n, best_val_auc


def paired_bootstrap_test(t22_pos, t22_neg, t2_pos, t2_neg, n_resamples=N_BOOTSTRAP, seed=0):
    """Bootstrap test: H0: AUC(T22) <= AUC(T2). Return p-value."""
    rng = np.random.default_rng(seed)
    n_pos = len(t22_pos)
    n_neg = len(t22_neg)

    obs_t22, _ = auc_ap(t22_pos, t22_neg)
    obs_t2, _ = auc_ap(t2_pos, t2_neg)
    obs_diff = obs_t22 - obs_t2

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        p_idx = rng.integers(0, n_pos, n_pos)
        n_idx = rng.integers(0, n_neg, n_neg)
        a, _ = auc_ap(t22_pos[p_idx], t22_neg[n_idx])
        b, _ = auc_ap(t2_pos[p_idx], t2_neg[n_idx])
        diffs[i] = a - b

    # 95% CI of difference
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    # One-sided p-value: P(diff <= 0)
    p_val = float((diffs <= 0).mean())
    return {
        "obs_diff": round(float(obs_diff), 6),
        "ci_low_95": round(float(ci_lo), 6),
        "ci_high_95": round(float(ci_hi), 6),
        "p_value_one_sided": round(p_val, 6),
        "n_bootstrap": n_resamples,
    }


def main():
    cfg = load_config()
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])
    models_root = Path(cfg["paths"]["models_dir"])
    graph_dir = Path(cfg["paths"]["graph_dir"])
    ret_dir = Path(cfg["paths"]["retrieval_dir"])

    log.info("loading graph + features")
    data = load_graph(str(graph_dir / "graph_data.pt"))
    paper_emb = data["paper"].x_abstract.cpu().numpy().astype(np.float32)
    edge_index = data["paper", "trajectory", "paper"].edge_index
    edge_weight = data["paper", "trajectory", "paper"].edge_attr[:, 4]
    eidx_np = edge_index.cpu().numpy()
    ew_np = edge_weight.cpu().numpy()
    n_papers = int(data["paper"].num_nodes)

    # Use existing T22 features.npz; this is the precomputed artifact from
    # script 67. It contains test_pos_s, test_pos_t, X_test_pos, etc.
    feat_path = models_root / "tier22_pair_mlp_expanded" / "features.npz"
    if not feat_path.exists():
        log.error("T22 features.npz not found at %s — run script 67 first", feat_path)
        return
    z = np.load(feat_path)
    X_tr_pos = z["X_train_pos"]; X_tr_neg = z["X_train_neg"]
    X_va_pos = z["X_val_pos"];  X_va_neg = z["X_val_neg"]
    X_te_pos = z["X_test_pos"]; X_te_neg = z["X_test_neg"]
    tr_pos_s = z["train_pos_s"]; tr_pos_t = z["train_pos_t"]
    tr_neg_s = z["train_neg_s"]; tr_neg_t = z["train_neg_t"]
    va_pos_s = z["val_pos_s"]; va_pos_t = z["val_pos_t"]
    va_neg_s = z["val_neg_s"]; va_neg_t = z["val_neg_t"]
    te_pos_s = z["test_pos_s"]; te_pos_t = z["test_pos_t"]
    te_neg_s = z["test_neg_s"]; te_neg_t = z["test_neg_t"]

    log.info("test set: %d positives, %d hard negatives", len(te_pos_s), len(te_neg_s))

    # T2 once (deterministic — no seed dependence)
    log.info("computing T2 aggregation (deterministic)")
    t2_emb = t2_aggregation(paper_emb, eidx_np, ew_np, n_papers)
    t2_pos_scores = cosine_similarity(t2_emb, te_pos_s, te_pos_t)
    t2_neg_scores = cosine_similarity(t2_emb, te_neg_s, te_neg_t)
    t2_auc, t2_ap = auc_ap(t2_pos_scores, t2_neg_scores)
    log.info("T2 AUC: %.4f  AP: %.4f", t2_auc, t2_ap)

    # T22 across 5 seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = []
    seed_test_logits = []  # for paired test
    for seed in SEEDS:
        log.info("=" * 60)
        log.info("seed %d", seed)
        log.info("=" * 60)
        t0 = time.time()
        t_p, t_n, best_val = train_one_seed(
            seed, X_tr_pos, X_tr_neg, X_va_pos, X_va_neg, X_te_pos, X_te_neg,
            paper_emb,
            tr_pos_s, tr_pos_t, tr_neg_s, tr_neg_t,
            va_pos_s, va_pos_t, va_neg_s, va_neg_t,
            te_pos_s, te_pos_t, te_neg_s, te_neg_t,
            device,
        )
        auc, ap = auc_ap(t_p, t_n)
        seed_results.append({
            "seed": seed, "test_auc": round(auc, 4), "test_ap": round(ap, 4),
            "best_val_auc": round(best_val, 4),
            "wall_seconds": round(time.time() - t0, 1),
        })
        seed_test_logits.append((t_p, t_n))
        log.info("seed %d → AUC=%.4f AP=%.4f (%.1fs)", seed, auc, ap, time.time() - t0)

    aucs = np.array([r["test_auc"] for r in seed_results])
    aps = np.array([r["test_ap"] for r in seed_results])
    log.info("=" * 60)
    log.info("T22 multi-seed: AUC = %.4f ± %.4f  AP = %.4f ± %.4f",
             aucs.mean(), aucs.std(), aps.mean(), aps.std())
    log.info("T2 baseline:    AUC = %.4f          AP = %.4f", t2_auc, t2_ap)
    log.info("=" * 60)

    # Paired bootstrap: average across seeds
    avg_t22_pos = np.mean([p for p, _ in seed_test_logits], axis=0)
    avg_t22_neg = np.mean([n for _, n in seed_test_logits], axis=0)
    boot = paired_bootstrap_test(avg_t22_pos, avg_t22_neg, t2_pos_scores, t2_neg_scores)
    log.info("paired bootstrap (T22 vs T2):")
    log.info("  obs Δ AUC = %.4f", boot["obs_diff"])
    log.info("  95%% CI    = [%.4f, %.4f]", boot["ci_low_95"], boot["ci_high_95"])
    log.info("  p-value   = %.6f (one-sided)", boot["p_value_one_sided"])

    out = {
        "task": "T22 multi-seed + paired-bootstrap test vs T2",
        "seeds": SEEDS,
        "n_test_positive": int(len(te_pos_s)),
        "n_test_negative": int(len(te_neg_s)),
        "t22_per_seed": seed_results,
        "t22_summary": {
            "auc_mean": round(float(aucs.mean()), 4),
            "auc_std":  round(float(aucs.std()),  4),
            "ap_mean":  round(float(aps.mean()),  4),
            "ap_std":   round(float(aps.std()),   4),
            "auc_min":  round(float(aucs.min()),  4),
            "auc_max":  round(float(aucs.max()),  4),
        },
        "t2_baseline": {
            "auc": round(t2_auc, 4),
            "ap":  round(t2_ap,  4),
        },
        "paired_bootstrap_t22_vs_t2": boot,
    }
    out_path = metrics_dir / "tier22_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("wrote: %s", out_path)

    # Markdown
    md = ["# T22 Multi-Seed Significance Test\n\n"]
    md.append(f"5 seeds: {SEEDS}\n")
    md.append(f"Test set: {len(te_pos_s):,} positives, {len(te_neg_s):,} hard negatives\n\n")
    md.append("## Per-seed results\n\n")
    md.append("| Seed | Best Val AUC | Test AUC | Test AP | Wall (s) |\n")
    md.append("|------|--------------|----------|---------|----------|\n")
    for r in seed_results:
        md.append(f"| {r['seed']} | {r['best_val_auc']:.4f} | "
                  f"{r['test_auc']:.4f} | {r['test_ap']:.4f} | {r['wall_seconds']:.1f} |\n")

    md.append("\n## Summary\n\n")
    md.append(f"- **T22 AUC**: {aucs.mean():.4f} ± {aucs.std():.4f}  ")
    md.append(f"(min {aucs.min():.4f}, max {aucs.max():.4f})\n")
    md.append(f"- **T22 AP**:  {aps.mean():.4f} ± {aps.std():.4f}\n")
    md.append(f"- **T2 baseline**: AUC = {t2_auc:.4f}, AP = {t2_ap:.4f}\n\n")

    md.append("## Significance test (paired bootstrap, T22 vs T2)\n\n")
    md.append(f"- Observed Δ AUC: {boot['obs_diff']:+.4f}\n")
    md.append(f"- 95% bootstrap CI: [{boot['ci_low_95']:+.4f}, {boot['ci_high_95']:+.4f}]\n")
    md.append(f"- One-sided p-value: {boot['p_value_one_sided']:.6f}\n")
    md.append(f"- N bootstrap resamples: {N_BOOTSTRAP:,}\n\n")

    md.append("## Reading\n\n")
    md.append("If the 95% CI excludes zero and p < 0.05, T22's improvement over T2 is\n")
    md.append("statistically significant.\n\n")
    md.append("If the CI includes zero, the lift is not distinguishable from sampling noise\n")
    md.append("and the paper should be honest about this.\n\n")
    md.append("This addresses all three reviewers' implicit concerns about the validity of\n")
    md.append("a single-seed point-estimate AUC comparison.\n")

    out_md = metrics_dir / "tier22_significance.md"
    out_md.write_text("".join(md))
    log.info("wrote: %s", out_md)


if __name__ == "__main__":
    main()
