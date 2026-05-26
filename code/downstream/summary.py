"""
94_downstream_summary.py

Aggregates results from all three downstream-task JSONs into a single
summary used by §8 of the paper and by the rebuttal letter.

Inputs:
  outputs/metrics/e10a_citation_augmentation.json
  outputs/metrics/e10b_future_citation_prediction.json
  outputs/metrics/e10c_typed_relation_classification.json

Outputs:
  outputs/metrics/downstream_summary.json
  outputs/metrics/downstream_summary.md
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs/metrics"


def load_optional(name):
    p = METRICS_DIR / name
    if not p.exists():
        print(f"  WARNING: {p} not found; skipping.")
        return None
    with open(p) as f:
        return json.load(f)


def main():
    a = load_optional("e10a_citation_augmentation.json")
    b = load_optional("e10b_future_citation_prediction.json")
    c = load_optional("e10c_typed_relation_classification.json")

    summary = {"tasks": {}}

    # Task A
    if a:
        modes = a.get("modes", {})
        bl = modes.get("baseline_specter2_only", {}).get("aggregate", {})
        t22 = modes.get("t22", {}).get("aggregate", {})
        summary["tasks"]["A_citation_augmentation"] = {
            "n_queries": a.get("config", {}).get("n_queries"),
            "baseline_specter2_only": {
                "recall_at_10": bl.get("mean_recall_at_10"),
                "recall_at_20": bl.get("mean_recall_at_20"),
                "mrr": bl.get("mean_mrr"),
                "ndcg_at_10": bl.get("mean_ndcg_at_10")
            },
            "t22": {
                "recall_at_10": t22.get("mean_recall_at_10"),
                "recall_at_20": t22.get("mean_recall_at_20"),
                "mrr": t22.get("mean_mrr"),
                "ndcg_at_10": t22.get("mean_ndcg_at_10")
            },
            "lift": a.get("lift_t22_over_baseline", {})
        }

    # Task B
    if b:
        per_year = {}
        for Y, r in b.get("results_by_year", {}).items():
            per_year[Y] = {
                "specter2_baseline_auc": r["specter2_baseline"]["auc"],
                "t22_retrained_auc": r["t22_retrained"]["auc"],
                "specter2_baseline_ap": r["specter2_baseline"]["ap"],
                "t22_retrained_ap": r["t22_retrained"]["ap"],
                "delta_auc": r["t22_retrained"]["auc"] - r["specter2_baseline"]["auc"],
                "n_test_edges": r["n_test_edges"]
            }
        summary["tasks"]["B_future_citation_prediction"] = {
            "test_years": list(per_year.keys()),
            "per_year": per_year
        }

    # Task C
    if c:
        results = c.get("results", {})
        summary["tasks"]["C_typed_relation_classification"] = {
            "n_classes": 9,
            "n_train": c.get("config", {}).get("n_train"),
            "n_test": c.get("config", {}).get("n_test"),
            "random_macro_f1": results.get("random_class_prior", {}).get("macro_f1"),
            "specter2_macro_f1": results.get("specter2_only_mlp", {}).get("macro_f1"),
            "t22_macro_f1": results.get("t22_engineered_features", {}).get("macro_f1"),
            "t22_per_class": results.get("t22_engineered_features", {}).get("per_class"),
            "lift_macro_f1": c.get("lift_t22_over_specter2", {}).get("macro_f1_delta")
        }

    # Save JSON
    out_json = METRICS_DIR / "downstream_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")

    # Markdown report
    lines = ["# Downstream Tasks — Summary Report", ""]

    # Task A
    if "A_citation_augmentation" in summary["tasks"]:
        ta = summary["tasks"]["A_citation_augmentation"]
        lines.append("## Task A: Citation Augmentation")
        lines.append("")
        lines.append(f"Held-out queries: {ta['n_queries']}")
        lines.append("")
        lines.append("| Method | Recall@10 | Recall@20 | MRR | NDCG@10 |")
        lines.append("|--------|-----------|-----------|-----|---------|")
        b = ta["baseline_specter2_only"]
        t = ta["t22"]
        def fmt(x):
            return f"{x:.4f}" if x is not None else "—"
        lines.append(f"| SPECTER2 baseline | {fmt(b['recall_at_10'])} | "
                     f"{fmt(b['recall_at_20'])} | {fmt(b['mrr'])} | "
                     f"{fmt(b['ndcg_at_10'])} |")
        lines.append(f"| **T22 (typed graph)** | **{fmt(t['recall_at_10'])}** | "
                     f"**{fmt(t['recall_at_20'])}** | **{fmt(t['mrr'])}** | "
                     f"**{fmt(t['ndcg_at_10'])}** |")
        lines.append("")
        if ta["lift"]:
            lines.append("**Lift T22 over SPECTER2:**")
            for metric, vals in ta["lift"].items():
                lines.append(f"- {metric}: Δ = {vals['absolute_delta']:+.4f} "
                             f"({vals['relative_delta']:+.1%})")
        lines.append("")

    # Task B
    if "B_future_citation_prediction" in summary["tasks"]:
        tb = summary["tasks"]["B_future_citation_prediction"]
        lines.append("## Task B: Future Citation Prediction")
        lines.append("")
        lines.append("| Test Year | SPECTER2 AUC | T22 AUC | Δ AUC | N_test |")
        lines.append("|-----------|--------------|---------|-------|--------|")
        for Y, r in tb["per_year"].items():
            lines.append(f"| {Y} | {r['specter2_baseline_auc']:.4f} | "
                         f"{r['t22_retrained_auc']:.4f} | "
                         f"{r['delta_auc']:+.4f} | {r['n_test_edges']} |")
        lines.append("")

    # Task C
    if "C_typed_relation_classification" in summary["tasks"]:
        tc = summary["tasks"]["C_typed_relation_classification"]
        lines.append("## Task C: Typed Relation Classification")
        lines.append("")
        lines.append(f"9-way multi-class classification, train={tc['n_train']}, "
                     f"test={tc['n_test']}")
        lines.append("")
        lines.append("| Method | Macro-F1 |")
        lines.append("|--------|----------|")
        lines.append(f"| Random (class prior) | {tc['random_macro_f1']:.4f} |")
        lines.append(f"| SPECTER2-only MLP | {tc['specter2_macro_f1']:.4f} |")
        lines.append(f"| **T22 (63 engineered features)** | "
                     f"**{tc['t22_macro_f1']:.4f}** |")
        lines.append("")
        if tc.get("t22_per_class"):
            lines.append("### T22 per-class breakdown")
            lines.append("")
            lines.append("| Relation | Precision | Recall | F1 | Support |")
            lines.append("|----------|-----------|--------|----|---------|")
            for name, m in tc["t22_per_class"].items():
                lines.append(f"| {name} | {m['precision']:.3f} | "
                             f"{m['recall']:.3f} | {m['f1']:.3f} | "
                             f"{m['support']} |")
            lines.append("")

    out_md = METRICS_DIR / "downstream_summary.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {out_md}")

    print("\n=== Console summary ===")
    if a:
        ta = summary["tasks"].get("A_citation_augmentation", {})
        bl = ta.get("baseline_specter2_only", {})
        tt = ta.get("t22", {})
        print(f"Task A — Citation Aug: SPECTER2 R@10={bl.get('recall_at_10', '—')}, "
              f"T22 R@10={tt.get('recall_at_10', '—')}")
    if b:
        for Y, r in summary["tasks"].get("B_future_citation_prediction", {}).get("per_year", {}).items():
            print(f"Task B — Year {Y}: SPECTER2 AUC={r['specter2_baseline_auc']:.4f}, "
                  f"T22 AUC={r['t22_retrained_auc']:.4f}, Δ={r['delta_auc']:+.4f}")
    if c:
        tc = summary["tasks"].get("C_typed_relation_classification", {})
        print(f"Task C — 9-way classification: SPECTER2 macro-F1={tc.get('specter2_macro_f1', '—')}, "
              f"T22 macro-F1={tc.get('t22_macro_f1', '—')}")


if __name__ == "__main__":
    main()
