#!/usr/bin/env python3
"""
SciTraj: Cross-venue idea-flow analysis for the §6 Findings section.

Computes directed citation flow rates between ACL, NeurIPS, and CVPR
across three time buckets (2010-2014, 2015-2019, 2020-2024) using
direct_extension + causal_extension edges. Produces:
  - A heatmap figure (cross_venue_flow.pdf / .png)
  - Numerical tables (flow_matrices.csv, asymmetry_ratios.csv)
  - A year-shuffle sanity check confirming the pattern is temporal

Usage:
    python cross_venue_flow.py

Expects:
    data/graph/typed_edges.json  OR  data/graph/graph_data.pt
    data/graph/paper_meta.csv    (will be derived from graph_data.pt if missing)

Outputs into figures/ and outputs/findings/
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data/graph")
FIG_DIR = Path("figures")
OUT_DIR = Path("outputs/findings")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

VENUE_ORDER = ["ACL", "NeurIPS", "CVPR"]
FLOW_RELATIONS = {"direct_extension", "causal_extension"}
BUCKETS = [
    ("2010-2014", 2010, 2014),
    ("2015-2019", 2015, 2019),
    ("2020-2024", 2020, 2024),
]
RANDOM_SEED = 42


# =============================================================================
# Data loading
# =============================================================================

def normalize_venue(v):
    """Map raw venue strings to canonical ACL / NeurIPS / CVPR labels."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower()
    if "acl" in s or "emnlp" in s or "naacl" in s or "eacl" in s:
        return "ACL"
    if "neurips" in s or "nips" in s:
        return "NeurIPS"
    if "cvpr" in s:
        return "CVPR"
    return None  # Drop other venues from this analysis


def load_paper_meta():
    """Load paper metadata (paper_id, venue, year). Try CSV first, fall back
    to extracting from graph_data.pt."""
    csv_path = DATA_DIR / "paper_meta.csv"
    if csv_path.exists():
        print(f"Loading paper metadata from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"paper_meta.csv not found; trying to extract from graph_data.pt")
        df = extract_paper_meta_from_pt()
        df.to_csv(csv_path, index=False)
        print(f"Cached paper metadata to {csv_path}")

    # Normalize venue
    df["venue"] = df["venue"].apply(normalize_venue)
    df = df.dropna(subset=["venue", "year"])
    df["year"] = df["year"].astype(int)
    df["paper_id"] = df["paper_id"].astype(str)
    print(f"Loaded {len(df):,} papers with valid venue+year")
    print(df.groupby("venue").size().rename("count"))
    return df


def extract_paper_meta_from_pt():
    """Extract paper_id, venue, year from PyTorch Geometric HeteroData object."""
    pt_path = DATA_DIR / "graph_data.pt"
    if not pt_path.exists():
        sys.exit(f"ERROR: neither paper_meta.csv nor graph_data.pt found in {DATA_DIR}")
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: graph_data.pt found but torch not installed")

    print(f"Loading {pt_path} (this may take a moment)")
    graph = torch.load(pt_path, weights_only=False)

    # HeteroData typically stores paper info under a 'paper' node type.
    # Adjust the access pattern to whatever your graph object actually uses.
    if hasattr(graph, "paper"):
        node = graph["paper"]
    elif "paper" in graph.node_types:
        node = graph["paper"]
    else:
        node = graph  # fallback: assume flat

    paper_ids = list(node.get("paper_id", []))
    if not paper_ids:
        paper_ids = list(node.get("id", []))
    if not paper_ids:
        sys.exit("ERROR: could not find paper IDs in graph_data.pt; "
                 "save a paper_meta.csv manually with columns: paper_id, venue, year")

    venues = list(node.get("venue", [None] * len(paper_ids)))
    years_tensor = node.get("year")
    if years_tensor is not None and hasattr(years_tensor, "numpy"):
        years = years_tensor.cpu().numpy().tolist()
    else:
        years = list(years_tensor) if years_tensor is not None else [None] * len(paper_ids)

    return pd.DataFrame({
        "paper_id": paper_ids,
        "venue": venues,
        "year": years,
    })


def load_typed_edges():
    """Load typed edges from JSON."""
    edges_path = DATA_DIR / "typed_edges.json"
    if not edges_path.exists():
        sys.exit(f"ERROR: {edges_path} not found")

    print(f"Loading typed edges from {edges_path}")
    with open(edges_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "edges" in raw:
        raw = raw["edges"]

    df = pd.DataFrame(raw)
    # Try common column-name variants
    rename_map = {}
    for src_key in ["source_id", "source", "src", "src_id", "from"]:
        if src_key in df.columns:
            rename_map[src_key] = "source_id"
            break
    for tgt_key in ["target_id", "target", "tgt", "tgt_id", "to"]:
        if tgt_key in df.columns:
            rename_map[tgt_key] = "target_id"
            break
    for rel_key in ["relation_type", "relation", "rel", "edge_type", "type"]:
        if rel_key in df.columns:
            rename_map[rel_key] = "relation_type"
            break
    df = df.rename(columns=rename_map)

    required = {"source_id", "target_id", "relation_type"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: typed_edges.json missing required columns: {missing}\n"
                 f"Available columns: {list(df.columns)}")

    df["source_id"] = df["source_id"].astype(str)
    df["target_id"] = df["target_id"].astype(str)
    print(f"Loaded {len(df):,} edges")
    print("Relation distribution:")
    print(df.groupby("relation_type").size().sort_values(ascending=False))
    return df


# =============================================================================
# Flow computation
# =============================================================================

def assign_bucket(year):
    for name, lo, hi in BUCKETS:
        if lo <= year <= hi:
            return name
    return None


def compute_flow_matrices(edges, label=""):
    """Returns dict[bucket_name -> DataFrame(src_venue x tgt_venue, flow_rate)]
    and a long-form DataFrame with raw counts and rates."""
    counts = (
        edges
        .groupby(["bucket", "src_venue", "tgt_venue"])
        .size()
        .reset_index(name="count")
    )
    src_totals = (
        edges
        .groupby(["bucket", "src_venue"])
        .size()
        .reset_index(name="src_total")
    )
    counts = counts.merge(src_totals, on=["bucket", "src_venue"])
    counts["flow_rate"] = counts["count"] / counts["src_total"]

    matrices = {}
    for bname, _, _ in BUCKETS:
        sub = counts[counts["bucket"] == bname]
        m = (
            sub.pivot(index="src_venue", columns="tgt_venue", values="flow_rate")
            .reindex(index=VENUE_ORDER, columns=VENUE_ORDER)
            .fillna(0)
        )
        matrices[bname] = m
        print(f"\n=== {label} {bname} (flow rates) ===")
        print(m.round(3))

    return matrices, counts


def compute_asymmetry_table(matrices):
    """For each unordered venue pair, compute fwd/rev ratio per bucket."""
    pairs = [("ACL", "CVPR"), ("ACL", "NeurIPS"), ("NeurIPS", "CVPR")]
    rows = []
    for a, b in pairs:
        for bname, _, _ in BUCKETS:
            m = matrices[bname]
            fwd = m.loc[a, b] if a in m.index and b in m.columns else 0.0
            rev = m.loc[b, a] if b in m.index and a in m.columns else 0.0
            ratio = fwd / rev if rev > 1e-9 else float("inf")
            rows.append({
                "pair": f"{a} <-> {b}",
                "bucket": bname,
                "fwd_rate": fwd,
                "rev_rate": rev,
                "ratio_fwd_over_rev": ratio,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Year-shuffle sanity check
# =============================================================================

def year_shuffle_check(flow_edges, paper_meta, n_repeats=5):
    """Permute the year array across papers, recompute matrices, and verify
    that cross-venue rates flatten across buckets. Returns mean over repeats
    of the cross-venue temporal variance under shuffling vs. real."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Real variance of ACL->CVPR rate across buckets
    real_mats, _ = compute_flow_matrices(flow_edges, label="real")
    def cross_variance(mats):
        rates = []
        for bname, _, _ in BUCKETS:
            m = mats[bname]
            for a, b in [("ACL", "CVPR"), ("ACL", "NeurIPS"),
                         ("NeurIPS", "CVPR"), ("CVPR", "ACL"),
                         ("NeurIPS", "ACL"), ("CVPR", "NeurIPS")]:
                if a in m.index and b in m.columns:
                    rates.append((a, b, bname, m.loc[a, b]))
        df = pd.DataFrame(rates, columns=["src", "tgt", "bucket", "rate"])
        return df.groupby(["src", "tgt"])["rate"].var().mean()

    real_var = cross_variance(real_mats)

    shuf_vars = []
    for rep in range(n_repeats):
        permuted_years = rng.permutation(paper_meta["year"].values)
        shuf_meta = paper_meta.copy()
        shuf_meta["year"] = permuted_years
        shuf_meta["paper_id"] = paper_meta["paper_id"].values  # keep IDs fixed

        # Rejoin years onto edges
        shuf_edges = (
            flow_edges.drop(columns=["src_year", "tgt_year", "bucket"], errors="ignore")
            .merge(
                shuf_meta.rename(columns={"paper_id": "source_id", "year": "src_year"})[["source_id", "src_year"]],
                on="source_id",
            )
            .merge(
                shuf_meta.rename(columns={"paper_id": "target_id", "year": "tgt_year"})[["target_id", "tgt_year"]],
                on="target_id",
            )
        )
        shuf_edges["bucket"] = shuf_edges["src_year"].apply(assign_bucket)
        shuf_edges = shuf_edges.dropna(subset=["bucket"])
        shuf_mats, _ = compute_flow_matrices(shuf_edges, label=f"shuf-{rep}")
        shuf_vars.append(cross_variance(shuf_mats))

    return real_var, float(np.mean(shuf_vars)), float(np.std(shuf_vars))


# =============================================================================
# Figure
# =============================================================================

def plot_heatmap(matrices, output_path):
    """Three side-by-side heatmaps, one per bucket, shared color scale."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    vmax = max(m.values.max() for m in matrices.values())

    for i, (ax, (bname, _, _)) in enumerate(zip(axes, BUCKETS)):
        m = matrices[bname]
        sns.heatmap(
            m,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            cbar=(i == len(BUCKETS) - 1),
            cbar_kws={"label": "flow rate"} if i == len(BUCKETS) - 1 else None,
            square=True,
            ax=ax,
            annot_kws={"size": 9},
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(bname, fontsize=11)
        ax.set_xlabel("Cited venue", fontsize=10)
        if i == 0:
            ax.set_ylabel("Citing venue", fontsize=10)
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        "Cross-venue idea flow (direct + causal extension), per outgoing flow edge",
        fontsize=11,
        y=1.04,
    )
    plt.tight_layout()
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"\nFigure saved to {output_path}.pdf and .png")


# =============================================================================
# Sanity checks
# =============================================================================

def sanity_checks(flow_edges, matrices):
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # 1. Source-venue distribution
    print("\n[1] Edges originating per source venue (expect ~60/20/20 ACL/NIPS/CVPR):")
    counts = flow_edges.groupby("src_venue").size()
    total = counts.sum()
    for v in VENUE_ORDER:
        if v in counts.index:
            pct = 100 * counts[v] / total
            print(f"    {v:<10}: {counts[v]:>8,} ({pct:.1f}%)")

    # 2. Self-loop dominance
    print("\n[2] Self-loop dominance check (intra-venue flow should be highest):")
    any_violation = False
    for bname, _, _ in BUCKETS:
        m = matrices[bname]
        for v in VENUE_ORDER:
            if v not in m.index:
                continue
            row = m.loc[v]
            self_rate = row.get(v, 0)
            max_other = max((row.get(o, 0) for o in VENUE_ORDER if o != v), default=0)
            mark = "OK" if self_rate >= max_other else "VIOLATION"
            if mark == "VIOLATION":
                any_violation = True
            print(f"    {bname} {v}->{v}={self_rate:.3f} vs max other={max_other:.3f}  [{mark}]")
    if any_violation:
        print("    WARNING: at least one venue cites cross-venue more than itself")
    else:
        print("    All self-loops dominate (expected for intra-community citation)")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("SciTraj cross-venue idea-flow analysis")
    print("=" * 60)

    paper_meta = load_paper_meta()
    edges = load_typed_edges()

    # Join venue and year onto both endpoints
    print("\nJoining venue and year onto edges...")
    edges = (
        edges
        .merge(
            paper_meta.rename(columns={
                "paper_id": "source_id",
                "venue": "src_venue",
                "year": "src_year",
            }),
            on="source_id",
            how="inner",
        )
        .merge(
            paper_meta.rename(columns={
                "paper_id": "target_id",
                "venue": "tgt_venue",
                "year": "tgt_year",
            }),
            on="target_id",
            how="inner",
        )
    )
    print(f"Edges with both endpoints resolved to known venues: {len(edges):,}")
    print(edges.groupby(["src_venue", "tgt_venue"]).size().unstack(fill_value=0))

    # Filter to idea-flow relations
    flow_edges = edges[edges["relation_type"].isin(FLOW_RELATIONS)].copy()
    print(f"\nFlow edges (direct + causal extension): {len(flow_edges):,}")

    flow_edges["bucket"] = flow_edges["src_year"].apply(assign_bucket)
    flow_edges = flow_edges.dropna(subset=["bucket"])
    print(f"Flow edges in defined buckets: {len(flow_edges):,}")
    print(flow_edges.groupby("bucket").size())

    # Compute flow matrices
    matrices, counts_long = compute_flow_matrices(flow_edges, label="real")
    counts_long.to_csv(OUT_DIR / "flow_matrices_long.csv", index=False)
    print(f"\nLong-form counts saved to {OUT_DIR / 'flow_matrices_long.csv'}")

    # Asymmetry table
    asym = compute_asymmetry_table(matrices)
    print("\n" + "=" * 60)
    print("ASYMMETRY TABLE")
    print("=" * 60)
    print(asym.to_string(index=False))
    asym.to_csv(OUT_DIR / "asymmetry_ratios.csv", index=False)

    # Sanity checks
    sanity_checks(flow_edges, matrices)

    # Year-shuffle sanity check
    print("\n" + "=" * 60)
    print("YEAR-SHUFFLE SANITY CHECK")
    print("=" * 60)
    print("If cross-venue temporal patterns are real, shuffling should flatten")
    print("the variance of cross-venue flow rates across buckets.")
    print("Running 5 shuffles (this may take a minute)...\n")
    real_var, shuf_var_mean, shuf_var_std = year_shuffle_check(
        flow_edges, paper_meta, n_repeats=5
    )
    print(f"\nMean cross-venue rate variance across buckets:")
    print(f"  Real:      {real_var:.6f}")
    print(f"  Shuffled:  {shuf_var_mean:.6f} ± {shuf_var_std:.6f}")
    ratio = real_var / shuf_var_mean if shuf_var_mean > 1e-12 else float("inf")
    print(f"  Real / Shuffled: {ratio:.2f}x")
    if ratio > 2:
        print("  -> Strong temporal signal: real variance >> shuffled. PUBLISHABLE.")
    elif ratio > 1.3:
        print("  -> Moderate temporal signal. Reportable but less striking.")
    else:
        print("  -> Weak temporal signal. Reconsider the framing.")

    # Figure
    plot_heatmap(matrices, FIG_DIR / "cross_venue_flow")

    # Summary for the paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR §6 FINDINGS SECTION")
    print("=" * 60)
    print("\nKey numbers to consider citing in the paper:")
    for a, b in [("ACL", "CVPR"), ("ACL", "NeurIPS"), ("NeurIPS", "CVPR")]:
        rates_fwd = [matrices[bn].loc[a, b] for bn, _, _ in BUCKETS
                     if a in matrices[bn].index and b in matrices[bn].columns]
        rates_rev = [matrices[bn].loc[b, a] for bn, _, _ in BUCKETS
                     if b in matrices[bn].index and a in matrices[bn].columns]
        if len(rates_fwd) == 3 and rates_fwd[0] > 1e-9:
            growth_fwd = rates_fwd[-1] / rates_fwd[0]
        else:
            growth_fwd = float("nan")
        if len(rates_rev) == 3 and rates_rev[0] > 1e-9:
            growth_rev = rates_rev[-1] / rates_rev[0]
        else:
            growth_rev = float("nan")
        print(f"  {a} -> {b}: rates {[round(r, 3) for r in rates_fwd]}, "
              f"growth 2010-14 to 2020-24: {growth_fwd:.2f}x")
        print(f"  {b} -> {a}: rates {[round(r, 3) for r in rates_rev]}, "
              f"growth 2010-14 to 2020-24: {growth_rev:.2f}x")

    print("\nDone. Files written to figures/ and outputs/findings/")


if __name__ == "__main__":
    main()