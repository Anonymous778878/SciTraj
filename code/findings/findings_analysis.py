"""
SciTraj §6 Findings analysis.
Runs three analyses for the paper's new Findings section:
  1. Cross-venue idea flow (uses venue_onehot as-stored)
  2. Topic-level emergence via temporal_semantic edges
  3. Venue-level claim conventions (uses has_limit, has_future, has_dispute)

Writes figures to figures/ and tables to outputs/findings/.
Run again after corrected venue mapping is available - same outputs,
just different numbers.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from collections import Counter

# =============================================================================
# Config
# =============================================================================
PT_PATH = Path("data/graph/graph_data.pt")
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("outputs/findings"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# Venue mapping. Column 0 = ACL (largest); cols 1 and 2 differ in count
# (686 vs 594). Per App. D Table 12: ACL 23,599 / NeurIPS 686 / CVPR 594.
# So col 1 = NeurIPS, col 2 = CVPR.
VENUE_COLS = {0: "ACL", 1: "NeurIPS", 2: "CVPR"}
VENUE_ORDER = ["ACL", "NeurIPS", "CVPR"]

RELATION_MAP = {
    0: "direct_extension",
    1: "future_realized",
    2: "limit_addressed",
    3: "causal_extension",
    6: "temporal_semantic",
    8: "dispute",
}
FLOW_RELATIONS = {"direct_extension", "causal_extension"}

# Buckets chosen for the actual year distribution
# 2010-2017: 4,410 papers (early)
# 2018-2021: 10,292 papers (middle)
# 2022-2024: 13,200 papers (recent)
BUCKETS = [("2010-2017", 2010, 2017),
           ("2018-2021", 2018, 2021),
           ("2022-2024", 2022, 2024)]

RANDOM_SEED = 42

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 100,
})

# =============================================================================
# Load
# =============================================================================
print("=" * 70)
print("LOADING GRAPH")
print("=" * 70)
graph = torch.load(PT_PATH, weights_only=False)

paper = graph["paper"]
venue_oh = paper["venue_onehot"].numpy()
year = paper["year"].numpy().astype(int)
n_papers = paper.num_nodes

venue_idx = venue_oh.argmax(axis=1)
venue = np.array([VENUE_COLS[i] for i in venue_idx])

paper_meta = pd.DataFrame({
    "node_id": np.arange(n_papers),
    "venue": venue,
    "year": year,
})

print(f"Papers: {n_papers:,}")
print("\nVenue distribution (as-stored):")
print(paper_meta.groupby("venue").size().reindex(VENUE_ORDER))

# Edges
edge_store = graph[("paper", "trajectory", "paper")]
edge_index = edge_store["edge_index"].numpy()
edge_type = edge_store["edge_type"].numpy()

edges = pd.DataFrame({
    "src": edge_index[0],
    "tgt": edge_index[1],
    "relation": [RELATION_MAP[t] for t in edge_type],
})
edges = (
    edges
    .merge(paper_meta.rename(columns={
        "node_id": "src", "venue": "src_venue", "year": "src_year"}), on="src")
    .merge(paper_meta.rename(columns={
        "node_id": "tgt", "venue": "tgt_venue", "year": "tgt_year"}), on="tgt")
)

def assign_bucket(y):
    for name, lo, hi in BUCKETS:
        if lo <= y <= hi:
            return name
    return None

edges["bucket"] = edges["src_year"].apply(assign_bucket)
edges = edges.dropna(subset=["bucket"])
print(f"\nTotal edges with valid bucket: {len(edges):,}")

# =============================================================================
# ANALYSIS 1: Cross-venue idea flow
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: CROSS-VENUE IDEA FLOW")
print("=" * 70)

flow = edges[edges["relation"].isin(FLOW_RELATIONS)].copy()
print(f"Flow edges (direct + causal extension): {len(flow):,}")

print("\nFlow edges by (src_venue, tgt_venue):")
print(flow.groupby(["src_venue", "tgt_venue"]).size().unstack(fill_value=0)
      .reindex(index=VENUE_ORDER, columns=VENUE_ORDER, fill_value=0))

def flow_matrix(df, bucket):
    sub = df[df["bucket"] == bucket]
    if len(sub) == 0:
        return pd.DataFrame(0.0, index=VENUE_ORDER, columns=VENUE_ORDER)
    counts = sub.groupby(["src_venue", "tgt_venue"]).size().reset_index(name="count")
    totals = sub.groupby("src_venue").size().reset_index(name="src_total")
    counts = counts.merge(totals, on="src_venue")
    counts["rate"] = counts["count"] / counts["src_total"]
    m = counts.pivot(index="src_venue", columns="tgt_venue", values="rate")
    return m.reindex(index=VENUE_ORDER, columns=VENUE_ORDER).fillna(0)

matrices = {b[0]: flow_matrix(flow, b[0]) for b in BUCKETS}
for bname, m in matrices.items():
    print(f"\n--- Flow rates: {bname} ---")
    print(m.round(4))

# Long-form table
long_rows = []
for bname, m in matrices.items():
    for src in VENUE_ORDER:
        for tgt in VENUE_ORDER:
            long_rows.append({
                "bucket": bname, "src_venue": src, "tgt_venue": tgt,
                "rate": m.loc[src, tgt],
            })
pd.DataFrame(long_rows).to_csv(OUT_DIR / "flow_rates.csv", index=False)

# Directional summary
print("\n--- Cross-venue directional growth ---")
print(f"{'Direction':<25}" + "".join(f"{b[0]:>12}" for b in BUCKETS) + f"{'growth':>10}")
growth_rows = []
for a in VENUE_ORDER:
    for b in VENUE_ORDER:
        if a == b:
            continue
        rates = [matrices[bn[0]].loc[a, b] for bn in BUCKETS]
        growth = rates[-1] / rates[0] if rates[0] > 1e-9 else float("nan")
        row_str = f"{a:>10} -> {b:<10}" + "".join(f"{r:>12.4f}" for r in rates)
        row_str += f"{growth:>9.2f}x" if not np.isnan(growth) else f"{'N/A':>10}"
        print(row_str)
        growth_rows.append({"src": a, "tgt": b,
                            "rate_2010_17": rates[0],
                            "rate_2018_21": rates[1],
                            "rate_2022_24": rates[2],
                            "growth": growth})
pd.DataFrame(growth_rows).to_csv(OUT_DIR / "flow_growth.csv", index=False)

# Year-shuffle check on cross-venue temporal variance
print("\n--- Year-shuffle sanity check on cross-venue flow ---")
def temporal_variance(mats):
    variances = []
    for src in VENUE_ORDER:
        for tgt in VENUE_ORDER:
            if src == tgt:
                continue
            rates = [mats[bn[0]].loc[src, tgt] for bn in BUCKETS]
            variances.append(np.var(rates))
    return float(np.mean(variances))

real_var = temporal_variance(matrices)
rng = np.random.default_rng(RANDOM_SEED)
shuf_vars = []
for rep in range(5):
    permuted = rng.permutation(paper_meta["year"].values)
    shuf_meta = paper_meta.copy()
    shuf_meta["year"] = permuted
    shuf_edges = (
        flow.drop(columns=["src_year", "tgt_year", "bucket"])
        .merge(shuf_meta[["node_id", "year"]].rename(
            columns={"node_id": "src", "year": "src_year"}), on="src")
        .merge(shuf_meta[["node_id", "year"]].rename(
            columns={"node_id": "tgt", "year": "tgt_year"}), on="tgt")
    )
    shuf_edges["bucket"] = shuf_edges["src_year"].apply(assign_bucket)
    shuf_edges = shuf_edges.dropna(subset=["bucket"])
    shuf_mats = {b[0]: flow_matrix(shuf_edges, b[0]) for b in BUCKETS}
    shuf_vars.append(temporal_variance(shuf_mats))

shuf_mean = np.mean(shuf_vars)
shuf_std = np.std(shuf_vars)
ratio = real_var / shuf_mean if shuf_mean > 1e-12 else float("inf")
print(f"Real temporal variance:        {real_var:.6e}")
print(f"Shuffled temporal variance:    {shuf_mean:.6e} +/- {shuf_std:.6e}")
print(f"Ratio (real/shuffled):         {ratio:.2f}x")
if ratio > 2:
    print("STRONG temporal signal in cross-venue flow.")
elif ratio > 1.3:
    print("Moderate temporal signal.")
else:
    print("WEAK temporal signal. Re-run after venue correction.")

# Figure 1: cross-venue heatmaps
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
vmax = max(m.values.max() for m in matrices.values())
for i, (ax, b) in enumerate(zip(axes, BUCKETS)):
    bname = b[0]
    m = matrices[bname]
    sns.heatmap(
        m, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=vmax,
        cbar=(i == len(BUCKETS) - 1),
        cbar_kws={"label": "flow rate"} if i == len(BUCKETS) - 1 else None,
        square=True, ax=ax,
        annot_kws={"size": 10}, linewidths=0.5, linecolor="white",
    )
    ax.set_title(bname)
    ax.set_xlabel("Cited venue")
    ax.set_ylabel("Citing venue" if i == 0 else "")
fig.suptitle(
    "Cross-venue idea flow (direct + causal extension), per outgoing flow edge",
    y=1.04)
plt.tight_layout()
plt.savefig(FIG_DIR / "cross_venue_flow.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "cross_venue_flow.png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Figure: {FIG_DIR}/cross_venue_flow.pdf")

# =============================================================================
# ANALYSIS 2: Topic-level emergence
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: TOPIC-LEVEL EMERGENCE VIA temporal_semantic EDGES")
print("=" * 70)

topic_membership = paper["topic_membership"].numpy()  # (24879, 30)
n_topics = topic_membership.shape[1]
hard_topic = topic_membership.argmax(axis=1)
print(f"Topics: {n_topics}")
print(f"Hard-topic distribution (top 10):")
print(pd.Series(Counter(hard_topic.tolist())).sort_values(ascending=False).head(10))

# temporal_semantic edges: for each edge, the *target* topic is the one being
# "updated" for a new era. Count by target topic and citing year.
ts_edges = edges[edges["relation"] == "temporal_semantic"].copy()
print(f"\ntemporal_semantic edges: {len(ts_edges):,}")

ts_edges["tgt_topic"] = hard_topic[ts_edges["tgt"].values]
# For each (topic, citing_year), count edges
ts_counts = (
    ts_edges.groupby(["tgt_topic", "src_year"])
    .size().reset_index(name="count")
)

# For each topic, total temporal_semantic citations - find spikiest ones
topic_totals = ts_counts.groupby("tgt_topic")["count"].sum().sort_values(ascending=False)
print(f"\nTop 10 topics by temporal_semantic-as-target count:")
print(topic_totals.head(10))

# Identify "emerging" topics: peak year shifted late, high concentration in recent years
def emergence_score(group):
    """High score = high recent concentration."""
    g = group.set_index("src_year")["count"]
    total = g.sum()
    if total < 50:
        return -1
    recent = g.reindex(range(2020, 2025), fill_value=0).sum()
    return recent / total

topic_emergence = (
    ts_counts.groupby("tgt_topic")
    .apply(lambda g: pd.Series({
        "total": g["count"].sum(),
        "recent_share": g[g["src_year"] >= 2020]["count"].sum() / g["count"].sum() if g["count"].sum() > 0 else 0,
        "peak_year": g.loc[g["count"].idxmax(), "src_year"] if len(g) > 0 else None,
    }))
    .reset_index()
)
topic_emergence = topic_emergence[topic_emergence["total"] >= 50].sort_values(
    "recent_share", ascending=False)
print(f"\nTopics with strongest recent emergence (total >= 50):")
print(topic_emergence.head(10).to_string(index=False))
topic_emergence.to_csv(OUT_DIR / "topic_emergence.csv", index=False)

# Plot top 6 emerging topics
top_topics = topic_emergence.head(6)["tgt_topic"].tolist()
fig, axes = plt.subplots(2, 3, figsize=(11.5, 5.5), sharex=True)
years_all = sorted(ts_edges["src_year"].unique())
for ax, t in zip(axes.flat, top_topics):
    g = ts_counts[ts_counts["tgt_topic"] == t].set_index("src_year")["count"]
    g = g.reindex(years_all, fill_value=0)
    ax.bar(g.index, g.values, color="#4C72B0", edgecolor="white")
    ax.set_title(f"Topic {t}")
    ax.set_xlabel("Citing year")
    ax.set_ylabel("temporal_semantic edges")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle(
    "Topic-level emergence via temporal_semantic edges (top 6 by recent concentration)",
    y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "topic_emergence.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "topic_emergence.png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Figure: {FIG_DIR}/topic_emergence.pdf")

# Heatmap: topic x year intensity (all topics)
heatmap_data = (
    ts_counts.pivot(index="tgt_topic", columns="src_year", values="count")
    .reindex(columns=years_all).fillna(0)
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap="YlOrRd", cbar_kws={"label": "edges"}, ax=ax)
ax.set_xlabel("Citing year")
ax.set_ylabel("Target topic (LDA component)")
ax.set_title("temporal_semantic edge intensity across topics and years")
plt.tight_layout()
plt.savefig(FIG_DIR / "topic_year_heatmap.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "topic_year_heatmap.png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Figure: {FIG_DIR}/topic_year_heatmap.pdf")

# =============================================================================
# ANALYSIS 3: Venue claim conventions
# =============================================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: VENUE-LEVEL CLAIM CONVENTIONS")
print("=" * 70)

has_limit = paper["has_limit"].numpy().flatten() > 0.5
has_future = paper["has_future"].numpy().flatten() > 0.5
has_dispute = paper["has_dispute"].numpy().flatten() > 0.5

paper_meta_signals = paper_meta.copy()
paper_meta_signals["has_limit"] = has_limit
paper_meta_signals["has_future"] = has_future
paper_meta_signals["has_dispute"] = has_dispute

conv = paper_meta_signals.groupby("venue")[
    ["has_limit", "has_future", "has_dispute"]].mean()
conv = conv.reindex(VENUE_ORDER)
print("Fraction of papers with each claim type, by venue:")
print(conv.round(3))
conv.to_csv(OUT_DIR / "claim_conventions.csv")

# Also compute the ACL/NeurIPS ratio for limitations
if "ACL" in conv.index and "NeurIPS" in conv.index:
    acl_lim = conv.loc["ACL", "has_limit"]
    nips_lim = conv.loc["NeurIPS", "has_limit"]
    if nips_lim > 1e-9:
        print(f"\nACL/NeurIPS limitation-discussion ratio: {acl_lim/nips_lim:.2f}x")

# Figure 3: grouped bar chart
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(VENUE_ORDER))
width = 0.25
for i, sig in enumerate(["has_limit", "has_future", "has_dispute"]):
    ax.bar(x + i*width - width, conv[sig].values, width,
           label=sig.replace("has_", "").replace("_", " "))
ax.set_xticks(x)
ax.set_xticklabels(VENUE_ORDER)
ax.set_ylabel("Fraction of papers")
ax.set_title("Claim conventions by venue")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "claim_conventions.pdf", bbox_inches="tight")
plt.savefig(FIG_DIR / "claim_conventions.png", bbox_inches="tight", dpi=300)
plt.close()
print(f"Figure: {FIG_DIR}/claim_conventions.pdf")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF FINDINGS")
print("=" * 70)
print(f"""
Files written:
  {FIG_DIR}/cross_venue_flow.pdf/.png
  {FIG_DIR}/topic_emergence.pdf/.png
  {FIG_DIR}/topic_year_heatmap.pdf/.png
  {FIG_DIR}/claim_conventions.pdf/.png
  {OUT_DIR}/flow_rates.csv
  {OUT_DIR}/flow_growth.csv
  {OUT_DIR}/topic_emergence.csv
  {OUT_DIR}/claim_conventions.csv

Top numbers to consider for §6:
  Cross-venue temporal-variance ratio (real/shuffled): {ratio:.2f}x
  Top emerging topic by recent share:
    Topic {topic_emergence.iloc[0]['tgt_topic']:.0f}, "
        f"{topic_emergence.iloc[0]['recent_share']*100:.0f}% of edges in 2020-2024
""")
if "ACL" in conv.index and "NeurIPS" in conv.index and nips_lim > 1e-9:
    print(f"  Claim-convention asymmetry: ACL papers discuss limitations "
          f"{acl_lim/nips_lim:.1f}x more than NeurIPS papers")
print("\nDone.")
