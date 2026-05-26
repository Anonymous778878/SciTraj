"""
compute_trajectory_stats.py

Compute corpus-wide trajectory statistics for §3.3.

DEFINITION (committed):
A trajectory is a directed simple path in the typed-edge graph where:
  (a) edge types are drawn from the "progression" set:
      {direct_extension, future_realized, causal_extension, limit_addressed}
  (b) papers along the path are publication-time-ordered
      (year[i+1] >= year[i] for all i)
  (c) papers are pairwise distinct (no node revisits within a path)

Rationale: dispute and temporal_semantic edges are "sideways" or
"reframing" relations rather than progression; excluding them isolates
the trajectories that represent forward research narratives.

We report two trajectory definitions:
  - Strict (progression types only): the headline definition
  - Inclusive (all 6 types): reported as a sensitivity check in appendix

OUTPUTS:
  outputs/metrics/trajectory_stats.json    — all stats for §3.3
  trajectory_histogram_data.csv             — for the appendix figure

NOTE ON COMPUTE:
Counting all simple paths in a graph this size is exponential in the
worst case. We:
  1. Cap path length at 7 (papers); few real trajectories go longer
  2. Use DFS with strict pruning (cycle check, year check, type check)
  3. Estimate sampling-corrected counts for length-6+ via stratified
     random restart sampling

Wall-clock target: <10 minutes on your A100 box.
"""
import argparse
import json
import random
import statistics
import time
from collections import defaultdict, Counter
from pathlib import Path


# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

PROGRESSION_TYPES = {
    "direct_extension",
    "future_realized",
    "causal_extension",
    "limit_addressed",
}

ALL_TYPES = {
    "direct_extension",
    "future_realized",
    "limit_addressed",
    "causal_extension",
    "temporal_semantic",
    "dispute",
}

MAX_LENGTH = 7         # cap at 7 papers (length-7 trajectory = 6 edges)
EXACT_UP_TO = 5        # exact enumeration for length 3, 4, 5
SAMPLE_FOR_LONGER = 6  # length 6, 7 use sampling
SAMPLE_N_STARTS = 5_000  # for sampling-based estimation


# ----------------------------------------------------------------------
# LOAD GRAPH
# ----------------------------------------------------------------------

def load_graph(corpus_dir, definition="strict"):
    """
    Load edges and paper years.
    Returns:
        adj: dict[src_id] -> list of (tgt_id, edge_type)
              filtered to the active type set
        years: dict[paper_id] -> int
    """
    print(f"Loading typed_edges.json...")
    edges = json.load(open(corpus_dir / "data/graph/typed_edges.json"))
    print(f"  {len(edges):,} edges")
    
    print(f"Loading paper metadata for years...")
    candidates = [
        corpus_dir / "data/filtered/corpus.json",
        corpus_dir / "data/standardized/corpus.json",
    ]
    papers_list = None
    for p in candidates:
        if p.exists():
            papers_list = json.load(open(p))
            print(f"  loaded {len(papers_list):,} papers from {p}")
            break
    if papers_list is None:
        raise FileNotFoundError("Couldn't find papers")
    
    years = {int(p["paper_id"]): p.get("year") for p in papers_list
             if p.get("year")}
    print(f"  {len(years):,} papers with valid year")
    
    # Select active type set
    active_types = PROGRESSION_TYPES if definition == "strict" else ALL_TYPES
    print(f"  using {definition} definition: {len(active_types)} edge types")
    
    # Build adjacency with year-ordering and type filter
    adj = defaultdict(list)
    n_edges_kept = 0
    n_edges_year_violation = 0
    n_edges_type_filtered = 0
    
    for e in edges:
        if e["edge_type"] not in active_types:
            n_edges_type_filtered += 1
            continue
        src = int(e["src"])
        tgt = int(e["tgt"])
        if src not in years or tgt not in years:
            continue
        if years[tgt] < years[src]:
            n_edges_year_violation += 1
            continue
        adj[src].append((tgt, e["edge_type"]))
        n_edges_kept += 1
    
    print(f"  edges kept: {n_edges_kept:,}")
    print(f"  edges type-filtered: {n_edges_type_filtered:,}")
    print(f"  edges year-violation: {n_edges_year_violation:,}")
    print(f"  source nodes with outgoing edges: {len(adj):,}")
    
    return adj, years


# ----------------------------------------------------------------------
# PATH ENUMERATION (DFS)
# ----------------------------------------------------------------------

def count_paths_from(adj, start, max_length, count_by_length):
    """
    DFS from start node, counting all simple paths up to max_length.
    Updates count_by_length[length] in place.
    
    A path of "length L" here means L papers, L-1 edges.
    Length-3 = the minimum interesting trajectory.
    """
    # Stack: (current_node, visited_set, current_length, path_types)
    # We count paths of length >= 3 at every internal node visit
    
    visited = {start}
    
    def dfs(node, length):
        # We're at `node`, having visited `length` papers
        # count paths of length L >= 3 ending at this node
        if length >= 3:
            count_by_length[length] += 1
        if length >= max_length:
            return
        for tgt, etype in adj.get(node, []):
            if tgt in visited:
                continue
            visited.add(tgt)
            dfs(tgt, length + 1)
            visited.remove(tgt)
    
    dfs(start, 1)


def count_paths_exact(adj, max_length):
    """
    Exact enumeration of all simple directed paths of length <= max_length.
    Returns dict[length] -> count.
    """
    count_by_length = Counter()
    start_nodes = list(adj.keys())
    
    print(f"  exact enumeration over {len(start_nodes):,} start nodes "
          f"(max_length={max_length})...")
    t0 = time.time()
    
    for i, start in enumerate(start_nodes):
        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(start_nodes) - i) / rate
            print(f"    [{i:,}/{len(start_nodes):,}]  "
                  f"counts so far: {dict(count_by_length)}  "
                  f"eta {eta:.0f}s")
        count_paths_from(adj, start, max_length, count_by_length)
    
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s")
    return dict(count_by_length)


# ----------------------------------------------------------------------
# SAMPLING-BASED ESTIMATION FOR LONGER PATHS
# ----------------------------------------------------------------------

def sample_paths(adj, max_length, n_starts, seed=42):
    """
    Random restart sampling of paths up to max_length.
    Returns:
        count_estimates: dict[length] -> estimated count
        actual_samples: dict[length] -> sample count from the restarts
    """
    random.seed(seed)
    start_nodes = list(adj.keys())
    sampled_starts = random.sample(start_nodes, min(n_starts, len(start_nodes)))
    
    print(f"  sampling from {len(sampled_starts):,} start nodes "
          f"(max_length={max_length})...")
    t0 = time.time()
    
    sample_counts = Counter()
    for start in sampled_starts:
        count_paths_from(adj, start, max_length, sample_counts)
    
    elapsed = time.time() - t0
    print(f"  sampling done in {elapsed:.1f}s")
    
    # Scale up: counts_full ≈ counts_sample × (total / sample) for each length
    scale = len(start_nodes) / len(sampled_starts)
    estimates = {L: int(sample_counts[L] * scale) for L in sample_counts}
    return estimates, dict(sample_counts)


# ----------------------------------------------------------------------
# COVERAGE: % OF PAPERS IN AT LEAST ONE LENGTH-3 TRAJECTORY
# ----------------------------------------------------------------------

def compute_coverage(adj):
    """
    For each paper, check if it appears in any length-3 trajectory.
    A paper appears in a length-3 path if:
      - it's a start with at least 2 outgoing edges leading to length-3, OR
      - it's a middle with at least 1 in-edge and 1 out-edge, OR
      - it's an end with 2 in-degrees back to a valid start
    
    We compute this by: for each node, check if there's any (u, v, w) where
    u → v → w involves this node.
    """
    n_in_trajectory = set()
    
    # For every (u, v) edge, check if v has any outgoing edge (u, v, *)
    # If yes, u, v, and at least one tgt are in a length-3 trajectory.
    for u, edges_out in adj.items():
        for v, _ in edges_out:
            v_out = adj.get(v, [])
            for w, _ in v_out:
                if w != u:  # not a back-edge
                    n_in_trajectory.add(u)
                    n_in_trajectory.add(v)
                    n_in_trajectory.add(w)
                    break  # at least one w; this (u,v) qualifies
    
    return n_in_trajectory


# ----------------------------------------------------------------------
# LONGEST PATH (BOUNDED)
# ----------------------------------------------------------------------

def find_longest_path(adj, max_length=8, n_starts=200, seed=43):
    """
    Find the longest simple directed path bounded by max_length.
    Uses DFS with random restart sampling.
    Returns: (length, path) where path is a list of node IDs.
    """
    random.seed(seed)
    start_nodes = random.sample(list(adj.keys()), min(n_starts, len(adj)))
    
    longest = (0, None)
    
    for start in start_nodes:
        visited = {start}
        path = [start]
        
        def dfs(node, depth):
            nonlocal longest
            if depth > longest[0]:
                longest = (depth, list(path))
            if depth >= max_length:
                return
            # Early termination: if we already found a path at max_length,
            # don't keep searching deeper from other starts
            if longest[0] >= max_length:
                return
            # Cap fanout per node to prevent explosion
            neighbors = adj.get(node, [])[:50]  # only first 50 outgoing
            for tgt, _ in neighbors:
                if tgt in visited:
                    continue
                visited.add(tgt)
                path.append(tgt)
                dfs(tgt, depth + 1)
                path.pop()
                visited.remove(tgt)
        
        dfs(start, 1)
    
    return longest


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", default=".")
    parser.add_argument("--output_path",
                        default="outputs/metrics/trajectory_stats.json")
    parser.add_argument("--definition", choices=["strict", "inclusive"],
                        default="strict",
                        help="strict = progression types only (headline);"
                             " inclusive = all 6 types (sensitivity)")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    corpus_dir = Path(args.corpus_dir).resolve()
    out_path = corpus_dir / args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run both definitions for the appendix sensitivity table
    all_results = {}
    
    for definition in ["strict", "inclusive"]:
        print("\n" + "=" * 70)
        print(f"DEFINITION: {definition}")
        print("=" * 70)
        adj, years = load_graph(corpus_dir, definition=definition)
        
        # ===== Path counts =====
        print(f"\nStep 1: Exact enumeration for lengths 3-{EXACT_UP_TO}...")
        exact_counts = count_paths_exact(adj, max_length=EXACT_UP_TO)
        
        print(f"\nStep 2: Sampling for lengths {SAMPLE_FOR_LONGER}-{args.max_length}...")
        sample_estimates, sample_actuals = sample_paths(
            adj, max_length=args.max_length,
            n_starts=SAMPLE_N_STARTS, seed=args.seed
        )
        
        # Merge: exact for 3-5, sampled for 6-7
        merged = {}
        for L in range(3, args.max_length + 1):
            if L <= EXACT_UP_TO:
                merged[L] = {
                    "count": exact_counts.get(L, 0),
                    "method": "exact",
                }
            else:
                merged[L] = {
                    "count": sample_estimates.get(L, 0),
                    "method": "sampled",
                    "sample_count": sample_actuals.get(L, 0),
                }
        
        # ===== Aggregate stats =====
        all_path_counts = []
        for L, info in merged.items():
            # For mean/median, expand: L appears `count` times
            all_path_counts.extend([L] * min(info["count"], 100_000))  # cap for memory
        
        total_paths_3_5 = sum(merged[L]["count"] for L in [3, 4, 5])
        total_paths_3plus = sum(info["count"] for info in merged.values())
        
        if all_path_counts:
            mean_length = statistics.mean(all_path_counts)
            median_length = statistics.median(all_path_counts)
        else:
            mean_length = median_length = 0
        
        # ===== Coverage =====
        print(f"\nStep 3: Computing coverage...")
        covered = compute_coverage(adj)
        n_total_papers = len(years)
        coverage_pct = 100 * len(covered) / max(n_total_papers, 1)
        print(f"  {len(covered):,} of {n_total_papers:,} papers "
              f"({coverage_pct:.1f}%) appear in some length-3 trajectory")
        
        # ===== Longest path =====
        print(f"\nStep 4: Finding longest path...")
        longest_length, longest_path = find_longest_path(
            adj, max_length=15, n_starts=2000, seed=args.seed
        )
        print(f"  longest: {longest_length} papers")
        
        all_results[definition] = {
            "total_trajectories_3plus": total_paths_3plus,
            "total_trajectories_length_3_to_5": total_paths_3_5,
            "by_length": merged,
            "mean_length_capped": mean_length,
            "median_length_capped": median_length,
            "coverage": {
                "papers_in_trajectory": len(covered),
                "papers_total": n_total_papers,
                "coverage_pct": coverage_pct,
            },
            "longest_path": {
                "length": longest_length,
                "path_node_ids": longest_path[:longest_length] if longest_path else None,
            },
            "edges_used": sum(len(v) for v in adj.values()),
            "source_nodes": len(adj),
        }
    
    # ===== Save =====
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Wrote {out_path}")
    
    # Histogram CSV for appendix figure
    hist_path = corpus_dir / "outputs/metrics/trajectory_histogram_data.csv"
    with open(hist_path, "w") as f:
        f.write("length,strict_count,inclusive_count\n")
        for L in range(3, args.max_length + 1):
            s = all_results["strict"]["by_length"].get(L, {}).get("count", 0)
            i = all_results["inclusive"]["by_length"].get(L, {}).get("count", 0)
            f.write(f"{L},{s},{i}\n")
    print(f"[+] Wrote {hist_path}")
    
    # ===== Summary =====
    print("\n" + "=" * 70)
    print("HEADLINE FOR §3.3 (using STRICT definition):")
    print("=" * 70)
    strict = all_results["strict"]
    print(f"  Active progression edges:     {strict['edges_used']:,}")
    print(f"  Source nodes:                 {strict['source_nodes']:,}")
    print(f"  Trajectories length 3-5 (exact): "
          f"{strict['total_trajectories_length_3_to_5']:,}")
    print(f"  Trajectories length 3+:       {strict['total_trajectories_3plus']:,}")
    print(f"  Mean length:                  {strict['mean_length_capped']:.2f}")
    print(f"  Median length:                {strict['median_length_capped']}")
    print(f"  Longest path found:           {strict['longest_path']['length']} papers")
    print(f"  Coverage:                     {strict['coverage']['papers_in_trajectory']:,} of "
          f"{strict['coverage']['papers_total']:,} papers "
          f"({strict['coverage']['coverage_pct']:.1f}%)")
    print()
    print(f"  Length 3: {strict['by_length'][3]['count']:,}")
    print(f"  Length 4: {strict['by_length'][4]['count']:,}")
    print(f"  Length 5: {strict['by_length'][5]['count']:,}")
    for L in range(6, args.max_length + 1):
        info = strict['by_length'][L]
        print(f"  Length {L}: ~{info['count']:,} (sampled, n={info.get('sample_count', 0):,})")


if __name__ == "__main__":
    main()
