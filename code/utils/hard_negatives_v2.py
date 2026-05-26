"""
Hard negative sampling — v2 utilities.

Adds a STRONGER hard-negative source: the Phase 5 retrieval candidates that
are NOT in the final graph. These are papers SPECTER2 found similar to the
source, but Phase 6 either filtered out (below similarity threshold) or
MMR-pruned (graph capped at 20 out-edges per source).

Why these are stronger negatives:
  Topic-year hard negatives still rely on KMeans-derived topics. KMeans
  topics correlate with SPECTER2 similarity by construction, so a "same
  topic" negative still tends to have low SPECTER2 similarity to the source.

  Candidate-pool negatives, on the other hand, were RETRIEVED by SPECTER2
  as among the source's top-50 most similar papers. Their similarity to
  source is in the same range as the positive target's. The model cannot
  use raw SPECTER2 embedding distance to separate them — it must use graph
  structure or signal alignment.
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch


def build_candidate_pool_negatives(
    candidates_records: list,
    edge_set: Set[Tuple[int, int]],
    pid_to_row: Dict[int, int],
) -> Dict[int, List[int]]:
    """
    For each source paper_id, return the list of candidate target rows that
    are NOT in the final graph (i.e., were retrieved by Phase 5 but excluded
    by Phase 6 thresholds or MMR).

    Returns: src_row -> [tgt_row, ...]
    """
    src_to_negatives = {}
    for rec in candidates_records:
        src_pid = rec["source_paper_id"]
        if src_pid not in pid_to_row:
            continue
        src_row = pid_to_row[src_pid]
        negs_for_src = []
        for cand in rec.get("candidates", []):
            tgt_pid = cand["target_paper_id"]
            if tgt_pid not in pid_to_row:
                continue
            tgt_row = pid_to_row[tgt_pid]
            if (src_row, tgt_row) in edge_set:
                continue                       # this candidate IS a real edge
            negs_for_src.append(tgt_row)
        if negs_for_src:
            src_to_negatives[src_row] = negs_for_src
    return src_to_negatives


def sample_candidate_pool_negatives(
    pos_edges: torch.Tensor,
    src_to_negatives: Dict[int, List[int]],
    rng: np.random.Generator,
    n_per_pos: int = 1,
    fallback_topic_pools: Dict[Tuple[int, int], List[int]] = None,
    topic_arr: np.ndarray = None,
    year_arr: np.ndarray = None,
) -> torch.Tensor:
    """
    For each (src, tgt) positive, sample n_per_pos negatives from the
    candidate pool of src. Falls back to topic-year pool if a source has no
    excluded candidates (rare, mostly latest-year sources).

    Returns: LongTensor (2, E_pos * n_per_pos)
    """
    src_arr = pos_edges[0].cpu().numpy()
    tgt_arr = pos_edges[1].cpu().numpy()
    n_pos = len(src_arr)
    out_size = n_pos * n_per_pos

    neg_src = np.empty(out_size, dtype=np.int64)
    neg_tgt = np.empty(out_size, dtype=np.int64)

    write = 0
    for i in range(n_pos):
        s = int(src_arr[i])
        t = int(tgt_arr[i])

        candidates = src_to_negatives.get(s, [])

        for _ in range(n_per_pos):
            chosen = -1
            if candidates:
                # Try up to 5 random picks from candidate pool, avoiding the actual target
                for _ in range(5):
                    pick = int(candidates[rng.integers(0, len(candidates))])
                    if pick != t:
                        chosen = pick
                        break

            if chosen < 0 and fallback_topic_pools is not None and topic_arr is not None:
                # Topic-year fallback
                t_topic = int(topic_arr[t])
                t_year  = int(year_arr[t])
                pool = fallback_topic_pools.get((t_topic, t_year), [])
                for _ in range(5):
                    if pool:
                        pick = int(pool[rng.integers(0, len(pool))])
                        if pick != t and (year_arr is None or year_arr[pick] > year_arr[s]):
                            chosen = pick
                            break

            if chosen < 0:
                # Last resort: random temporally valid pair
                for _ in range(5):
                    pick = int(rng.integers(0, len(year_arr) if year_arr is not None else 10000))
                    if pick != t and pick != s:
                        chosen = pick
                        break

            neg_src[write] = s
            neg_tgt[write] = chosen if chosen >= 0 else s
            write += 1

    return torch.from_numpy(np.stack([neg_src[:write], neg_tgt[:write]], axis=0))
