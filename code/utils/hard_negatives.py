"""
Hard negative sampling utilities (Phase 9 v2).

Problem with random negatives:
  Random (src, tgt) pairs almost always have low cosine similarity in
  SPECTER2 space (~0.3-0.5), while real edges have similarity > 0.55 by
  the Phase 5 retrieval threshold. So distinguishing "is this edge real?"
  is trivial — just compute cosine similarity. Tier 1 hits AUC=0.99
  without learning anything.

Hard negative sampling:
  Given a positive edge (src, tgt), sample a negative (src, tgt') where
  tgt' is:
    - in the same topic cluster as tgt
    - within a similar publication-year range
    - NOT actually connected to src in the graph
  These negatives have similar abstract similarity to src as tgt does,
  forcing the model to learn distinctions beyond raw embedding distance.

Functions:
  build_topic_year_pools()   : precompute candidate pools per (topic, year)
  sample_hard_negatives()    : draw hard negatives for a batch of positive edges
"""
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import torch


def build_topic_year_pools(
    paper_ids: torch.Tensor,
    topic_ids: torch.Tensor,
    years: torch.Tensor,
    year_window: int = 2,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Bucket papers by (topic_id, year_bucket) for fast hard-negative lookup.
    A paper is in pool key (T, Y) if its topic == T and its year is within
    [Y - year_window, Y + year_window].

    Returns: dict mapping (topic_id, target_year) -> list of paper row indices.
    Note: the row indices are positions in the embedding tensor (0..N-1),
    NOT paper_ids. Caller should map paper_ids to rows via pid_to_row.
    """
    n = len(paper_ids)
    pools = defaultdict(list)

    paper_arr = paper_ids.cpu().numpy() if isinstance(paper_ids, torch.Tensor) else np.asarray(paper_ids)
    topic_arr = topic_ids.cpu().numpy() if isinstance(topic_ids, torch.Tensor) else np.asarray(topic_ids)
    year_arr  = years.cpu().numpy()     if isinstance(years, torch.Tensor)     else np.asarray(years)

    # For each (topic, target_year), gather papers in that topic within ±window
    for row in range(n):
        t = int(topic_arr[row])
        y = int(year_arr[row])
        # This paper is a candidate negative for queries asking
        # "papers in topic t with target year close to y"
        for query_year in range(y - year_window, y + year_window + 1):
            pools[(t, query_year)].append(row)

    return dict(pools)


def sample_hard_negatives(
    pos_edges: torch.Tensor,           # (2, E_pos)  — row indices
    pos_set: Set[Tuple[int, int]],     # set of all positive (src_row, tgt_row) pairs
    topic_arr: np.ndarray,             # (N,) topic per row
    year_arr: np.ndarray,              # (N,) year per row
    pools: Dict[Tuple[int, int], List[int]],
    rng: np.random.Generator,
    n_per_pos: int = 1,
    fallback_pool: List[int] = None,
) -> torch.Tensor:
    """
    For each positive edge (src, tgt), sample n_per_pos hard negatives:
    paper rows that share tgt's topic and tgt's year-bucket, excluding tgt
    itself and any other positives connected to src.

    Returns: LongTensor (2, E_pos * n_per_pos)
    """
    src_arr = pos_edges[0].cpu().numpy()
    tgt_arr = pos_edges[1].cpu().numpy()
    n_pos = len(src_arr)

    neg_src = np.empty(n_pos * n_per_pos, dtype=np.int64)
    neg_tgt = np.empty(n_pos * n_per_pos, dtype=np.int64)

    write = 0
    for i in range(n_pos):
        s = int(src_arr[i])
        t = int(tgt_arr[i])
        t_topic = int(topic_arr[t])
        t_year  = int(year_arr[t])

        pool = pools.get((t_topic, t_year), fallback_pool or [])

        for _ in range(n_per_pos):
            attempts = 0
            while attempts < 8:
                if pool:
                    cand = int(pool[rng.integers(0, len(pool))])
                else:
                    cand = int(rng.integers(0, len(year_arr)))
                # Constraints:
                #  - candidate target must be year-after source (forward-only)
                #  - candidate must not be the actual target
                #  - candidate must not already be a positive neighbor of src
                if cand == t:
                    attempts += 1; continue
                if int(year_arr[cand]) <= int(year_arr[s]):
                    attempts += 1; continue
                if (s, cand) in pos_set:
                    attempts += 1; continue
                neg_src[write] = s
                neg_tgt[write] = cand
                write += 1
                break
            else:
                # Fallback: random temporally-valid pair
                neg_src[write] = s
                # try a few times to find any valid target year
                for _ in range(5):
                    cand = int(rng.integers(0, len(year_arr)))
                    if int(year_arr[cand]) > int(year_arr[s]) and cand != s:
                        break
                neg_tgt[write] = cand
                write += 1

    return torch.from_numpy(np.stack([neg_src[:write], neg_tgt[:write]], axis=0))


def build_positive_set(edge_index: torch.Tensor) -> Set[Tuple[int, int]]:
    """Build a set of (src_row, tgt_row) positive pairs for fast lookup."""
    src = edge_index[0].cpu().numpy()
    tgt = edge_index[1].cpu().numpy()
    return set(zip(src.tolist(), tgt.tolist()))
