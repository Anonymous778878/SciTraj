"""
Phase 5.2 — Retrieve candidate successor papers per source paper.

For each source paper A, we query the FAISS abstract index for the top-K most
similar papers globally. We then apply hard filters:

  1. Temporal:   year(target) > year(source)  (strict forward direction)
  2. Similarity: abstract_sim >= min_abstract_similarity
  3. Self:       target != source

The output is the candidate pool. Phase 6 (graph construction) will then
classify each (source, target) candidate pair into an edge type by looking at
multiple signal views (future, limit, causal, etc.).

We retrieve top-K via the abstract view because abstract similarity is the
most reliable general-purpose signal for scientific similarity. Phase 6 uses
the other views to refine and classify.

Output:
  data/retrieval/candidates.json
      { "source_paper_id": int,
        "candidates": [
          { "target_paper_id": int, "abstract_sim": float, "time_delta": int },
          ...
        ] }
"""
import time
from pathlib import Path

import faiss
import numpy as np

from utils import ensure_dir, get_logger, load_config, load_json, save_json

log = get_logger("phase5.2")


def main():
    cfg = load_config()
    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    ret_dir = Path(cfg["paths"]["retrieval_dir"])
    val_dir = Path(cfg["paths"]["validated_dir"])

    top_k = cfg["retrieval"]["top_k"]
    min_sim = cfg["retrieval"]["min_abstract_similarity"]
    max_gap = cfg["retrieval"]["max_temporal_gap_years"]

    # Load validated signals to get year per paper
    log.info("loading papers and embeddings")
    papers = load_json(val_dir / "signals_with_reliability.json")
    papers.sort(key=lambda p: p["paper_id"])
    paper_ids = [p["paper_id"] for p in papers]
    years = np.array([p["year"] for p in papers], dtype=np.int32)
    n = len(papers)

    # Sanity-check row order matches the paper_id_order.json from Phase 4
    saved_order = load_json(emb_dir / "paper_id_order.json")
    if saved_order != paper_ids:
        raise RuntimeError(
            "paper_id order mismatch between Phase 3 and Phase 4. "
            "Rerun Phase 4 before Phase 5."
        )

    # Load abstract embeddings and FAISS index
    abstract_embs = np.load(emb_dir / "abstract_embs_norm.npy").astype(np.float32)
    index = faiss.read_index(str(ret_dir / "abstract.index"))
    log.info("loaded abstract index with %d vectors", index.ntotal)

    # Query the index in batches of 512 papers for speed
    log.info("retrieving top-%d candidates per paper (min_sim=%.2f)", top_k, min_sim)
    t0 = time.time()
    batch_size = 512
    all_sims    = np.empty((n, top_k + 1), dtype=np.float32)
    all_indices = np.empty((n, top_k + 1), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sims, idxs = index.search(abstract_embs[start:end], top_k + 1)
        all_sims[start:end]    = sims
        all_indices[start:end] = idxs

    log.info("FAISS search done in %.1fs", time.time() - t0)

    # Filter candidates per source paper
    log.info("filtering candidates by temporal + similarity constraints")
    candidates_per_paper = []
    total_candidates = 0
    empty_count = 0

    for i in range(n):
        src_year = years[i]
        kept = []

        for j in range(top_k + 1):
            tgt_idx = int(all_indices[i, j])
            sim     = float(all_sims[i, j])

            # Skip self
            if tgt_idx == i:
                continue

            tgt_year = int(years[tgt_idx])
            time_delta = tgt_year - int(src_year)

            # Strict forward temporal
            if time_delta <= 0:
                continue
            # Too far in the future — down-weighted (skipped) by config gap
            if time_delta > max_gap:
                continue
            # Similarity floor
            if sim < min_sim:
                continue

            kept.append({
                "target_paper_id": int(paper_ids[tgt_idx]),
                "abstract_sim":    round(sim, 4),
                "time_delta":      time_delta,
            })

        candidates_per_paper.append({
            "source_paper_id": int(paper_ids[i]),
            "source_year":     int(src_year),
            "n_candidates":    len(kept),
            "candidates":      kept,
        })
        total_candidates += len(kept)
        if not kept:
            empty_count += 1

    save_json(candidates_per_paper, ret_dir / "candidates.json")

    # Stats
    n_nonempty = n - empty_count
    avg_candidates = total_candidates / n_nonempty if n_nonempty else 0
    log.info("=" * 55)
    log.info("PHASE 5 COMPLETE")
    log.info("=" * 55)
    log.info("source papers:           %d", n)
    log.info("papers with candidates:  %d (%.1f%%)", n_nonempty, 100 * n_nonempty / n)
    log.info("papers with no candidates: %d", empty_count)
    log.info("total candidates:        %d", total_candidates)
    log.info("avg candidates per paper: %.1f", avg_candidates)

    # Write summary report
    save_json({
        "n_papers":              n,
        "n_with_candidates":     n_nonempty,
        "n_empty":               empty_count,
        "total_candidates":      total_candidates,
        "avg_candidates":        round(avg_candidates, 2),
        "top_k":                 top_k,
        "min_abstract_sim":      min_sim,
        "max_temporal_gap":      max_gap,
    }, ret_dir / "retrieval_report.json")


if __name__ == "__main__":
    main()
