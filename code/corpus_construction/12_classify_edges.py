"""
Phase 6.2 — Edge type classification (FIXED priority).

Bug in the previous version: direct_extension was checked first with threshold
0.85, capturing almost any related pair (SPECTER2 readily exceeds 0.85 for
related papers). 98% of edges collapsed to direct_extension, destroying
the typed-graph contribution.

Correct priority (most specific first):
  1. future_realized   — A's stated future direction matches B's contribution
  2. limit_addressed   — A's stated limitation matches B's contribution
  3. causal_extension  — A's causal claim matches B's causal claim
  4. direct_extension  — Very high signal similarity AND no specific match
                         (raised threshold to 0.92)
  5. temporal_semantic — Strong abstract similarity (generic)
  6. related_work      — Weak similarity (floor)

Specific types only fire when the source paper actually HAS the relevant
signal text. This prevents fallback-embedding pairs (where ~88% of papers
have no limit/future text and the embedding is just the abstract embedding)
from spuriously matching.

Dispute edges are detected separately via lexical overlap and override the
type to 'dispute' with negative confidence (signed-graph signal).
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from utils import ensure_dir, get_logger, load_config, load_json, save_json

log = get_logger("phase6.2")


# Stable edge-type ID mapping. Don't reorder — Phase 9 GNN depends on this.
EDGE_TYPES = {
    "direct_extension":     0,
    "future_realized":      1,
    "limit_addressed":      2,
    "causal_extension":     3,
    "performance_successor": 4,
    "method_reuse":         5,
    "temporal_semantic":    6,
    "related_work":         7,
    "dispute":              8,
}


def topic_similarity(soft_a: list, soft_b: list) -> float:
    """Dot product of two sparse topic distributions."""
    weights_b = {t: w for t, w in soft_b}
    return sum(w_a * weights_b.get(t, 0.0) for t, w_a in soft_a)


def has_real_signal(signal_text: str, min_chars: int = 30) -> bool:
    """A signal is 'real' if its text is substantive (not just whitespace)."""
    return bool(signal_text and len(signal_text.strip()) >= min_chars)


def classify_edge(
    src: dict, tgt: dict,
    abstract_sim: float,
    embeddings: dict,
    src_idx: int, tgt_idx: int,
    src_signals: dict, tgt_signals: dict,
    cfg: dict,
) -> tuple[str, float, dict]:
    """
    Classify a candidate (source, target) pair into one edge type.

    Priority order (most specific to least specific):
      future_realized -> limit_addressed -> causal_extension ->
      direct_extension -> temporal_semantic -> related_work
    """
    th = cfg["graph"]
    extras = {}

    tgt_has_contrib = has_real_signal(tgt_signals.get("contrib_text"))

    # ── 1. Future-Realized ──
    if has_real_signal(src_signals.get("future_text")) and tgt_has_contrib:
        future_sim = float(np.dot(
            embeddings["future"][src_idx],
            embeddings["contrib"][tgt_idx],
        ))
        extras["future_sim"] = round(future_sim, 4)
        if future_sim >= th["thresh_future_realized"]:
            return "future_realized", future_sim, extras

    # ── 2. Limitation-Addressed ──
    if has_real_signal(src_signals.get("limit_text")) and tgt_has_contrib:
        limit_sim = float(np.dot(
            embeddings["limit"][src_idx],
            embeddings["contrib"][tgt_idx],
        ))
        extras["limit_sim"] = round(limit_sim, 4)
        if limit_sim >= th["thresh_limit_addressed"]:
            return "limit_addressed", limit_sim, extras

    # ── 3. Causal-Extension ──
    if has_real_signal(src_signals.get("causal_text")) and has_real_signal(tgt_signals.get("causal_text")):
        causal_sim = float(np.dot(
            embeddings["causal"][src_idx],
            embeddings["causal"][tgt_idx],
        ))
        extras["causal_sim"] = round(causal_sim, 4)
        if causal_sim >= th["thresh_causal_extension"]:
            return "causal_extension", causal_sim, extras

    # ── 4. Direct-Extension (high threshold; only true continuations) ──
    sig_sim = float(np.dot(embeddings["signal"][src_idx], embeddings["signal"][tgt_idx]))
    extras["sig_sim"] = round(sig_sim, 4)
    if sig_sim >= th["thresh_direct_extension"]:
        return "direct_extension", sig_sim, extras

    # ── 5. Temporal-Semantic (generic strong abstract similarity) ──
    if abstract_sim >= th["thresh_temporal_semantic"]:
        return "temporal_semantic", abstract_sim, extras

    # ── 6. Related-Work (weak similarity floor) ──
    if abstract_sim >= th["thresh_related_work"]:
        return "related_work", abstract_sim, extras

    return None, 0.0, extras


def detect_dispute(src_signals: dict, tgt_signals: dict, tgt_title: str) -> bool:
    """Heuristic: source's dispute_text mentions content words from target's title."""
    dispute_text = (src_signals.get("dispute_text") or "").lower()
    if not dispute_text or len(dispute_text) < 30:
        return False

    title_words = set()
    for w in tgt_title.lower().split():
        w = "".join(c for c in w if c.isalpha())
        if len(w) > 4:
            title_words.add(w)

    if len(title_words) < 2:
        return False

    matches = sum(1 for w in title_words if w in dispute_text)
    return matches >= 2


def author_overlap(src: dict, tgt: dict) -> int:
    """Count overlapping authors. Returns 0 if metadata missing."""
    src_authors = src.get("authors") or []
    tgt_authors = tgt.get("authors") or []
    if not src_authors or not tgt_authors:
        return 0
    src_names = {a.get("name", "").strip().lower() for a in src_authors if isinstance(a, dict)}
    tgt_names = {a.get("name", "").strip().lower() for a in tgt_authors if isinstance(a, dict)}
    src_names.discard("")
    tgt_names.discard("")
    return len(src_names & tgt_names)


def main():
    cfg = load_config()
    emb_dir   = Path(cfg["paths"]["embeddings_dir"])
    ret_dir   = Path(cfg["paths"]["retrieval_dir"])
    graph_dir = Path(cfg["paths"]["graph_dir"])

    log.info("loading data")

    signals_list = load_json(Path(cfg["paths"]["validated_dir"]) / "signals_with_reliability.json")
    signals_list.sort(key=lambda p: p["paper_id"])
    signals_by_id = {p["paper_id"]: p for p in signals_list}

    corpus = load_json(Path(cfg["paths"]["filtered_dir"]) / "corpus.json")
    paper_by_id = {p["paper_id"]: p for p in corpus}

    topic_assignments = load_json(graph_dir / "topic_assignments.json")
    topic_by_id = {ta["paper_id"]: ta for ta in topic_assignments}

    log.info("loading 5 view embeddings (signal, future, limit, contrib, causal)")
    embeddings = {}
    for view in ["signal", "future", "limit", "contrib", "causal"]:
        embeddings[view] = np.load(emb_dir / f"{view}_embs_norm.npy").astype(np.float32)

    paper_id_order = load_json(emb_dir / "paper_id_order.json")
    pid_to_row = {pid: i for i, pid in enumerate(paper_id_order)}

    candidates_per_paper = load_json(ret_dir / "candidates.json")
    log.info("classifying edges from %d source papers", len(candidates_per_paper))

    typed_edges = []
    type_counts = Counter()
    cross_venue_count = 0
    dispute_count = 0
    author_cont_count = 0
    n_processed = 0
    min_topic_sim = cfg["graph"]["topic_similarity_threshold"]

    for src_record in candidates_per_paper:
        src_id = src_record["source_paper_id"]
        if not src_record["candidates"]:
            continue
        n_processed += 1
        if n_processed % 2000 == 0:
            log.info("  processed %d/%d sources, %d edges so far",
                     n_processed, len(candidates_per_paper), len(typed_edges))

        src_signals = signals_by_id[src_id]
        src_paper   = paper_by_id[src_id]
        src_topic   = topic_by_id[src_id]
        src_idx     = pid_to_row[src_id]

        for cand in src_record["candidates"]:
            tgt_id = cand["target_paper_id"]
            tgt_signals = signals_by_id[tgt_id]
            tgt_paper   = paper_by_id[tgt_id]
            tgt_topic   = topic_by_id[tgt_id]
            tgt_idx     = pid_to_row[tgt_id]

            t_sim = topic_similarity(src_topic["soft_topics"], tgt_topic["soft_topics"])
            if t_sim < min_topic_sim and cand["abstract_sim"] < 0.85:
                continue

            etype, conf, extras = classify_edge(
                src_paper, tgt_paper,
                abstract_sim=cand["abstract_sim"],
                embeddings=embeddings,
                src_idx=src_idx, tgt_idx=tgt_idx,
                src_signals=src_signals, tgt_signals=tgt_signals,
                cfg=cfg,
            )
            if etype is None:
                continue

            cross_venue = (src_paper["venue"] != tgt_paper["venue"])
            if cross_venue:
                cross_venue_count += 1

            n_shared = author_overlap(src_paper, tgt_paper)
            author_continuation = n_shared >= cfg["graph"]["author_min_overlap"]
            if author_continuation:
                author_cont_count += 1

            if detect_dispute(src_signals, tgt_signals, tgt_paper["title"]):
                etype = "dispute"
                conf = -abs(conf)
                dispute_count += 1

            type_counts[etype] += 1

            typed_edges.append({
                "src": src_id,
                "tgt": tgt_id,
                "edge_type":          etype,
                "edge_type_id":       EDGE_TYPES[etype],
                "confidence":         round(conf, 4),
                "abstract_sim":       cand["abstract_sim"],
                "topic_sim":          round(t_sim, 4),
                "time_delta":         cand["time_delta"],
                "src_venue":          src_paper["venue"],
                "tgt_venue":          tgt_paper["venue"],
                "cross_venue":        cross_venue,
                "shared_authors":     n_shared,
                "author_continuation": author_continuation,
                "extras":             extras,
            })

    save_json(typed_edges, graph_dir / "typed_edges.json")

    by_venue_pair = Counter()
    for e in typed_edges:
        by_venue_pair[(e["src_venue"], e["tgt_venue"])] += 1

    log.info("=" * 55)
    log.info("PHASE 6.2 COMPLETE")
    log.info("=" * 55)
    log.info("total typed edges:    %d", len(typed_edges))
    log.info("cross-venue edges:    %d (%.1f%%)",
             cross_venue_count, 100*cross_venue_count/max(len(typed_edges),1))
    log.info("dispute edges:        %d", dispute_count)
    log.info("author-continuation:  %d", author_cont_count)
    log.info("")
    log.info("by edge type:")
    for etype, n in type_counts.most_common():
        log.info("  %-22s  %7d  (%.1f%%)",
                 etype, n, 100*n/max(len(typed_edges),1))
    log.info("")
    log.info("top venue-pair combinations:")
    for (sv, tv), c in by_venue_pair.most_common(10):
        log.info("  %s -> %s : %d", sv, tv, c)

    stats = {
        "total_edges":       len(typed_edges),
        "cross_venue_edges": cross_venue_count,
        "dispute_edges":     dispute_count,
        "author_cont_edges": author_cont_count,
        "by_type":           dict(type_counts),
        "by_venue_pair":     {f"{sv}->{tv}": c for (sv, tv), c in by_venue_pair.items()},
    }
    save_json(stats, graph_dir / "edge_stats.json")


if __name__ == "__main__":
    main()
