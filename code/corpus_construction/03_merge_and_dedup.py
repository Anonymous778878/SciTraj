"""
Phase 1.3  — Hard-filter CVPR and NeurIPS (no quality ranking; pre-curated venues).
Phase 1.4  — Cross-venue deduplication by normalized title hash.
Phase 1.5  — Assign global paper IDs and produce unified corpus.

Output: data/filtered/corpus.json  (the single authoritative corpus file)
"""
import hashlib
import re
from collections import Counter
from pathlib import Path

from utils import (
    count_alpha,
    count_words,
    ensure_dir,
    get_logger,
    load_config,
    load_json,
    save_json,
)

log = get_logger("phase1.3-5")

STOPWORDS = {"a", "an", "the", "of", "for", "to", "in", "on", "and", "or",
             "with", "by", "from", "is", "are", "at"}

VENUE_PRIORITY = {"CVPR": 2, "NeurIPS": 1, "ACL": 0}


def hard_filter(paper: dict, filter_cfg: dict) -> tuple[bool, str | None]:
    """Same hard exclusions as ACL filter, minus the ranking part."""
    year = paper.get("year")
    abstract = paper.get("abstract", "") or ""

    if not year or year < filter_cfg["year_min"] or year > filter_cfg["year_max"]:
        return False, f"year_out_of_range({year})"
    if len(abstract) < filter_cfg["min_abstract_chars"]:
        return False, f"abstract_too_short({len(abstract)})"

    full_text = " ".join([abstract] + list((paper.get("sections") or {}).values())).lower()
    if count_alpha(full_text) < filter_cfg["min_alpha"] or count_words(full_text) < filter_cfg["min_words"]:
        return False, "insufficient_text"

    return True, None


def normalized_title_hash(title: str) -> str:
    """Hash the normalized title for cross-venue duplicate detection."""
    t = (title or "").lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = " ".join(w for w in t.split() if w not in STOPWORDS)
    return hashlib.md5(t[:60].encode()).hexdigest()


def hard_filter_venue(papers: list, venue: str, filter_cfg: dict) -> tuple[list, list]:
    """Apply hard filters to a venue. Returns (kept, disqualified)."""
    kept, disq = [], []
    for p in papers:
        ok, reason = hard_filter(p, filter_cfg)
        if ok:
            # CVPR and NeurIPS don't get quality scores; assign neutral baseline
            kept.append({**p, "quality_score": 50.0, "score_components": {}})
        else:
            disq.append({"source_id": p["source_id"], "reason": reason})

    log.info("[%s] kept %d/%d (disqualified %d)", venue, len(kept), len(papers), len(disq))
    reasons = Counter(d["reason"].split("(")[0] for d in disq)
    for r, c in reasons.most_common():
        log.info("  %s: %d", r, c)
    return kept, disq


def deduplicate(merged: list) -> tuple[list, list]:
    """Cross-venue deduplication by title hash. Keeps higher-quality duplicate."""
    by_hash = {}
    removed = []

    for p in merged:
        h = normalized_title_hash(p.get("title", ""))
        if h in by_hash:
            existing = by_hash[h]
            # Tie-break: higher quality score, then venue priority
            p_key = (p["quality_score"], VENUE_PRIORITY.get(p["venue"], 0))
            e_key = (existing["quality_score"], VENUE_PRIORITY.get(existing["venue"], 0))

            if p_key > e_key:
                removed.append({
                    "dropped_source_id": existing["source_id"],
                    "dropped_venue":     existing["venue"],
                    "kept_source_id":    p["source_id"],
                    "kept_venue":        p["venue"],
                })
                by_hash[h] = p
            else:
                removed.append({
                    "dropped_source_id": p["source_id"],
                    "dropped_venue":     p["venue"],
                    "kept_source_id":    existing["source_id"],
                    "kept_venue":        existing["venue"],
                })
        else:
            by_hash[h] = p

    return list(by_hash.values()), removed


def assign_paper_ids(corpus: list) -> None:
    """Assign deterministic global IDs in place, sorted by (year, venue, source_id)."""
    corpus.sort(key=lambda p: (p.get("year", 0), p["venue"], p["source_id"] or ""))
    for i, p in enumerate(corpus):
        p["paper_id"] = i


def main():
    cfg = load_config()
    filter_cfg = {
        **cfg["filtering"],
        "year_min": cfg["corpus"]["year_min"],
        "year_max": cfg["corpus"]["year_max"],
    }
    std_dir  = Path(cfg["paths"]["standardized_dir"])
    filt_dir = ensure_dir(cfg["paths"]["filtered_dir"])

    # ── Load ACL (already filtered) ──
    acl = load_json(filt_dir / "acl_kept.json")

    # ── Hard-filter CVPR ──
    cvpr_raw = load_json(std_dir / "cvpr.json")
    cvpr_kept, cvpr_disq = hard_filter_venue(cvpr_raw, "CVPR", filter_cfg)
    save_json(cvpr_disq, filt_dir / "cvpr_disqualified.json")

    # ── Hard-filter NeurIPS ──
    nips_raw = load_json(std_dir / "neurips.json")
    nips_kept, nips_disq = hard_filter_venue(nips_raw, "NeurIPS", filter_cfg)
    save_json(nips_disq, filt_dir / "neurips_disqualified.json")

    # ── Merge ──
    merged = acl + cvpr_kept + nips_kept
    log.info("merged: %d papers before dedup", len(merged))

    # ── Deduplicate ──
    deduped, dups = deduplicate(merged)
    log.info("deduped: %d papers (%d duplicates removed)", len(deduped), len(dups))
    save_json(dups, filt_dir / "duplicates_removed.json")

    # ── Assign IDs ──
    assign_paper_ids(deduped)

    # ── Persist ──
    save_json(deduped, filt_dir / "corpus.json")
    log.info("wrote corpus.json with %d papers", len(deduped))

    # Summary
    by_venue = Counter(p["venue"] for p in deduped)
    log.info("final corpus by venue:")
    for v, c in by_venue.most_common():
        log.info("  %s: %d", v, c)


if __name__ == "__main__":
    main()
