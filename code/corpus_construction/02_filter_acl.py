"""
Phase 1.2 — Quality-score ACL and filter to target.

Scoring dimensions (max 100):
  - Year recency (0-15): linear 2010..2024
  - Section richness (0-20): based on total section chars, capped at 2000
  - Limitation presence (0-20): limitation language matches
  - Future direction presence (0-20): future language matches
  - Contribution explicitness (0-15): contribution language matches
  - Abstract quality (0-10): abstract length proxy

Hard exclusions:
  - Year outside [2010, 2024]
  - Abstract < 100 chars
  - Under 10 words or 50 alpha chars
  - No contribution language anywhere in full text (papers without a claim
    cannot generate meaningful trajectory edges)

Keep ratio: 15000 / 57028 ≈ 0.263 — applied uniformly regardless of sample size.
"""
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

log = get_logger("phase1.2")


# Primary extraction patterns used for scoring (superset of Phase 2 patterns
# because during filtering we only need presence/absence, not extraction).
LIMITATION_PATS = [
    r"\blimitation(s)?\b", r"\bweakness(es)?\b", r"\bdrawback(s)?\b",
    r"\bcannot\b", r"\bfails?\s+(to|when|at|on)\b", r"\blimited\s+to\b",
    r"\bdoes\s+not\s+address\b", r"\bstruggle(s)?\s+with\b",
    r"\bshortcoming(s)?\b",
]
FUTURE_PATS = [
    r"\bfuture\s+work\b", r"\bfuture\s+direction(s)?\b",
    r"\bfuture\s+research\b", r"\bopen\s+question(s)?\b",
    r"\bremains?\s+an?\s+open\b", r"\bwe\s+plan\s+to\b",
    r"\bcould\s+be\s+extended\b", r"\binteresting\s+direction\b",
    r"\bnext\s+steps?\b",
]
CONTRIBUTION_PATS = [
    r"\bwe\s+propose\b", r"\bwe\s+introduce\b", r"\bwe\s+present\b",
    r"\bwe\s+develop\b", r"\bwe\s+demonstrate\b", r"\bthis\s+paper\b",
    r"\bour\s+contribution(s)?\b", r"\bwe\s+show\s+that\b",
    r"\bnovel\b", r"\bnovelty\b",
]


def build_full_text(paper: dict) -> str:
    """Concatenate abstract + all sections into a single lowercase string."""
    parts = [paper.get("abstract", "") or ""]
    for _, v in (paper.get("sections") or {}).items():
        if v:
            parts.append(v)
    return " ".join(parts).lower()


def count_pattern_hits(text: str, patterns: list) -> int:
    """Count how many distinct patterns fire in the text."""
    return sum(1 for pat in patterns if re.search(pat, text))


def score_paper(paper: dict, weights: dict, filter_cfg: dict):
    """
    Returns (total_score, component_dict, disqual_reason_or_None).
    If disqual_reason is not None, the paper is hard-excluded.
    """
    year = paper.get("year")
    abstract = paper.get("abstract", "") or ""

    # ── Hard exclusions ──
    if not year or year < filter_cfg["year_min"] or year > filter_cfg["year_max"]:
        return 0, {}, f"year_out_of_range({year})"
    if len(abstract) < filter_cfg["min_abstract_chars"]:
        return 0, {}, f"abstract_too_short({len(abstract)})"

    full_text = build_full_text(paper)
    if count_alpha(full_text) < filter_cfg["min_alpha"] or count_words(full_text) < filter_cfg["min_words"]:
        return 0, {}, "insufficient_text"

    # Require at least one contribution signal — papers without claims are useless for trajectory edges
    contrib_hits = count_pattern_hits(full_text, CONTRIBUTION_PATS)
    if contrib_hits == 0:
        return 0, {}, "no_contribution_language"

    # ── Scoring components ──
    total_section_chars = sum(len(v) for v in (paper.get("sections") or {}).values())
    year_min, year_max = filter_cfg["year_min"], filter_cfg["year_max"]

    lim_hits = count_pattern_hits(full_text, LIMITATION_PATS)
    fut_hits = count_pattern_hits(full_text, FUTURE_PATS)

    components = {
        "year":             weights["weight_year"] * (year - year_min) / (year_max - year_min),
        "section_richness": min(total_section_chars / 2000, 1.0) * weights["weight_section_richness"],
        "limitation":       min(lim_hits / 2, 1.0) * weights["weight_limitation_presence"],
        "future":           min(fut_hits / 2, 1.0) * weights["weight_future_presence"],
        "contribution":     min(contrib_hits / 3, 1.0) * weights["weight_contribution_explicit"],
        "abstract":         min(len(abstract) / 800, 1.0) * weights["weight_abstract_quality"],
    }
    total = sum(components.values())
    return total, components, None


def main():
    cfg = load_config()
    filter_cfg = {
        **cfg["filtering"],
        "year_min": cfg["corpus"]["year_min"],
        "year_max": cfg["corpus"]["year_max"],
    }
    weights = cfg["filtering"]
    keep_ratio = cfg["filtering"]["acl_keep_ratio"]

    in_path = Path(cfg["paths"]["standardized_dir"]) / "acl.json"
    out_dir = ensure_dir(cfg["paths"]["filtered_dir"])

    papers = load_json(in_path)
    log.info("loaded %d ACL papers", len(papers))

    scored, disqualified = [], []
    for p in papers:
        score, comps, reason = score_paper(p, weights, filter_cfg)
        if reason:
            disqualified.append({"source_id": p["source_id"], "reason": reason})
        else:
            scored.append({**p, "quality_score": score, "score_components": comps})

    # Sort and cut
    scored.sort(key=lambda x: x["quality_score"], reverse=True)
    n_keep = max(1, int(keep_ratio * len(papers)))
    kept = scored[:n_keep]
    discarded = scored[n_keep:]

    save_json(kept,         out_dir / "acl_kept.json")
    save_json(discarded,    out_dir / "acl_discarded_by_rank.json")
    save_json(disqualified, out_dir / "acl_disqualified.json")

    # Report
    log.info("ACL filtering:")
    log.info("  input:                %d", len(papers))
    log.info("  hard-disqualified:    %d", len(disqualified))
    log.info("  scored:               %d", len(scored))
    log.info("  kept (top %.1f%%):     %d", 100 * keep_ratio, len(kept))
    log.info("  discarded by rank:    %d", len(discarded))
    if kept:
        log.info("  score range (kept):   [%.2f, %.2f]", kept[-1]["quality_score"], kept[0]["quality_score"])
        log.info("  median score:         %.2f", kept[len(kept) // 2]["quality_score"])

    reasons = Counter(d["reason"].split("(")[0] for d in disqualified)
    log.info("  disqualification reasons:")
    for k, v in reasons.most_common():
        log.info("    %s: %d", k, v)


if __name__ == "__main__":
    main()
