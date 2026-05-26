"""
Phase 1.6 — Create temporal train/val/test splits and write the corpus
statistics report that will go into the paper's Data section.

Train: 2010-2020, Val: 2021-2022, Test: 2023-2024.
"""
import re
from collections import Counter
from pathlib import Path

from utils import ensure_dir, get_logger, load_config, load_json, save_json

log = get_logger("phase1.6")

# Simple presence patterns for the statistics report (not extraction)
LIM_PATS = [r"\blimitation", r"\bweakness", r"\bcannot\b", r"\bfails?\b", r"\bdrawback"]
FUT_PATS = [r"\bfuture\s+work", r"\bopen\s+question", r"\bplan\s+to\b",
            r"\bcould\s+be\s+extended"]


def assign_split(year: int, train: list, val: list, test: list) -> str:
    """Assign split label by year."""
    if train[0] <= year <= train[1]:
        return "train"
    if val[0]   <= year <= val[1]:
        return "val"
    if test[0]  <= year <= test[1]:
        return "test"
    return "unknown"


def per_venue_stats(corpus: list) -> dict:
    """Compute per-venue statistics for the report."""
    venues = sorted(set(p["venue"] for p in corpus))
    out = {
        "mean_abstract_len_per_venue":       {},
        "mean_section_chars_per_venue":      {},
        "pct_limitation_language_per_venue": {},
        "pct_future_language_per_venue":     {},
        "pct_with_authors_per_venue":        {},
        "pct_abstract_recovered_per_venue":  {},
    }

    for v in venues:
        vp = [p for p in corpus if p["venue"] == v]
        if not vp:
            continue
        abs_lens = [len(p.get("abstract", "") or "") for p in vp]
        sec_lens = [sum(len(s) for s in (p.get("sections") or {}).values()) for p in vp]

        n_lim = n_fut = 0
        for p in vp:
            text = " ".join([p.get("abstract", "") or ""] +
                            list((p.get("sections") or {}).values())).lower()
            if any(re.search(pat, text) for pat in LIM_PATS):
                n_lim += 1
            if any(re.search(pat, text) for pat in FUT_PATS):
                n_fut += 1

        n = len(vp)
        out["mean_abstract_len_per_venue"][v]       = int(sum(abs_lens) / n)
        out["mean_section_chars_per_venue"][v]      = int(sum(sec_lens) / n)
        out["pct_limitation_language_per_venue"][v] = round(100 * n_lim / n, 1)
        out["pct_future_language_per_venue"][v]     = round(100 * n_fut / n, 1)
        out["pct_with_authors_per_venue"][v]        = round(100 * sum(1 for p in vp if p.get("authors")) / n, 1)
        out["pct_abstract_recovered_per_venue"][v]  = round(100 * sum(1 for p in vp if p.get("abstract_recovered")) / n, 1)

    return out


def year_venue_matrix(corpus: list) -> dict:
    """Build a year-by-venue count matrix."""
    matrix = {}
    for p in corpus:
        matrix.setdefault(p["year"], {}).setdefault(p["venue"], 0)
        matrix[p["year"]][p["venue"]] += 1
    return {str(y): matrix[y] for y in sorted(matrix.keys())}


def main():
    cfg = load_config()
    corpus_path = Path(cfg["paths"]["filtered_dir"]) / "corpus.json"
    split_dir   = ensure_dir(cfg["paths"]["splits_dir"])
    metrics_dir = ensure_dir(cfg["paths"]["metrics_dir"])

    corpus = load_json(corpus_path)
    log.info("loaded corpus with %d papers", len(corpus))

    # Assign split per paper
    train = cfg["corpus"]["train_years"]
    val   = cfg["corpus"]["val_years"]
    test  = cfg["corpus"]["test_years"]
    for p in corpus:
        p["split"] = assign_split(p["year"], train, val, test)

    # Persist split index files
    splits = {s: [p["paper_id"] for p in corpus if p["split"] == s]
              for s in ("train", "val", "test", "unknown")}
    for s in ("train", "val", "test"):
        save_json(splits[s], split_dir / f"{s}_ids.json")
    save_json(corpus, corpus_path)

    # Build report
    report = {
        "total_papers":               len(corpus),
        "venues":                     dict(Counter(p["venue"] for p in corpus)),
        "splits":                     {s: len(splits[s]) for s in splits},
        "year_range":                 [min(p["year"] for p in corpus),
                                       max(p["year"] for p in corpus)],
        "papers_per_year_per_venue":  year_venue_matrix(corpus),
        **per_venue_stats(corpus),
    }
    save_json(report, metrics_dir / "corpus_statistics.json")

    # Print summary
    log.info("=" * 55)
    log.info("PHASE 1 COMPLETE — CORPUS STATISTICS")
    log.info("=" * 55)
    log.info("total papers: %d", report["total_papers"])
    log.info("year range:   %d – %d", *report["year_range"])
    log.info("by venue:")
    for v, c in report["venues"].items():
        log.info("  %-8s %d", v, c)
    log.info("by split:")
    for s, c in report["splits"].items():
        if c:
            log.info("  %-8s %d", s, c)
    log.info("%% with limitation language:")
    for v, c in report["pct_limitation_language_per_venue"].items():
        log.info("  %-8s %.1f%%", v, c)
    log.info("%% with future direction language:")
    for v, c in report["pct_future_language_per_venue"].items():
        log.info("  %-8s %.1f%%", v, c)


if __name__ == "__main__":
    main()
