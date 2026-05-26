"""
Phase 1.1 — Standardize raw JSON inputs into a canonical schema.

What it does:
  1. Loads all raw files listed in config.input_files per venue.
  2. Deduplicates within each venue by source paperId.
  3. Flattens section content (list or string) into a single string.
  4. Recovers abstracts polluted with the NeurIPS OpenReview boilerplate by
     using the first ~500 chars of the Introduction.
  5. Normalizes missing authors to None.

Output: data/standardized/{acl,cvpr,neurips}.json
        data/standardized/standardization_report.json
"""
from pathlib import Path

from utils import (
    BOILERPLATE_MARKER,
    ensure_dir,
    get_logger,
    load_config,
    load_json,
    save_json,
)

log = get_logger("phase1.1")


def stringify_section(val) -> str:
    """Section content may be a string or list of paragraphs — unify to string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return " ".join(str(x).strip() for x in val if x is not None)
    return str(val).strip()


def recover_abstract(raw_abstract: str, intro_text: str) -> tuple[str, bool]:
    """
    If abstract is empty/too short or contains the OpenReview boilerplate,
    recover from the Introduction.
    Returns: (abstract_text, recovered_flag)
    """
    clean = (raw_abstract or "").strip()
    if len(clean) < 100 or BOILERPLATE_MARKER in clean:
        if intro_text and len(intro_text) > 100:
            return intro_text[:500].strip(), True
        return clean, False
    return clean, False


def standardize_paper(raw: dict, venue: str) -> dict:
    """Map one raw paper dict to canonical schema."""
    sections = {k: stringify_section(v) for k, v in (raw.get("sections") or {}).items()}
    intro = sections.get("Introduction", "")
    abstract, recovered = recover_abstract(raw.get("abstract"), intro)

    authors = raw.get("authors") or []
    if not authors:
        authors = None

    return {
        "paper_id":           None,          # assigned later
        "source_id":          raw.get("paperId"),
        "title":              (raw.get("title") or "").strip(),
        "year":               raw.get("year"),
        "venue":              venue,
        "abstract":           abstract,
        "abstract_recovered": recovered,
        "sections":           sections,
        "authors":            authors,
        "references":         None,
        "citations":          None,
        "url":                raw.get("url"),
    }


def load_and_union(raw_dir: Path, filenames: list, venue: str) -> list:
    """Load all files for one venue, deduplicate by source paperId."""
    by_sid = {}
    for fname in filenames:
        path = raw_dir / fname
        if not path.exists():
            log.warning("missing input file: %s", path)
            continue
        raw_papers = load_json(path)
        before = len(by_sid)
        for raw in raw_papers:
            sid = raw.get("paperId")
            if sid and sid not in by_sid:
                by_sid[sid] = raw
        log.info("  %s: %d raw, %d new unique", fname, len(raw_papers), len(by_sid) - before)
    return list(by_sid.values())


def compute_stats(standardized: list, venue: str) -> dict:
    """Compute per-venue standardization statistics."""
    n = len(standardized)
    if n == 0:
        return {"venue": venue, "n_papers_after_union": 0}

    years = [p["year"] for p in standardized if p["year"]]
    return {
        "venue":                            venue,
        "n_papers_after_union":             n,
        "abstracts_recovered_from_intro":   sum(1 for p in standardized if p["abstract_recovered"]),
        "papers_with_usable_abstract":      sum(1 for p in standardized if len(p["abstract"]) >= 100),
        "papers_with_authors":              sum(1 for p in standardized if p["authors"]),
        "year_range":                       f"{min(years)}-{max(years)}" if years else "n/a",
    }


def main():
    cfg = load_config()
    raw_dir = Path(cfg["input_files"]["raw_dir"])
    out_dir = ensure_dir(cfg["paths"]["standardized_dir"])

    all_stats = {}
    for venue, filenames in cfg["input_files"]["venues"].items():
        log.info("[%s] loading and merging files:", venue)
        raw_papers = load_and_union(raw_dir, filenames, venue)
        standardized = [standardize_paper(p, venue) for p in raw_papers]

        stats = compute_stats(standardized, venue)
        all_stats[venue] = stats

        out_path = out_dir / f"{venue.lower()}.json"
        save_json(standardized, out_path)
        log.info("[%s] wrote %d papers -> %s", venue, len(standardized), out_path)
        for k, v in stats.items():
            log.info("  %s: %s", k, v)

    save_json(all_stats, out_dir / "standardization_report.json")
    total = sum(s["n_papers_after_union"] for s in all_stats.values())
    log.info("[TOTAL] %d papers across all venues", total)


if __name__ == "__main__":
    main()
