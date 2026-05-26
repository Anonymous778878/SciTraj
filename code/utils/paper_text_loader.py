"""
Auto-discover paper title+abstract text on disk.

Used by tiers that need raw text (T20 cross-encoder, T21 contrastive).
Tries several config keys and several common filenames before giving up.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from utils import get_logger, load_config, load_json

log = get_logger("paper_text_loader")


# Config keys to try, in order of likelihood
CONFIG_KEY_CANDIDATES = [
    "corpus_master",
    "merged_corpus",
    "corpus_path",
    "papers_master",
    "papers_path",
    "master_corpus",
    "master",
]

# Filename patterns to try inside data/, in order
FILENAME_CANDIDATES = [
    "data/corpus/master.json",
    "data/corpus/master.jsonl",
    "data/corpus/merged.json",
    "data/corpus/papers.json",
    "data/master.json",
    "data/master.jsonl",
    "data/merged.json",
    "data/merged.jsonl",
    "data/papers.json",
    "data/papers.jsonl",
    "data/corpus_master.json",
    "data/processed/merged.json",
    "data/processed/master.json",
    "data/processed/papers.json",
]


def _is_record(rec) -> bool:
    return (
        isinstance(rec, dict)
        and ("paper_id" in rec or "id" in rec)
        and any(k in rec for k in ("abstract", "title", "text"))
    )


def _normalise_record(rec: dict) -> Optional[dict]:
    """Coerce a record to {paper_id, title, abstract}."""
    pid = rec.get("paper_id") or rec.get("id")
    if not pid:
        return None
    title = rec.get("title", "") or ""
    abstract = rec.get("abstract", "") or rec.get("text", "") or ""
    return {"paper_id": str(pid), "title": str(title), "abstract": str(abstract)}


def _read_jsonl(path: Path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _read_json_or_jsonl(path: Path):
    """Read both list-of-dicts JSON and JSONL files."""
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    try:
        return load_json(path)
    except json.JSONDecodeError:
        # Try as JSONL (some pipelines use .json with line-delimited records)
        return _read_jsonl(path)


def discover_paper_text(paper_ids: List[str], project_root: Optional[Path] = None) -> Dict[str, str]:
    """
    Return {paper_id: 'title. abstract'} for every paper_id if possible.
    Raises FileNotFoundError if no source can be located.

    Args:
        paper_ids: list of paper_id strings (in any order).
        project_root: scan starting directory (default: cwd).
    """
    if project_root is None:
        project_root = Path.cwd()

    cfg = load_config()
    candidate_paths: List[Path] = []

    # 1) Try config keys
    paths_cfg = cfg.get("paths", {})
    for key in CONFIG_KEY_CANDIDATES:
        v = paths_cfg.get(key)
        if v:
            candidate_paths.append(Path(v))

    # 2) Try common filenames in project root
    for name in FILENAME_CANDIDATES:
        candidate_paths.append(project_root / name)

    # Try each candidate
    paper_id_set = set(paper_ids)
    for path in candidate_paths:
        if not path.exists() or not path.is_file():
            continue
        log.info("trying paper-text source: %s", path)
        try:
            data = _read_json_or_jsonl(path)
        except Exception as e:
            log.warning("  failed to parse %s: %s", path, e)
            continue

        if not isinstance(data, list):
            log.warning("  %s is not a list of records, skipping", path)
            continue

        # Find records that look like papers
        records: Dict[str, str] = {}
        for rec in data:
            if not _is_record(rec):
                continue
            norm = _normalise_record(rec)
            if not norm:
                continue
            pid = norm["paper_id"]
            if pid in paper_id_set:
                title = norm["title"].strip()
                abstract = norm["abstract"].strip()
                if len(abstract) > 800:
                    abstract = abstract[:800]
                records[pid] = f"{title}. {abstract}".strip()

        if len(records) >= 0.5 * len(paper_ids):  # at least half coverage
            log.info("  found %d / %d papers in %s", len(records), len(paper_ids), path)
            return records

        log.warning("  %s has only %d/%d papers, trying next", path, len(records), len(paper_ids))

    raise FileNotFoundError(
        "Could not locate paper title+abstract on disk. Searched config keys "
        f"{CONFIG_KEY_CANDIDATES} and common filenames "
        f"{FILENAME_CANDIDATES[:5]}... none had >50% coverage of the {len(paper_ids)} "
        f"paper IDs in the graph. Add a path to config/config.yaml under "
        f"paths.corpus_master pointing to your master papers JSON file."
    )


if __name__ == "__main__":
    # Smoke test
    import sys
    sys.path.insert(0, ".")
    from gnn_utils import load_graph
    cfg = load_config()
    data = load_graph(str(Path(cfg["paths"]["graph_dir"]) / "graph_data.pt"))
    paper_ids = data["paper"].paper_id.tolist()
    log.info("looking for text for %d papers", len(paper_ids))
    text = discover_paper_text(paper_ids)
    log.info("found %d entries", len(text))
    log.info("example: %s", list(text.items())[0])
