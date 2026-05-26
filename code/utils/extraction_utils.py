"""
Shared utilities for Phase 2 signal extraction.

Each signal type has:
  - A source-section selector (which sections to read from)
  - A set of regex patterns (from config)
  - A classifier that assigns a type to each extracted signal
  - Quality filters (length, deduplication)
"""
import re
from typing import Iterable


# ---------------------------------------------------------------------------
# Section routing — which canonical section bucket does each signal come from?
# ---------------------------------------------------------------------------

def normalize_section_name(name: str) -> str:
    """Map a raw section name to a canonical bucket."""
    n = (name or "").lower().strip()
    if any(kw in n for kw in ["abstract"]):
        return "abstract"
    if any(kw in n for kw in ["introduction", "intro"]):
        return "introduction"
    if any(kw in n for kw in ["related", "prior work", "literature", "background"]):
        return "related_work"
    if any(kw in n for kw in ["method", "approach", "model", "architecture", "system", "framework"]):
        return "methods"
    if any(kw in n for kw in ["experiment", "result", "evaluation", "empirical", "analysis"]):
        return "results"
    if any(kw in n for kw in ["discussion", "analysis"]):
        return "discussion"
    if any(kw in n for kw in ["limitation", "threat", "shortcoming"]):
        return "limitations"
    if any(kw in n for kw in ["future", "next steps"]):
        return "future_work"
    if any(kw in n for kw in ["conclusion", "summary", "closing"]):
        return "conclusion"
    return "other"


def get_normalized_sections(paper: dict) -> dict:
    """
    Return dict mapping canonical_section_name -> concatenated text.
    If the same canonical bucket appears multiple times (e.g. two 'Results'
    subsections), their text is joined.
    """
    buckets = {}
    for raw_name, text in (paper.get("sections") or {}).items():
        if not text:
            continue
        canonical = normalize_section_name(raw_name)
        buckets.setdefault(canonical, []).append(text)
    return {k: " ".join(v) for k, v in buckets.items()}


def get_section_text(sections: dict, section_types: Iterable[str], fallback: str = "") -> str:
    """Extract concatenated text from a set of canonical sections.
    Returns fallback (usually abstract) if none are present."""
    parts = []
    for st in section_types:
        if st in sections and sections[st]:
            parts.append(sections[st])
    return " ".join(parts) if parts else fallback


# ---------------------------------------------------------------------------
# Sentence splitting — simple and robust (no dependency on external tools)
# ---------------------------------------------------------------------------

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter. Good enough for pattern-based extraction."""
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    sentences = SENTENCE_SPLIT.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Pattern-based extraction
# ---------------------------------------------------------------------------

def compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Pre-compile a list of regex patterns with IGNORECASE."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def extract_matching_sentences(
    text: str,
    patterns: list[re.Pattern],
    min_chars: int = 20,
    max_chars: int = 500,
    max_signals: int = 5,
) -> list[dict]:
    """
    Return sentences where at least one pattern matches.
    Each signal includes the sentence and which patterns fired.
    Deduplicates by normalized prefix.
    """
    sentences = split_sentences(text)
    out = []
    seen_prefixes = set()

    for sent in sentences:
        if len(sent) < min_chars:
            continue
        if len(sent) > max_chars:
            sent = sent[:max_chars].rsplit(" ", 1)[0] + "..."

        matching = [p.pattern for p in patterns if p.search(sent)]
        if not matching:
            continue

        # Simple dedup: skip if first 80 chars match an already-seen signal
        prefix = sent[:80].lower().strip()
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        out.append({
            "text":           sent,
            "matched_pattern_count": len(matching),
            "matched_patterns":      matching[:3],  # cap at 3 for compactness
        })
        if len(out) >= max_signals:
            break

    return out


# ---------------------------------------------------------------------------
# Uncertainty / hedging scoring
# ---------------------------------------------------------------------------

def compute_uncertainty_score(text: str, markers: dict) -> float:
    """
    Return a hedging score in [0, 1].
    markers: dict with keys 'low', 'medium', 'high' mapping to lists of strings.
    """
    if not text:
        return 0.0
    text_lower = text.lower()

    score = 0.0
    for tier, weight in (("low", 0.1), ("medium", 0.3), ("high", 0.5)):
        for m in markers.get(tier, []):
            if m in text_lower:
                score += weight

    return min(score, 1.0)
