"""
Phase 2 — Scientific Signal Extraction.

For each paper in the filtered corpus, extract all signal types:

  2.1 Section normalization           (canonical section buckets)
  2.2 Contributions                    (+ typed)
  2.3 Limitations                      (+ typed)
  2.4 Future directions                (+ typed)
  2.5 Disputes / refutations           (novel: signed-graph support)
  2.6 Quantitative claims              (novel: performance trajectories)
  2.7 Causal claims                    (novel: causal-extension edges)
  2.8 Uncertainty / hedging            (novel: trajectory fertility score)
  2.9 Figure / table captions          (novel: parallel signal channel)
  2.10 Method / dataset mentions       (entity-based methodological links)

Each signal is stored with source section, reliability score, and type label.

Output:
  data/signals/extracted.json       (complete per-paper record with all signals)
  outputs/metrics/extraction_stats.json
"""
import re
from collections import Counter, defaultdict
from pathlib import Path

from extraction_utils import (
    compile_patterns,
    compute_uncertainty_score,
    extract_matching_sentences,
    get_normalized_sections,
    get_section_text,
)
from utils import (
    ensure_dir,
    get_logger,
    load_config,
    load_json,
    save_json,
)

log = get_logger("phase2")


# ---------------------------------------------------------------------------
# Typing rules for each signal type
# ---------------------------------------------------------------------------

def classify_contribution_type(text: str) -> str:
    """Assign a contribution to one of 6 types."""
    t = text.lower()
    if any(kw in t for kw in ["dataset", "benchmark", "corpus", "collection"]):
        return "dataset"
    if any(kw in t for kw in ["prove", "theorem", "proof", "theoretical", "analytically"]):
        return "theoretical"
    if any(kw in t for kw in ["tool", "system", "pipeline", "platform", "library", "framework"]):
        return "system"
    if any(kw in t for kw in ["survey", "review", "analysis of", "comparative study"]):
        return "survey"
    if any(kw in t for kw in ["improve", "outperform", "achieve", "state-of-the-art", "sota", "accuracy", "f1"]):
        return "empirical"
    return "method"                 # default: new algorithm/model/approach


def classify_limitation_type(text: str) -> str:
    """Assign a limitation to one of 6 types."""
    t = text.lower()
    if any(kw in t for kw in ["data", "dataset", "corpus", "label"]):
        return "data"
    if any(kw in t for kw in ["generali", "domain", "transfer", "out-of-distribution"]):
        return "generalization"
    if any(kw in t for kw in ["comput", "gpu", "memory", "time", "expensive", "scalab"]):
        return "computational"
    if any(kw in t for kw in ["evaluat", "metric", "benchmark", "real-world"]):
        return "evaluation"
    if any(kw in t for kw in ["interpret", "explain", "understand", "unclear"]):
        return "interpretability"
    return "method"                 # default: model-architectural limitation


def classify_future_direction_type(text: str) -> str:
    """Assign a future direction to one of 7 types."""
    t = text.lower()
    if any(kw in t for kw in ["bias", "fair", "privacy", "safety", "ethic", "harm"]):
        return "safety_ethics"
    if any(kw in t for kw in ["theor", "analy", "prove", "understand why"]):
        return "theory"
    if any(kw in t for kw in ["scale", "larger", "scaling", "bigger"]):
        return "scaling"
    if any(kw in t for kw in ["combine", "integrat", "with other", "along with"]):
        return "integration"
    if any(kw in t for kw in ["real-world", "deploy", "production", "validate"]):
        return "validation"
    if any(kw in t for kw in ["improve", "better", "enhance", "boost"]):
        return "improvement"
    return "extension"              # default: apply to new domain/task


# ---------------------------------------------------------------------------
# Quantitative claim extraction (structured parsing, not just patterns)
# ---------------------------------------------------------------------------

QUANT_PATTERN = re.compile(
    r"(?:achieve(?:s|d)?|obtain(?:s|ed)?|improve(?:s|d)?|reduce(?:s|d)?|"
    r"outperform(?:s|ed)?|surpass(?:es|ed)?|reach(?:es|ed)?|report(?:s|ed)?|"
    r"increase(?:s|d)?|decrease(?:s|d)?|gain(?:s|ed)?)\s+"
    r"(?:\w+\s+){0,4}"
    r"(?P<value>\d+\.?\d*)\s*"
    r"(?P<unit>%|percent|points?|x|×|bleu|f1|accuracy|map|ap|auc|iou|fps|ms)",
    re.IGNORECASE,
)


def extract_quantitative_claims(text: str, max_claims: int) -> list[dict]:
    """Extract structured quantitative claims with metric, value, unit, context."""
    claims = []
    seen_values = set()

    for m in QUANT_PATTERN.finditer(text):
        value = m.group("value")
        unit = m.group("unit").lower()
        value_key = (value, unit)
        if value_key in seen_values:
            continue
        seen_values.add(value_key)

        # Local context: ±80 chars around the match
        start = max(0, m.start() - 80)
        end   = min(len(text), m.end() + 80)
        context = text[start:end].strip()

        claims.append({
            "value":   float(value) if value else None,
            "unit":    unit,
            "context": context,
            "verb":    m.group(0).split()[0].lower(),
        })
        if len(claims) >= max_claims:
            break

    return claims


# ---------------------------------------------------------------------------
# Causal claim extraction
# ---------------------------------------------------------------------------

def extract_causal_claims(text: str, patterns: list[re.Pattern], max_claims: int) -> list[dict]:
    """Extract causal sentences and split cause/effect around the connective."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text or "")
    claims = []
    seen_prefixes = set()

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 30 or len(sent) > 400:
            continue
        prefix = sent[:80].lower()
        if prefix in seen_prefixes:
            continue

        matched = None
        for p in patterns:
            m = p.search(sent)
            if m:
                matched = m
                break

        if matched is None:
            continue

        seen_prefixes.add(prefix)
        cause  = sent[: matched.start()].strip()
        effect = sent[matched.end():].strip()

        if len(cause) < 10 or len(effect) < 10:
            continue

        claims.append({
            "cause":     cause[:200],
            "effect":    effect[:200],
            "connector": matched.group().strip(),
            "sentence":  sent,
        })

        if len(claims) >= max_claims:
            break

    return claims


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------

CAPTION_PATTERN = re.compile(
    r"(?:Figure|Fig\.?|Table|Tab\.?)\s*\d+\s*[:.]\s*([^.]{20,400}\.)",
    re.IGNORECASE,
)


def extract_captions(text: str) -> list[str]:
    """Extract figure/table caption sentences."""
    if not text:
        return []
    out = []
    seen = set()
    for m in CAPTION_PATTERN.finditer(text):
        cap = m.group(1).strip()
        key = cap[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cap)
    return out


# ---------------------------------------------------------------------------
# Reliability scoring
# ---------------------------------------------------------------------------

def compute_reliability(signal: dict, section_appropriate: bool) -> float:
    """
    Reliability weight in [0, 1] based on:
      - how many patterns fired (stronger signal)
      - signal length (moderate length is more reliable)
      - whether it came from an appropriate section
    """
    r = 0.0

    # Pattern-count component
    n_patterns = signal.get("matched_pattern_count", 1)
    r += min(n_patterns / 3, 1.0) * 0.4

    # Length component (prefer 40-300 chars)
    text = signal.get("text", "")
    L = len(text)
    if 40 <= L <= 300:
        r += 0.3
    elif 30 <= L < 40 or 300 < L <= 400:
        r += 0.2
    else:
        r += 0.1

    # Section appropriateness component
    r += 0.3 if section_appropriate else 0.1

    return round(min(r, 1.0), 3)


# ---------------------------------------------------------------------------
# Master extraction — one paper at a time
# ---------------------------------------------------------------------------

def extract_signals_for_paper(paper: dict, patterns: dict, caps: dict, uncertainty_markers: dict) -> dict:
    """Run all extraction passes for one paper."""
    sections = get_normalized_sections(paper)
    abstract = paper.get("abstract", "") or ""

    # Section routing
    intro_text       = get_section_text(sections, ["introduction"], fallback=abstract)
    conclusion_text  = get_section_text(sections, ["conclusion", "discussion", "limitations", "future_work"])
    discussion_text  = get_section_text(sections, ["discussion", "conclusion", "limitations"])
    related_text     = get_section_text(sections, ["related_work", "introduction"])
    full_text        = abstract + " " + " ".join(sections.values())

    # ── 2.2 Contributions ──
    contrib_source = intro_text + " " + abstract + " " + sections.get("conclusion", "")
    contribs_raw   = extract_matching_sentences(
        contrib_source, patterns["contribution"],
        max_chars=caps["max_signal_chars"], min_chars=caps["min_signal_chars"],
        max_signals=caps["max_contributions"],
    )
    contributions = []
    for sig in contribs_raw:
        ctype = classify_contribution_type(sig["text"])
        contributions.append({
            **sig,
            "type": ctype,
            "reliability": compute_reliability(sig, section_appropriate=True),
        })

    # ── 2.3 Limitations ──
    limit_source = discussion_text + " " + sections.get("limitations", "")
    if not limit_source.strip():
        limit_source = abstract + " " + sections.get("conclusion", "")
    limits_raw = extract_matching_sentences(
        limit_source, patterns["limitation"],
        max_chars=caps["max_signal_chars"], min_chars=caps["min_signal_chars"],
        max_signals=caps["max_limitations"],
    )
    limitations = []
    for sig in limits_raw:
        ltype = classify_limitation_type(sig["text"])
        limitations.append({
            **sig,
            "type": ltype,
            "reliability": compute_reliability(sig, section_appropriate=True),
        })

    # ── 2.4 Future directions ──
    future_source = conclusion_text + " " + sections.get("future_work", "")
    if not future_source.strip():
        future_source = abstract
    futures_raw = extract_matching_sentences(
        future_source, patterns["future_direction"],
        max_chars=caps["max_signal_chars"], min_chars=caps["min_signal_chars"],
        max_signals=caps["max_future"],
    )
    futures = []
    for sig in futures_raw:
        ftype = classify_future_direction_type(sig["text"])
        unc = compute_uncertainty_score(sig["text"], uncertainty_markers)
        futures.append({
            **sig,
            "type": ftype,
            "uncertainty": unc,
            "reliability": compute_reliability(sig, section_appropriate=True),
        })

    # ── 2.5 Disputes (novel) ──
    dispute_source = related_text + " " + sections.get("introduction", abstract)
    disputes_raw = extract_matching_sentences(
        dispute_source, patterns["dispute"],
        max_chars=caps["max_signal_chars"], min_chars=caps["min_signal_chars"],
        max_signals=caps["max_disputes"],
    )
    disputes = []
    for sig in disputes_raw:
        # Strength heuristic: "does not hold / is incorrect" = strong
        strong_markers = ["does not hold", "incorrect", "wrong", "flawed", "refute"]
        weak_markers = ["unlike", "in contrast to"]
        text_lower = sig["text"].lower()
        if any(m in text_lower for m in strong_markers):
            strength = "strong"
        elif any(m in text_lower for m in weak_markers):
            strength = "weak"
        else:
            strength = "moderate"
        disputes.append({
            **sig,
            "strength": strength,
            "reliability": compute_reliability(sig, section_appropriate=True),
        })

    # ── 2.6 Quantitative claims (novel) ──
    quant_source = abstract + " " + sections.get("results", "")
    quant_claims = extract_quantitative_claims(quant_source, caps["max_quant_claims"])

    # ── 2.7 Causal claims (novel) ──
    causal_claims = extract_causal_claims(full_text, patterns["causal"], caps["max_causal"])

    # ── 2.8 Uncertainty (paper-level) ──
    paper_uncertainty = compute_uncertainty_score(abstract + " " + intro_text, uncertainty_markers)

    # ── 2.9 Captions (novel) ──
    captions = extract_captions(full_text)

    # Signal richness aggregate
    richness = (
        len(contributions)
        + len(limitations)
        + len(futures)
        + (2 if disputes else 0)
        + len(quant_claims) / 2
        + len(causal_claims) / 3
    )

    return {
        "paper_id":            paper["paper_id"],
        "title":               paper["title"],
        "year":                paper["year"],
        "venue":               paper["venue"],
        "abstract":            abstract,
        "quality_score":       paper.get("quality_score", 0),

        # Extracted signals
        "contributions":       contributions,
        "limitations":         limitations,
        "future_directions":   futures,
        "disputes":            disputes,
        "quantitative_claims": quant_claims,
        "causal_claims":       causal_claims,
        "captions":            captions[:20],       # cap to keep files manageable
        "paper_uncertainty":   paper_uncertainty,

        # Consolidated text fields for Phase 4 embedding
        "contrib_text":  " [SEP] ".join(c["text"] for c in contributions),
        "limit_text":    " [SEP] ".join(l["text"] for l in limitations),
        "future_text":   " [SEP] ".join(f["text"] for f in futures),
        "dispute_text":  " [SEP] ".join(d["text"] for d in disputes),
        "causal_text":   " [SEP] ".join(c["sentence"] for c in causal_claims),
        "caption_text":  " [SEP] ".join(captions[:20]),
        "signal_text":   " [SEP] ".join(
            [c["text"] for c in contributions]
            + [l["text"] for l in limitations]
            + [f["text"] for f in futures]
        ),

        "signal_richness": round(richness, 2),

        # Presence flags for statistics
        "has_contributions":   len(contributions) > 0,
        "has_limitations":     len(limitations) > 0,
        "has_future":          len(futures) > 0,
        "has_disputes":        len(disputes) > 0,
        "has_quant_claims":    len(quant_claims) > 0,
        "has_causal_claims":   len(causal_claims) > 0,
        "has_captions":        len(captions) > 0,
    }


def aggregate_stats(signals_list: list) -> dict:
    """Compute extraction statistics by venue."""
    by_venue = defaultdict(lambda: {
        "n_papers": 0,
        "n_contributions": 0, "n_limitations": 0, "n_future": 0,
        "n_disputes": 0, "n_quant": 0, "n_causal": 0, "n_captions": 0,
        "has_contributions_pct": 0, "has_limitations_pct": 0, "has_future_pct": 0,
        "has_disputes_pct": 0, "has_quant_pct": 0, "has_causal_pct": 0,
    })
    contribution_types = Counter()
    limitation_types   = Counter()
    future_types       = Counter()
    dispute_strengths  = Counter()

    for s in signals_list:
        v = s["venue"]
        stats = by_venue[v]
        stats["n_papers"] += 1
        stats["n_contributions"] += len(s["contributions"])
        stats["n_limitations"]   += len(s["limitations"])
        stats["n_future"]        += len(s["future_directions"])
        stats["n_disputes"]      += len(s["disputes"])
        stats["n_quant"]         += len(s["quantitative_claims"])
        stats["n_causal"]        += len(s["causal_claims"])
        stats["n_captions"]      += len(s["captions"])

        stats["has_contributions_pct"] += s["has_contributions"]
        stats["has_limitations_pct"]   += s["has_limitations"]
        stats["has_future_pct"]        += s["has_future"]
        stats["has_disputes_pct"]      += s["has_disputes"]
        stats["has_quant_pct"]         += s["has_quant_claims"]
        stats["has_causal_pct"]        += s["has_causal_claims"]

        contribution_types.update(c["type"] for c in s["contributions"])
        limitation_types.update(l["type"] for l in s["limitations"])
        future_types.update(f["type"] for f in s["future_directions"])
        dispute_strengths.update(d["strength"] for d in s["disputes"])

    # Convert counts to percentages
    for v, stats in by_venue.items():
        n = stats["n_papers"]
        if n > 0:
            for key in list(stats.keys()):
                if key.endswith("_pct"):
                    stats[key] = round(100 * stats[key] / n, 1)

    return {
        "per_venue":           dict(by_venue),
        "contribution_types":  dict(contribution_types),
        "limitation_types":    dict(limitation_types),
        "future_types":        dict(future_types),
        "dispute_strengths":   dict(dispute_strengths),
    }


def main():
    cfg = load_config()
    corpus_path  = Path(cfg["paths"]["filtered_dir"]) / "corpus.json"
    signals_dir  = ensure_dir(cfg["paths"]["signals_dir"])
    metrics_dir  = ensure_dir(cfg["paths"]["metrics_dir"])

    corpus = load_json(corpus_path)
    log.info("loaded %d papers from corpus", len(corpus))

    # Compile patterns once
    patterns = {
        "contribution":     compile_patterns(cfg["extraction_patterns"]["contribution"]),
        "limitation":       compile_patterns(cfg["extraction_patterns"]["limitation"]),
        "future_direction": compile_patterns(cfg["extraction_patterns"]["future_direction"]),
        "dispute":          compile_patterns(cfg["extraction_patterns"]["dispute"]),
        "causal":           compile_patterns(cfg["extraction_patterns"]["causal"]),
    }
    uncertainty_markers = cfg["extraction_patterns"]["uncertainty"]
    caps = cfg["signals"]

    # Extract (serial; for full corpus use multiprocessing.Pool)
    results = []
    for i, paper in enumerate(corpus):
        if i % 500 == 0 and i > 0:
            log.info("  processed %d/%d papers", i, len(corpus))
        results.append(extract_signals_for_paper(paper, patterns, caps, uncertainty_markers))

    save_json(results, signals_dir / "extracted.json")
    log.info("wrote signals/extracted.json (%d records)", len(results))

    # Stats
    stats = aggregate_stats(results)
    save_json(stats, metrics_dir / "extraction_stats.json")

    log.info("=" * 55)
    log.info("EXTRACTION STATISTICS BY VENUE")
    log.info("=" * 55)
    for v, s in stats["per_venue"].items():
        log.info("[%s] %d papers", v, s["n_papers"])
        log.info("  contributions:      %d  (%.1f%% have)", s["n_contributions"], s["has_contributions_pct"])
        log.info("  limitations:        %d  (%.1f%% have)", s["n_limitations"], s["has_limitations_pct"])
        log.info("  future directions:  %d  (%.1f%% have)", s["n_future"], s["has_future_pct"])
        log.info("  disputes:           %d  (%.1f%% have)", s["n_disputes"], s["has_disputes_pct"])
        log.info("  quant claims:       %d  (%.1f%% have)", s["n_quant"], s["has_quant_pct"])
        log.info("  causal claims:      %d  (%.1f%% have)", s["n_causal"], s["has_causal_pct"])
        log.info("  captions:           %d", s["n_captions"])

    log.info("\ncontribution-type distribution: %s", stats["contribution_types"])
    log.info("limitation-type distribution:   %s", stats["limitation_types"])
    log.info("future-type distribution:       %s", stats["future_types"])
    log.info("dispute-strength distribution:  %s", stats["dispute_strengths"])


if __name__ == "__main__":
    main()
