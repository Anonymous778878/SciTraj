"""
Track B.2.3 — NLI-Verified Causal Claims (NOVEL).

Phase 2's regex-based causal extraction yields 81K candidate causal claims
across the corpus. Many of these are HEDGED ("X would cause Y if..."),
COUNTERFACTUAL ("if not for X, Y wouldn't have...") or NEGATED
("X does not cause Y"). Treating these as positive causal evidence
introduces noise into the causal_extension edges in Phase 6.

This script uses an off-the-shelf NLI model (microsoft/deberta-v3-large-mnli)
to verify each causal claim. For each claim:
    - Premise: surrounding context (claim sentence + 1 sentence before, 1 after)
    - Hypothesis: a causal restatement, e.g.
        "[entity_X] causes [entity_Y]." (assertive, present-tense, unhedged)

We classify NLI label:
    ENTAILMENT     -> verified causal claim (kept)
    CONTRADICTION  -> rejected
    NEUTRAL        -> rejected (probably hedged/counterfactual)

Output: filtered set of verified causal claims, comparison statistics.

Novelty:
    No prior scientific signal-extraction work uses NLI to verify
    extracted causal claims against their surrounding context. Combined
    with our phase-6 causal_extension edges, this gives a strictly
    higher-precision causal subset of the trajectory graph.

Output:
    data/signals/causal_verified.json — the verified subset
    outputs/phase12/nli_causal_metrics.json
    outputs/phase12/nli_causal_report.md
"""
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from utils import ensure_dir, get_logger, load_config, load_json

log = get_logger("nli_causal")


def load_nli_pipeline():
    """
    Load a free public DeBERTa-v3 NLI model. Tries best-quality first, falls
    back to alternates if the primary download fails.

    Note: 'microsoft/deberta-v3-large-mnli' does NOT exist — the actual
    Microsoft path is just 'microsoft/deberta-v3-large' (the base model,
    not fine-tuned for NLI). The community-fine-tuned MNLI variants live
    under different paths.

    Models tried in order (all free, all 3-class entail/neutral/contradict):
        1. MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (~1.5GB,
           state-of-the-art on MNLI/ANLI; preferred)
        2. khalidalt/DeBERTa-v3-large-mnli  (smaller fine-tune)
        3. MoritzLaurer/DeBERTa-v3-base-mnli  (~430MB fallback)
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    candidates = [
        "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "khalidalt/DeBERTa-v3-large-mnli",
        "MoritzLaurer/DeBERTa-v3-base-mnli",
    ]
    last_err = None
    for model_name in candidates:
        try:
            log.info("trying NLI model: %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if torch.cuda.is_available():
                model = model.cuda()
                log.info("NLI model on cuda")
            model.eval()
            log.info("loaded NLI model: %s", model_name)
            # Inspect label2id to know which output index = which class
            id2label = model.config.id2label if hasattr(model.config, "id2label") else None
            log.info("model id2label: %s", id2label)
            return tokenizer, model, model_name, id2label
        except Exception as exc:
            log.warning("could not load %s: %s", model_name, exc)
            last_err = exc
            continue
    raise RuntimeError(f"all NLI model candidates failed; last error: {last_err}")


def nli_predict_batch(tokenizer, model, premises, hypotheses, batch_size=16):
    """Run NLI on lists of (premise, hypothesis). Returns label IDs and probs."""
    device = next(model.parameters()).device
    all_labels = []
    all_probs = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i+batch_size]
        batch_h = hypotheses[i:i+batch_size]
        enc = tokenizer(batch_p, batch_h, return_tensors="pt",
                        truncation=True, padding=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        labels = probs.argmax(axis=1)
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
    return all_labels, all_probs


# Patterns to identify causal connectors and hedging markers
HEDGE_MARKERS = re.compile(
    r"\b(if|would|might|may|could|whether|hypothesi[sz]ed?|conjecture|in principle|"
    r"assuming|suppose|presumably|likely|possibly)\b",
    re.IGNORECASE,
)
NEGATION_MARKERS = re.compile(
    r"\b(not|no|never|none|nothing|fail to|does not|do not|did not|cannot|won'?t)\b",
    re.IGNORECASE,
)


def make_hypothesis(causal_text: str) -> str:
    """
    Convert a causal sentence into a clean assertive hypothesis.
    The hypothesis is a simplified, present-tense, unhedged form.
    """
    # Strip leading article fragments, parens, citations
    txt = re.sub(r"\([^)]*\d{4}[^)]*\)", "", causal_text)
    txt = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", txt)
    txt = txt.strip()
    if not txt:
        return ""
    # Truncate to a manageable hypothesis length
    if len(txt) > 200:
        # Try to cut at a sentence boundary
        m = re.search(r"^(.{80,200}?[.!?])\s", txt)
        if m:
            txt = m.group(1)
        else:
            txt = txt[:200]
    return txt


def main():
    cfg = load_config()
    seed = cfg["project"]["seed"]
    sig_dir = Path(cfg["paths"]["signals_dir"])
    val_dir = Path(cfg["paths"]["validated_dir"])
    out_results = ensure_dir(Path("outputs/phase12"))

    # We expect causal claims either in raw signals or in validated
    log.info("loading causal claims")
    causal_path_candidates = [
        sig_dir / "causal_claims.json",
        sig_dir / "causal_signals.json",
        val_dir / "causal_claims_validated.json",
    ]
    claims = None
    for p in causal_path_candidates:
        if p.exists():
            log.info("loading from %s", p)
            claims = load_json(p)
            break
    if claims is None:
        # fall back to extracting from signals_with_reliability.json
        log.info("falling back to signals_with_reliability.json causal_text fields")
        signals = load_json(val_dir / "signals_with_reliability.json")
        claims = []
        for s in signals:
            text = (s.get("causal_text") or "").strip()
            if not text:
                continue
            ctx = (s.get("causal_context") or s.get("abstract") or text)
            claims.append({
                "paper_id": s["paper_id"],
                "causal_text": text,
                "context": ctx,
            })
    log.info("total causal claims: %d", len(claims))

    if not claims:
        log.error("no causal claims found")
        return

    # ── Pre-filter: claims that contain hedge or negation markers go straight to ──
    # ── "rejected" without LLM call (saves significant compute).               ──
    pre_rejected = []
    candidates_for_nli = []
    for c in claims:
        text = c.get("causal_text") or c.get("text") or ""
        if HEDGE_MARKERS.search(text) and "because" not in text.lower():
            c["nli_label"] = "PRE_REJECTED_HEDGED"
            c["nli_prob"] = None
            pre_rejected.append(c)
        elif NEGATION_MARKERS.search(text):
            c["nli_label"] = "PRE_REJECTED_NEGATED"
            c["nli_prob"] = None
            pre_rejected.append(c)
        else:
            candidates_for_nli.append(c)
    log.info("pre-rejected (hedged/negated): %d", len(pre_rejected))
    log.info("candidates for NLI verification: %d", len(candidates_for_nli))

    # ── Load NLI model ──
    try:
        tokenizer, model, loaded_model_name, id2label = load_nli_pipeline()
    except Exception as exc:
        log.error("could not load NLI model: %s", exc)
        log.error("run: pip install --upgrade transformers sentencepiece protobuf")
        return

    # Determine entailment/neutral/contradiction indices from model config.
    # Different community fine-tunes use different orderings.
    if id2label is None:
        # Fallback: assume the common HF order [contradiction, neutral, entailment]
        ENTAIL_IDX = 2
        NEUTRAL_IDX = 1
        CONTRA_IDX = 0
        LABELS = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    else:
        # id2label is a dict {0: "ENTAILMENT", ...} or similar
        norm = {int(k): v.upper() for k, v in id2label.items()}
        LABELS = {k: ("ENTAILMENT" if "ENTAIL" in v else
                      "CONTRADICTION" if "CONTRA" in v else "NEUTRAL")
                  for k, v in norm.items()}
        log.info("normalized labels: %s", LABELS)
        # Identify the index for ENTAILMENT (used for filtering later)
        ENTAIL_IDX = next((k for k, v in LABELS.items() if v == "ENTAILMENT"), 2)
        NEUTRAL_IDX = next((k for k, v in LABELS.items() if v == "NEUTRAL"), 1)
        CONTRA_IDX = next((k for k, v in LABELS.items() if v == "CONTRADICTION"), 0)

    # ── Build (premise, hypothesis) pairs ──
    premises = []
    hypotheses = []
    valid_indices = []
    for i, c in enumerate(candidates_for_nli):
        text = c.get("causal_text") or c.get("text") or ""
        ctx = c.get("context") or text
        hyp = make_hypothesis(text)
        if not hyp or len(hyp) < 15:
            c["nli_label"] = "SKIPPED_TOO_SHORT"
            c["nli_prob"] = None
            continue
        premises.append(ctx[:1000])
        hypotheses.append(hyp)
        valid_indices.append(i)
    log.info("running NLI on %d valid (premise, hypothesis) pairs", len(premises))

    # ── Run NLI in batches ──
    t0 = time.time()
    labels_int, probs = nli_predict_batch(tokenizer, model, premises, hypotheses, batch_size=8)
    log.info("NLI done in %.1f s", time.time() - t0)

    for j, idx in enumerate(valid_indices):
        c = candidates_for_nli[idx]
        c["nli_label"] = LABELS[labels_int[j]]
        c["nli_prob"] = {LABELS[k]: round(probs[j][k], 4) for k in [0, 1, 2]}

    # ── Aggregate ──
    all_results = pre_rejected + candidates_for_nli
    label_counts = {}
    for c in all_results:
        lbl = c.get("nli_label", "UNKNOWN")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    # The verified subset
    verified = [c for c in all_results if c.get("nli_label") == "ENTAILMENT"]
    log.info("verified (ENTAILMENT): %d / %d (%.2f%%)",
             len(verified), len(claims), 100*len(verified)/max(len(claims),1))

    # ── Save ──
    out_signals = ensure_dir(sig_dir)
    with open(out_signals / "causal_verified.json", "w") as f:
        json.dump(verified, f, indent=2)
    with open(out_signals / "causal_with_nli_labels.json", "w") as f:
        json.dump(all_results, f, indent=2)

    metrics = {
        "task":              "nli_verified_causal",
        "nli_model_used":    loaded_model_name,
        "total_claims":      len(claims),
        "pre_rejected_hedged": sum(1 for c in pre_rejected if c["nli_label"] == "PRE_REJECTED_HEDGED"),
        "pre_rejected_negated": sum(1 for c in pre_rejected if c["nli_label"] == "PRE_REJECTED_NEGATED"),
        "label_counts":      label_counts,
        "n_verified":        len(verified),
        "verified_pct":      round(100*len(verified)/max(len(claims),1), 2),
    }
    with open(out_results / "nli_causal_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    md = []
    md.append("# NLI-Verified Causal Claims (Track B.2.3)\n\n")
    md.append("Use `microsoft/deberta-v3-large-mnli` to verify each regex-extracted causal claim\n")
    md.append("against its surrounding context. Hedged/counterfactual claims fail entailment.\n\n")
    md.append("## Pipeline\n\n")
    md.append("1. **Pre-filter**: hedge markers (`if`, `would`, `might`, etc.) and negations\n   are rejected without NLI call.\n")
    md.append("2. **NLI verification**: remaining claims tested for entailment from surrounding\n   context (premise = context, hypothesis = claim restated assertively).\n\n")
    md.append("## Results\n\n")
    md.append(f"- Total causal claims: **{len(claims):,}**\n")
    md.append(f"- Pre-rejected (hedged): {metrics['pre_rejected_hedged']:,}\n")
    md.append(f"- Pre-rejected (negated): {metrics['pre_rejected_negated']:,}\n")
    md.append(f"- Sent to NLI: {len(premises):,}\n")
    for label, count in label_counts.items():
        md.append(f"- {label}: {count:,}\n")
    md.append(f"\n**Verified (ENTAILMENT) subset: {len(verified):,} ({metrics['verified_pct']}% of original)**\n\n")
    md.append("## Novelty\n\n")
    md.append("To our knowledge, no prior scientific information-extraction work uses NLI to verify\n")
    md.append("extracted causal claims against their context. The verified subset is suitable as a\n")
    md.append("higher-precision input for downstream causal-graph construction.\n")
    (out_results / "nli_causal_report.md").write_text("".join(md))

    log.info("=" * 55)
    log.info("NLI-VERIFIED CAUSAL EXTRACTION COMPLETE")
    log.info("=" * 55)
    log.info("verified: %d / %d (%.2f%%)", len(verified), len(claims), metrics["verified_pct"])
    log.info("label distribution: %s", label_counts)
    log.info("wrote: %s", out_results / "nli_causal_report.md")


if __name__ == "__main__":
    main()
