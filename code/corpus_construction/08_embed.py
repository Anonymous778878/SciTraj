"""
Phase 4 — Multi-view embedding generation.

For each paper we produce up to 8 embedding "views":
  abstract   : title + abstract            (primary similarity signal, topic clustering)
  signal     : contributions+limits+future (research intent, general trajectory)
  contrib    : contributions only          (Direct-Extension edges)
  limit      : limitations only            (Limitation-Addressed edges)
  future     : future directions only      (Future-Realized edges)
  dispute    : dispute signals             (negative edges)
  causal     : causal claims               (Causal-Extension edges)
  caption    : figure/table captions       (CVPR-heavy parallel channel)

Fallback policy:
  If a paper's signal text for a view is empty, we fall back to the abstract
  embedding for that view. This ensures every paper has every view, which
  keeps downstream phases simpler and means edge construction never needs to
  special-case missing data.

Output (under data/embeddings/):
  {view}_embs.npy          — raw embeddings, shape (N, D)
  {view}_embs_norm.npy     — L2-normalized, shape (N, D)
  paper_id_order.json      — list of paper_ids in the row order of every .npy
  embedding_report.json    — model, dim, per-view fallback rate, time taken

Memory notes:
  - Processes one view at a time, so peak memory is a single view's worth.
  - For 35,000 papers × 768 dims: 35K × 768 × 4 bytes = 107 MB per view.
  - Model footprint ~500MB for specter2_base in FP32.
  - Total RAM needed: ~1-2 GB on GPU, ~3-4 GB on CPU.

Usage:
  python3 src/08_embed.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import ensure_dir, get_logger, load_config, load_json, save_json

log = get_logger("phase4")


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def pick_device(preference: str) -> str:
    """Resolve 'auto' to cuda if available, else cpu."""
    if preference == "cuda":
        if not torch.cuda.is_available():
            log.warning("config says 'cuda' but no GPU available, falling back to cpu")
            return "cpu"
        return "cuda"
    if preference == "cpu":
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Text construction per view
# ---------------------------------------------------------------------------

def build_view_text(paper: dict, view: str) -> tuple[str, bool]:
    """
    Return (text, used_fallback) for a given view.
    used_fallback=True means we fell back to abstract because the view-specific
    text was empty or too short.
    """
    abstract = paper.get("abstract", "") or ""
    title    = paper.get("title", "") or ""

    if view == "abstract":
        # Title acts as a stronger prior for clustering
        return (title + " [SEP] " + abstract).strip(), False

    # Map view name to the signal field written by Phase 2
    field_map = {
        "signal":  "signal_text",
        "contrib": "contrib_text",
        "limit":   "limit_text",
        "future":  "future_text",
        "dispute": "dispute_text",
        "causal":  "causal_text",
        "caption": "caption_text",
    }

    if view not in field_map:
        raise ValueError(f"unknown view: {view}")

    text = (paper.get(field_map[view], "") or "").strip()

    # Fallback: use abstract if the view-specific text is too short to embed usefully
    if len(text) < 20:
        fallback = (title + " [SEP] " + abstract).strip()
        return fallback, True

    # Prepend title so the embedding has consistent anchoring across views
    return (title + " [SEP] " + text).strip(), False


# ---------------------------------------------------------------------------
# Batched encoding
# ---------------------------------------------------------------------------

def encode_view(
    model: SentenceTransformer,
    papers: list[dict],
    view: str,
    batch_size: int,
    max_seq_length: int,
) -> tuple[np.ndarray, int]:
    """
    Encode one view for all papers. Returns (embeddings, n_fallbacks).
    """
    texts = []
    n_fallbacks = 0

    for p in papers:
        text, used_fallback = build_view_text(p, view)
        texts.append(text)
        if used_fallback:
            n_fallbacks += 1

    # Respect max_seq_length at the tokenizer level
    model.max_seq_length = max_seq_length

    # `encode` batches internally and shows its own progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,          # we do L2 normalization separately
    )
    return embeddings.astype(np.float32), n_fallbacks


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows, safe against zero vectors."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (x / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    emb_cfg = cfg["embeddings"]
    emb_dir = ensure_dir(cfg["paths"]["embeddings_dir"])

    device = pick_device(emb_cfg.get("device", "auto"))
    log.info("device: %s", device)

    # Load validated signals (from Phase 3)
    signals_path = Path(cfg["paths"]["validated_dir"]) / "signals_with_reliability.json"
    if not signals_path.exists():
        log.error("missing %s — run Phase 3 first", signals_path)
        return
    papers = load_json(signals_path)
    log.info("loaded %d papers from Phase 3 output", len(papers))

    # Sort papers by paper_id so the row order is deterministic and stable
    papers.sort(key=lambda p: p["paper_id"])
    paper_ids = [p["paper_id"] for p in papers]
    save_json(paper_ids, emb_dir / "paper_id_order.json")

    # Load model once
    log.info("loading model: %s", emb_cfg["model_name"])
    t0 = time.time()
    model = SentenceTransformer(emb_cfg["model_name"], device=device)
    load_time = time.time() - t0
    log.info("model loaded in %.1fs  (dim=%d)",
             load_time, model.get_sentence_embedding_dimension())

    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != emb_cfg["embedding_dim"]:
        log.warning("config dim=%d but model dim=%d — using model dim",
                    emb_cfg["embedding_dim"], actual_dim)

    # Encode each view sequentially
    report = {
        "model_name":    emb_cfg["model_name"],
        "embedding_dim": actual_dim,
        "n_papers":      len(papers),
        "device":        device,
        "load_time_s":   round(load_time, 1),
        "per_view":      {},
    }

    for view in emb_cfg["views"]:
        log.info("[%s] encoding %d papers", view, len(papers))
        t0 = time.time()
        embs, n_fallbacks = encode_view(
            model, papers, view,
            batch_size=emb_cfg["batch_size"],
            max_seq_length=emb_cfg["max_seq_length"],
        )
        elapsed = time.time() - t0

        # Save raw and normalized
        raw_path  = emb_dir / f"{view}_embs.npy"
        norm_path = emb_dir / f"{view}_embs_norm.npy"
        np.save(raw_path, embs)

        if emb_cfg["normalize"]:
            np.save(norm_path, l2_normalize(embs))

        fallback_pct = round(100 * n_fallbacks / len(papers), 1) if papers else 0
        log.info("[%s] done in %.1fs  (fallback=%d/%d = %.1f%%)",
                 view, elapsed, n_fallbacks, len(papers), fallback_pct)

        report["per_view"][view] = {
            "seconds":      round(elapsed, 1),
            "n_fallback":   n_fallbacks,
            "fallback_pct": fallback_pct,
            "shape":        list(embs.shape),
        }

    save_json(report, emb_dir / "embedding_report.json")
    log.info("wrote embedding_report.json")

    # Summary
    total_s = sum(v["seconds"] for v in report["per_view"].values())
    log.info("=" * 55)
    log.info("PHASE 4 COMPLETE")
    log.info("=" * 55)
    log.info("total encoding time: %.1f min", total_s / 60)
    log.info("views produced:")
    for view, info in report["per_view"].items():
        log.info("  %-10s  shape=%s  fallback=%.1f%%",
                 view, info["shape"], info["fallback_pct"])


if __name__ == "__main__":
    main()
