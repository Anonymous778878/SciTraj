"""
Phase 5.1 — Build FAISS indexes for each embedding view.

FAISS IndexFlatIP is exact inner-product search. On L2-normalized vectors this
is equivalent to cosine similarity. For corpora up to ~1M vectors this is fine
on CPU and milliseconds per query. For very large corpora, switch to
IndexIVFFlat (approximate but still very good quality).

Output: data/retrieval/{view}.index          FAISS index files
        data/retrieval/retrieval_report.json
"""
import time
from pathlib import Path

import faiss
import numpy as np

from utils import ensure_dir, get_logger, load_config, load_json, save_json

log = get_logger("phase5.1")


def build_index(embeddings: np.ndarray, index_type: str, nlist: int = 1024):
    """Construct the appropriate FAISS index."""
    dim = embeddings.shape[1]
    if index_type == "flat_ip":
        # Exact inner-product search. Best quality, O(N) per query.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    if index_type == "ivf_flat":
        # Approximate, much faster on >500K vectors.
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        # IVF needs training on a sample
        train_size = min(50_000, embeddings.shape[0])
        sample_idx = np.random.choice(embeddings.shape[0], train_size, replace=False)
        index.train(embeddings[sample_idx])
        index.add(embeddings)
        index.nprobe = 16                     # search quality knob
        return index

    raise ValueError(f"unknown index_type: {index_type}")


def main():
    cfg = load_config()
    emb_dir = Path(cfg["paths"]["embeddings_dir"])
    ret_dir = ensure_dir(cfg["paths"]["retrieval_dir"])

    index_type = cfg["retrieval"]["index_type"]
    views = cfg["embeddings"]["views"]

    report = {
        "index_type": index_type,
        "per_view":   {},
    }

    for view in views:
        emb_path = emb_dir / f"{view}_embs_norm.npy"
        if not emb_path.exists():
            log.warning("missing %s — skipping", emb_path)
            continue

        log.info("[%s] loading embeddings", view)
        embs = np.load(emb_path).astype(np.float32)
        log.info("[%s] shape=%s  building %s index", view, embs.shape, index_type)

        t0 = time.time()
        index = build_index(embs, index_type)
        build_time = time.time() - t0

        index_path = ret_dir / f"{view}.index"
        faiss.write_index(index, str(index_path))
        log.info("[%s] wrote %s  (built in %.1fs)", view, index_path, build_time)

        report["per_view"][view] = {
            "n_vectors":    int(embs.shape[0]),
            "dim":          int(embs.shape[1]),
            "build_time_s": round(build_time, 1),
            "index_path":   str(index_path),
        }

    save_json(report, ret_dir / "index_build_report.json")
    log.info("wrote index_build_report.json (%d indexes)", len(report["per_view"]))


if __name__ == "__main__":
    main()
