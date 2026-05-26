# SciTraj

**A typed, claim-grounded citation corpus for tracing how research evolves across NLP, ML, and Vision.**

[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-blue.svg)](LICENSE-DATA)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE-CODE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

SciTraj is a typed citation corpus of **32,559 papers** from NLP, ML, and Vision (2015–2024), connected by **573,126 directed edges** across six relation types. Unlike standard citation graphs, each edge is anchored to the specific **claim sentence** in the citing paper that motivated it, and the four claim-driven relations are verified by **DeBERTa-v3-MNLI entailment** against in-paper context.

This repository accompanies an anonymous EMNLP 2026 submission (under review).

---

## Highlights

- 32,559 papers across NLP (41.3%), ML (28.7%), and Vision (30.0%), 2015–2024
- 573,126 typed edges across six relation types
- 90,020 NLI-verified claim seeds at 86–91% per-relation retention
- 287M typed length-≥3 trajectories covering 72.8% of papers
- **SciTraj-Pair benchmark**: AUC 0.914 ± 0.005, macro-F₁ 0.948 on six-way relation classification
- **Year-shuffle falsifiability protocol**: SciTraj-Pair drops 0.288 AUC under year permutation; content-only baselines unchanged
- 3-annotator pilot: Fleiss' κ = 0.74, 79.9% precision (520 items)

---

## Repository structure

```
scitraj/
├── data/                    # The corpus (~1.4 GB)
│   ├── corpus/              # 32,559 paper records + metadata
│   ├── graph/               # Typed graph, edges, topic assignments
│   ├── signals/             # Extracted claims + NLI-verified seeds
│   ├── embeddings/          # SPECTER2 abstract embeddings
│   └── splits/              # Temporal train/val/test IDs
|   |__ RAW/                 # RAW CORPUS
│
├── models/                  # Trained checkpoints (~200 MB)
│   ├── scitraj_pair/        # Main model + 48-dim features
│   └── gnn_baselines/       # T3–T8, T19 (App. K)
│
├── code/                    # Pipeline + evaluation
│   ├── corpus_construction/ # 4-stage build pipeline
│   ├── nli_verification/    # DeBERTa-v3-MNLI driver
│   ├── benchmark/           # 5-model ablation chain
│   ├── gnn_baselines/       # 7 typed-GNN architectures
│   ├── diagnostics/         # Year-shuffle + feature ablation
│   ├── findings/            # §6.1 siloing + §6.2 emergence
│   ├── downstream/          # Tasks A, B, C
│   ├── utils/               # Shared helpers
│   └── config/              # main.yaml
│
├── pilot_annotation/        # §8 human validation
│
├── results/                 # Reproducible numeric outputs
```

---

## Quick start

```bash
git clone <repo-url>
cd scitraj
python3.10 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Load the corpus:

```python
import torch, json, numpy as np

graph = torch.load("data/graph/graph_data.pt", weights_only=False)
edges = json.load(open("data/graph/typed_edges.json"))
embs  = np.load("data/embeddings/abstract_embs_norm.npy")

print(f"Papers: {graph['paper'].num_nodes}")        # 32,559
print(f"Edges:  {len(edges)}")                       # 573,126
print(f"Embeddings: {embs.shape}")                   # (32559, 768)
```

Reproduce the headline result:

```bash
python code/benchmark/scitraj_pair_multiseed.py \
    --graph data/graph/graph_data.pt \
    --features models/scitraj_pair/features.npz \
    --output results/benchmark/scitraj_pair_multiseed.json
```

---

## Pipeline at a glance

```
Stage 1: INPUT                Stage 2: CLAIM EXTRACTION
S2ORC, 2015–2024              Regex over section text
NLP/ML/Vision papers          → ~103K candidates
→ 32,559 papers

Stage 3: VERIFICATION         Stage 4: RELATION EXPANSION
NLI-verified (4 relations):   FAISS top-200 per source
DeBERTa-v3-MNLI, claim ⊢ ctx  Per-relation cosine ≥ 0.92
Similarity-only (2 relations) Year-gap gating
→ 90,020 NLI seeds            → 573,126 typed edges
```

---

## Relation schema

| Relation | Meaning | Verification |
|----------|---------|--------------|
| `causal_extension` | Result in *t* causally enables *s* | NLI entailment |
| `limit_addressed` | *s* addresses a limitation in *t* | NLI entailment |
| `future_realized` | *s* realises a future direction of *t* | NLI entailment |
| `dispute` | *s* disputes a claim in *t* | NLI entailment |
| `direct_extension` | *s* extends a method of *t* (Δy ≤ 2) | Abstract cosine + year gap |
| `temporal_semantic` | *s* updates *t* in a new period (Δy ≥ 5) | Abstract cosine + year gap |

---

## Hardware & environment

- **GPU**: NVIDIA A100 (40 GB)
- **Python**: 3.10
- **Key libs**: `torch`, `transformers`, `torch-geometric`, `lightgbm`, `faiss`, `sentence-transformers`

SciTraj-Pair trains in ~18 min/seed on an A100. Full 5-seed sweep + ablations finishes in under 4 hours.

---

## Licenses

- **Corpus** (`data/`): [CC BY 4.0](LICENSE-DATA)
- **Code** (`code/`): [MIT](LICENSE-CODE)
- **Pre-trained models**: Inherit upstream licenses — SPECTER2 (Apache 2.0), DeBERTa-v3 (MIT)

---



