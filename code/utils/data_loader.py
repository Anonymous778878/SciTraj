"""
data_loader.py

Shared loader for SciTraj-V2 downstream task scripts. Handles the actual
file structure on your server:

  - data/graph/graph_data.pt           HeteroData (num_nodes=24909)
  - data/embeddings/abstract_embs.npy  (24879, 768) float32  ← "SPECTER2"
  - data/filtered/corpus.json          list of 24879 paper dicts
  - models/tier22_pair_mlp_expanded/best.pt
  - models/tier22_pair_mlp_expanded/features.npz   precomputed train/val/test pair features

Use this from the downstream scripts:

  from data_loader import load_all
  graph, edge_index, edge_type, corpus, abstract_embs, t22_features = load_all()

Where:
  graph: HeteroData
  edge_index: np.ndarray [2, E_total] flattened across all 9 edge types
  edge_type: np.ndarray [E_total] in {0..8}
  corpus: list of dicts (length 24879, paper_id matches list index)
  abstract_embs: np.ndarray [24879, 768] float32
  t22_features: dict from features.npz (X_train_pos, train_pos_s, etc.)
"""

import json
from pathlib import Path

import numpy as np
import torch


# Edge type names (consistent with your script 14)
EDGE_TYPE_NAMES = {
    0: "direct_extension", 1: "future_realized", 2: "limit_addressed",
    3: "causal_extension", 4: "performance_successor", 5: "method_reuse",
    6: "temporal_semantic", 7: "related_work", 8: "dispute"
}


def get_paths(project_root=None):
    """Returns dict of all paths used by the downstream scripts."""
    if project_root is None:
        # Walk up from the calling script's location to find scitraj_v2/
        project_root = Path(__file__).resolve().parent
        if project_root.name == "src":
            project_root = project_root.parent
    project_root = Path(project_root)
    return {
        "project_root": project_root,
        "graph": project_root / "data/graph/graph_data.pt",
        "corpus": project_root / "data/filtered/corpus.json",
        "abstract_embs": project_root / "data/embeddings/abstract_embs.npy",
        "t22_model": project_root / "models/tier22_pair_mlp_expanded/best.pt",
        "t22_features": project_root / "models/tier22_pair_mlp_expanded/features.npz",
        "outputs": project_root / "outputs/metrics",
    }


def flatten_hetero_edges(graph):
    """
    HeteroData stores edges per (src_type, rel, dst_type) triple. We want a
    single (edge_index [2, E], edge_type [E]) where edge_type indexes into
    {0..8} per the EDGE_TYPE_NAMES map.

    Returns:
        edge_index: np.ndarray [2, E_total]
        edge_type: np.ndarray [E_total]
        edge_type_id_map: dict {edge_type_name: numeric_id} actually used
    """
    # The graph keys included 'edge_index' and 'edge_type' at the top level
    # in your output. That suggests it may also have a flat representation
    # (older PyG style) or it may be HeteroData with flat fallback.
    # Try the simple route first: the graph may already expose flat arrays.

    # Method 1: top-level edge_index + edge_type (homogeneous fallback)
    if hasattr(graph, "edge_index") and graph.edge_index is not None:
        if hasattr(graph.edge_index, "shape") and graph.edge_index.dim() == 2:
            ei = graph.edge_index.cpu().numpy()
            if hasattr(graph, "edge_type") and graph.edge_type is not None:
                et = graph.edge_type.cpu().numpy()
                if ei.shape[1] == et.shape[0]:
                    print(f"  [data_loader] Using top-level flat edge layout: "
                          f"{ei.shape[1]} edges, {len(np.unique(et))} types")
                    return ei.astype(np.int64), et.astype(np.int64), {
                        EDGE_TYPE_NAMES[i]: i for i in np.unique(et)
                    }

    # Method 2: HeteroData with per-relation stores
    print("  [data_loader] Flattening HeteroData edges per relation...")
    parts_ei, parts_et = [], []
    name_to_id = {}
    next_id = 0

    if hasattr(graph, "edge_types"):
        for et_triple in graph.edge_types:
            store = graph[et_triple]
            if not hasattr(store, "edge_index") or store.edge_index is None:
                continue
            ei = store.edge_index.cpu().numpy()
            rel_name = et_triple[1] if isinstance(et_triple, tuple) else str(et_triple)
            # Map relation name to canonical numeric id
            if rel_name in [v for v in EDGE_TYPE_NAMES.values()]:
                tid = [k for k, v in EDGE_TYPE_NAMES.items() if v == rel_name][0]
            else:
                if rel_name not in name_to_id:
                    name_to_id[rel_name] = next_id
                    next_id += 1
                tid = name_to_id[rel_name]
            et_arr = np.full(ei.shape[1], tid, dtype=np.int64)
            parts_ei.append(ei)
            parts_et.append(et_arr)
            print(f"    relation {rel_name!r} → type_id={tid}, "
                  f"{ei.shape[1]} edges")

    if not parts_ei:
        raise RuntimeError(
            "Could not extract edges from graph_data.pt. Inspect manually."
        )

    edge_index = np.concatenate(parts_ei, axis=1).astype(np.int64)
    edge_type = np.concatenate(parts_et).astype(np.int64)
    print(f"  [data_loader] Flattened: {edge_index.shape[1]} total edges")
    return edge_index, edge_type, {EDGE_TYPE_NAMES.get(i, f"type_{i}"): i
                                   for i in np.unique(edge_type)}


def load_all(project_root=None, load_t22_features=True):
    """
    Loads everything needed by the downstream scripts.

    Returns dict with keys:
      graph, edge_index, edge_type, corpus, paper_id_to_idx, abstract_embs,
      t22_features (or None if load_t22_features=False), n_papers
    """
    paths = get_paths(project_root)

    # Validate
    for k in ["graph", "corpus", "abstract_embs", "t22_model"]:
        if not paths[k].exists():
            raise FileNotFoundError(f"Missing required file: {paths[k]}")

    print(f"[data_loader] Loading graph: {paths['graph']}")
    graph = torch.load(paths["graph"], map_location="cpu", weights_only=False)
    print(f"  type: {type(graph).__name__}")

    edge_index, edge_type, _ = flatten_hetero_edges(graph)

    print(f"[data_loader] Loading abstract embeddings: {paths['abstract_embs']}")
    abstract_embs = np.load(paths["abstract_embs"])
    print(f"  shape: {abstract_embs.shape}, dtype: {abstract_embs.dtype}")
    n_papers = abstract_embs.shape[0]

    print(f"[data_loader] Loading corpus: {paths['corpus']}")
    with open(paths["corpus"]) as f:
        corpus = json.load(f)
    print(f"  papers: {len(corpus)}")
    if len(corpus) != n_papers:
        print(f"  WARNING: corpus length {len(corpus)} != "
              f"abstract_embs {n_papers}. paper_id alignment may be off.")

    # Build paper_id → list-index map (in case paper_id != list index)
    paper_id_to_idx = {p["paper_id"]: i for i, p in enumerate(corpus)}

    # Filter edges: drop any that reference nodes >= n_papers
    # (HeteroData may include topic nodes etc.)
    mask = (edge_index[0] < n_papers) & (edge_index[1] < n_papers)
    if not mask.all():
        n_dropped = (~mask).sum()
        print(f"  Dropping {n_dropped} edges referencing non-paper nodes "
              f"(e.g. topic centroids)")
        edge_index = edge_index[:, mask]
        edge_type = edge_type[mask]

    # Load T22 pair features if requested
    t22_features = None
    if load_t22_features and paths["t22_features"].exists():
        print(f"[data_loader] Loading T22 pair features: {paths['t22_features']}")
        t22_features = dict(np.load(paths["t22_features"]))
        for k, v in t22_features.items():
            print(f"    {k}: shape={v.shape}")

    return {
        "graph": graph,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "corpus": corpus,
        "paper_id_to_idx": paper_id_to_idx,
        "abstract_embs": abstract_embs,
        "t22_features": t22_features,
        "n_papers": n_papers,
        "paths": paths,
    }


def load_t22_model(weights_path, device, in_dim=63):
    """Load T22 PairMLP architecture and weights."""
    sd = torch.load(weights_path, map_location=device, weights_only=False)
    if "model_state" in sd:
        sd = sd["model_state"]
    elif "state_dict" in sd:
        sd = sd["state_dict"]
    # Detect input dim from first linear layer
    first_w = None
    for k, v in sd.items():
        if k.endswith(".weight") and v.dim() == 2:
            first_w = v
            break
    if first_w is not None:
        detected_in_dim = first_w.shape[1]
        if detected_in_dim != in_dim:
            print(f"  T22 input dim: {detected_in_dim} (overriding {in_dim})")
            in_dim = detected_in_dim

    class PairMLP(torch.nn.Module):
        def __init__(self, in_dim, hidden=(128, 64)):
            super().__init__()
            layers = []
            prev = in_dim
            for h in hidden:
                layers += [
                    torch.nn.Linear(prev, h),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                ]
                prev = h
            layers += [torch.nn.Linear(prev, 1)]
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = PairMLP(in_dim).to(device)
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"  WARNING: strict load failed: {e}")
        print(f"  Trying non-strict load...")
        model.load_state_dict(sd, strict=False)
    model.eval()
    return model, in_dim
