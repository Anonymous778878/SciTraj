#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EvoBench-ML Advanced System with GNN-based Temporal Knowledge Graph
====================================================================

Features:
- SciBERT embeddings for scientific text
- Graph Neural Network (GNN) for paper representations
- Temporal Knowledge Graph construction
- Multiple baseline comparisons
- Comprehensive benchmarking
- Advanced validation metrics

Author: Enhanced EvoBench-ML System
Version: 2.0
"""

import os
import re
import json
import math
import hashlib
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

# NLP & Embeddings
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

# Graph Processing
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("torch_geometric not installed. GNN features will be limited.")

# Utilities
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import networkx as nx

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    """System configuration"""
    # Paths
    seed_file: str = "all_sample.json"
    fulltext_file: str = "imrad_corpus.json"
    output_dir: str = "output"
    cache_dir: str = "cache"
    
    # Model settings
    use_scibert: bool = True
    use_gnn: bool = True
    embedding_model: str = "allenai/scibert_scivocab_uncased"
    embedding_dim: int = 768
    gnn_hidden_dim: int = 256
    gnn_output_dim: int = 128
    
    # Graph settings
    temporal_window: int = 3  # years
    min_edge_weight: float = 0.3
    top_k_edges: int = 10
    
    # Processing
    batch_size: int = 32
    use_cache: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Benchmarking
    run_baselines: bool = True
    baseline_methods: List[str] = None
    
    def __post_init__(self):
        if self.baseline_methods is None:
            self.baseline_methods = [
                "tfidf_kmeans",
                "scibert_kmeans", 
                "scibert_hierarchical",
                "gnn_clustering"
            ]


# ============================================================
# Utilities
# ============================================================

class CacheManager:
    """Smart caching for expensive operations"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
    def get_cache_path(self, key: str, suffix: str = ".npy") -> Path:
        """Get cache file path"""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}{suffix}"
    
    def save(self, key: str, data: Any, suffix: str = ".npy"):
        """Save to cache"""
        path = self.get_cache_path(key, suffix)
        if suffix == ".npy":
            np.save(path, data)
        elif suffix == ".json":
            with open(path, 'w') as f:
                json.dump(data, f)
        elif suffix == ".pt":
            torch.save(data, path)
    
    def load(self, key: str, suffix: str = ".npy") -> Optional[Any]:
        """Load from cache"""
        path = self.get_cache_path(key, suffix)
        if not path.exists():
            return None
        
        try:
            if suffix == ".npy":
                return np.load(path, allow_pickle=True)
            elif suffix == ".json":
                with open(path, 'r') as f:
                    return json.load(f)
            elif suffix == ".pt":
                return torch.load(path)
        except Exception as e:
            print(f"Cache load error: {e}")
            return None
    
    def exists(self, key: str, suffix: str = ".npy") -> bool:
        """Check if cached"""
        return self.get_cache_path(key, suffix).exists()


def ensure_dir(path: str):
    """Create directory if not exists"""
    Path(path).mkdir(exist_ok=True, parents=True)


def normalize_text(text: str) -> str:
    """Normalize text"""
    text = (text or "").lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def stable_hash(text: str) -> str:
    """Generate stable hash"""
    return hashlib.md5(text.encode()).hexdigest()


# ============================================================
# Data Loading
# ============================================================

def load_all_sample_json(path: str) -> List[Dict[str, Any]]:
    """Load papers from all_sample.json"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    papers = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and 'data' in item:
                papers.extend(item.get('data', []))
    
    return papers


def load_imrad_corpus(path: str) -> Dict[str, Dict[str, Any]]:
    """Load fulltext from imrad_corpus.json"""
    if not os.path.exists(path):
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    corpus = {}
    if isinstance(data, list):
        for paper in data:
            paper_id = paper.get('paperId')
            sections = paper.get('sections', {})
            if paper_id and sections:
                # Join section paragraphs
                processed = {}
                for key, value in sections.items():
                    if isinstance(value, list):
                        processed[key.lower()] = "\n\n".join(str(v) for v in value if v)
                    elif isinstance(value, str):
                        processed[key.lower()] = value
                corpus[paper_id] = processed
    
    return corpus


def normalize_paper(paper: Dict[str, Any], fulltext: Dict[str, str]) -> Dict[str, Any]:
    """Normalize paper format"""
    paper_id = paper.get('paperId', '')
    
    # Get sections
    sections = fulltext.get(paper_id, {})
    
    # Build combined text
    abstract = paper.get('abstract', '')
    intro = sections.get('introduction', '')
    methods = sections.get('methods', '')
    results = sections.get('results', '')
    discussion = sections.get('discussion', '')
    conclusion = sections.get('conclusion', '')
    
    # Handle year - ensure it's never None
    year = paper.get('year')
    if year is None or not isinstance(year, int):
        year = 2020  # Default year for papers without year info
    
    return {
        'paper_id': paper_id,
        'title': paper.get('title', ''),
        'abstract': abstract,
        'year': year,
        'venue': paper.get('venue', ''),
        'authors': [a.get('name', '') for a in paper.get('authors', [])],
        'citation_count': paper.get('citationCount', 0),
        'reference_count': paper.get('referenceCount', 0),
        'fields_of_study': paper.get('fieldsOfStudy', []),
        'url': paper.get('url', ''),
        
        # Fulltext sections
        'sections': sections,
        'has_fulltext': bool(sections),
        
        # Combined text for embedding
        'full_text': f"{abstract}\n\n{intro}\n\n{methods}\n\n{results}\n\n{discussion}\n\n{conclusion}".strip(),
        'title_abstract': f"{paper.get('title', '')} {abstract}".strip(),
    }


# ============================================================
# SciBERT Embedding Extractor
# ============================================================

class SciBERTEmbedder:
    """Extract embeddings using SciBERT"""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased", 
                 device: str = "cpu", cache_manager: CacheManager = None):
        self.device = device
        self.cache = cache_manager
        
        print(f"Loading SciBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        print(f"SciBERT loaded on {device}")
    
    def embed_batch(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Embed batch of texts"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                batch = texts[i:i+32]
                
                # Tokenize
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**encoded)
                
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def embed_papers(self, papers: List[Dict[str, Any]], 
                     text_field: str = 'title_abstract') -> np.ndarray:
        """Embed all papers with caching"""
        
        # Try cache
        cache_key = f"scibert_embeddings_{text_field}_{len(papers)}"
        if self.cache and self.cache.exists(cache_key):
            print("Loading embeddings from cache...")
            return self.cache.load(cache_key)
        
        print(f"Computing SciBERT embeddings for {len(papers)} papers...")
        texts = [p.get(text_field, '') for p in papers]
        
        embeddings = self.embed_batch(texts)
        
        # Normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
        
        # Cache
        if self.cache:
            self.cache.save(cache_key, embeddings)
        
        print(f"Embeddings shape: {embeddings.shape}")
        return embeddings


# ============================================================
# Graph Neural Network
# ============================================================

class TemporalGNN(nn.Module):
    """
    Graph Neural Network for temporal knowledge graph
    Learns paper representations considering citation structure and temporal ordering
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, 
                 output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass"""
        
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Final layer (no activation)
        x = self.convs[-1](x, edge_index, edge_weight)
        
        return F.normalize(x, p=2, dim=1)


class GNNTrainer:
    """Train GNN on temporal knowledge graph"""
    
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.config = config
        self.cache = cache_manager
        self.device = config.device
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for GNN. Install with: pip install torch-geometric")
    
    def build_graph(self, papers: List[Dict[str, Any]], 
                    initial_embeddings: np.ndarray,
                    edges: List[Dict[str, Any]]) -> Data:
        """Build PyG graph data"""
        
        # Node features
        x = torch.FloatTensor(initial_embeddings)
        
        # Build edge index
        paper_to_idx = {p['paper_id']: i for i, p in enumerate(papers)}
        
        edge_list = []
        edge_weights = []
        
        for edge in edges:
            src = edge.get('src_paper_id')
            tgt = edge.get('tgt_paper_id')
            
            if src in paper_to_idx and tgt in paper_to_idx:
                src_idx = paper_to_idx[src]
                tgt_idx = paper_to_idx[tgt]
                
                edge_list.append([src_idx, tgt_idx])
                edge_weights.append(edge.get('score', 1.0))
        
        if not edge_list:
            # Create self-loops if no edges
            edge_list = [[i, i] for i in range(len(papers))]
            edge_weights = [1.0] * len(papers)
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_weight = torch.FloatTensor(edge_weights)
        
        # Node metadata
        years = torch.LongTensor([p['year'] for p in papers])
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight,
            years=years,
            num_nodes=len(papers)
        )
        
        return data
    
    def train(self, data: Data, num_epochs: int = 100) -> TemporalGNN:
        """Train GNN model"""
        
        # Check cache
        cache_key = f"gnn_model_{data.num_nodes}_{num_epochs}"
        if self.cache and self.cache.exists(cache_key, ".pt"):
            print("Loading trained GNN from cache...")
            model_state = self.cache.load(cache_key, ".pt")
            model = TemporalGNN(
                input_dim=data.x.shape[1],
                hidden_dim=self.config.gnn_hidden_dim,
                output_dim=self.config.gnn_output_dim
            ).to(self.device)
            model.load_state_dict(model_state)
            return model
        
        print(f"Training GNN for {num_epochs} epochs...")
        
        model = TemporalGNN(
            input_dim=data.x.shape[1],
            hidden_dim=self.config.gnn_hidden_dim,
            output_dim=self.config.gnn_output_dim
        ).to(self.device)
        
        optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        data = data.to(self.device)
        model.train()
        
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward
            embeddings = model(data.x, data.edge_index, data.edge_attr)
            
            # Self-supervised loss: link prediction
            loss = self.compute_loss(embeddings, data.edge_index, data.num_nodes)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Cache
        if self.cache:
            self.cache.save(cache_key, model.state_dict(), ".pt")
        
        return model
    
    def compute_loss(self, embeddings: torch.Tensor, edge_index: torch.Tensor, 
                     num_nodes: int) -> torch.Tensor:
        """Compute link prediction loss"""
        
        # Positive edges
        src = embeddings[edge_index[0]]
        dst = embeddings[edge_index[1]]
        pos_scores = (src * dst).sum(dim=1)
        
        # Negative sampling
        neg_dst_idx = torch.randint(0, num_nodes, (edge_index.shape[1],), device=embeddings.device)
        neg_dst = embeddings[neg_dst_idx]
        neg_scores = (src * neg_dst).sum(dim=1)
        
        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        
        return pos_loss + neg_loss
    
    def get_embeddings(self, model: TemporalGNN, data: Data) -> np.ndarray:
        """Extract final embeddings"""
        model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index, data.edge_attr)
        
        return embeddings.cpu().numpy()


# ============================================================
# Edge Construction
# ============================================================

class EdgeBuilder:
    """Build temporal edges between papers"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build_edges(self, papers: List[Dict[str, Any]], 
                   embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Build all edges"""
        
        print(f"Building temporal edges...")
        
        # Compute similarity matrix
        similarity = embeddings @ embeddings.T
        
        edges = []
        
        # Add temporal ordering constraint
        for i, paper_i in enumerate(tqdm(papers, desc="Building edges")):
            year_i = paper_i.get('year')
            
            # Skip if year is invalid
            if year_i is None or not isinstance(year_i, int):
                continue
            
            # Find candidates (future papers within window)
            candidates = []
            for j, paper_j in enumerate(papers):
                if i == j:
                    continue
                
                year_j = paper_j.get('year')
                
                # Skip if year is invalid
                if year_j is None or not isinstance(year_j, int):
                    continue
                
                # Temporal constraint
                if year_j < year_i or year_j > year_i + self.config.temporal_window:
                    continue
                
                sim = float(similarity[i, j])
                if sim >= self.config.min_edge_weight:
                    candidates.append((j, sim))
            
            # Sort by similarity and take top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:self.config.top_k_edges]
            
            # Create edges
            for j, score in candidates:
                paper_j = papers[j]
                
                edges.append({
                    'edge_id': f"{paper_i['paper_id']}->{paper_j['paper_id']}",
                    'src_paper_id': paper_i['paper_id'],
                    'tgt_paper_id': paper_j['paper_id'],
                    'src_year': year_i,
                    'tgt_year': paper_j['year'],
                    'edge_type': 'temporal_semantic',
                    'score': score,
                    'src_title': paper_i['title'],
                    'tgt_title': paper_j['title']
                })
        
        print(f"Created {len(edges)} temporal edges")
        return edges


# ============================================================
# Clustering Methods (Baselines)
# ============================================================

class ClusteringBaselines:
    """Multiple clustering baselines for comparison"""
    
    @staticmethod
    def tfidf_kmeans(papers: List[Dict[str, Any]], k: int = 20) -> np.ndarray:
        """Baseline 1: TF-IDF + K-Means"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        print("Running TF-IDF + K-Means...")
        texts = [p['title_abstract'] for p in papers]
        
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = vectorizer.fit_transform(texts).toarray()
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        return labels
    
    @staticmethod
    def scibert_kmeans(embeddings: np.ndarray, k: int = 20) -> np.ndarray:
        """Baseline 2: SciBERT + K-Means"""
        print("Running SciBERT + K-Means...")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        
        return labels
    
    @staticmethod
    def scibert_hierarchical(embeddings: np.ndarray, k: int = 20) -> np.ndarray:
        """Baseline 3: SciBERT + Hierarchical Clustering"""
        print("Running SciBERT + Hierarchical...")
        
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(embeddings)
        
        return labels
    
    @staticmethod
    def scibert_dbscan(embeddings: np.ndarray) -> np.ndarray:
        """Baseline 4: SciBERT + DBSCAN"""
        print("Running SciBERT + DBSCAN...")
        
        clustering = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        return labels
    
    @staticmethod
    def gnn_clustering(gnn_embeddings: np.ndarray, k: int = 20) -> np.ndarray:
        """Method 5: GNN + K-Means (Our method)"""
        print("Running GNN + K-Means...")
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(gnn_embeddings)
        
        return labels


# ============================================================
# Evaluation Metrics
# ============================================================

class BenchmarkMetrics:
    """Comprehensive evaluation metrics"""
    
    @staticmethod
    def clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Clustering quality metrics"""
        
        # Remove noise points for DBSCAN
        mask = labels >= 0
        if mask.sum() < 2:
            return {
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'num_clusters': 0,
                'noise_ratio': 1.0
            }
        
        emb_filtered = embeddings[mask]
        labels_filtered = labels[mask]
        
        if len(np.unique(labels_filtered)) < 2:
            return {
                'silhouette_score': -1.0,
                'calinski_harabasz_score': 0.0,
                'num_clusters': len(np.unique(labels_filtered)),
                'noise_ratio': 1.0 - mask.mean()
            }
        
        return {
            'silhouette_score': silhouette_score(emb_filtered, labels_filtered),
            'calinski_harabasz_score': calinski_harabasz_score(emb_filtered, labels_filtered),
            'num_clusters': len(np.unique(labels_filtered)),
            'noise_ratio': 1.0 - mask.mean()
        }
    
    @staticmethod
    def temporal_coherence(papers: List[Dict[str, Any]], labels: np.ndarray) -> float:
        """Measure temporal coherence within clusters"""
        
        coherence_scores = []
        
        for cluster_id in np.unique(labels):
            if cluster_id < 0:  # Skip noise
                continue
            
            cluster_papers = [p for p, l in zip(papers, labels) if l == cluster_id]
            if len(cluster_papers) < 2:
                continue
            
            years = [p['year'] for p in cluster_papers]
            year_std = np.std(years)
            
            # Lower std = better temporal coherence
            coherence_scores.append(1.0 / (1.0 + year_std))
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    @staticmethod
    def topic_diversity(papers: List[Dict[str, Any]], labels: np.ndarray) -> float:
        """Measure topic diversity within clusters"""
        
        diversity_scores = []
        
        for cluster_id in np.unique(labels):
            if cluster_id < 0:
                continue
            
            cluster_papers = [p for p, l in zip(papers, labels) if l == cluster_id]
            if len(cluster_papers) < 2:
                continue
            
            # Get all fields of study in cluster
            all_fields = []
            for p in cluster_papers:
                all_fields.extend(p.get('fields_of_study', []))
            
            unique_fields = len(set(all_fields))
            total_fields = len(all_fields)
            
            # Diversity = unique / total
            diversity_scores.append(unique_fields / max(1, total_fields))
        
        return np.mean(diversity_scores) if diversity_scores else 0.0


# ============================================================
# Main System
# ============================================================

class EvoBenchMLSystem:
    """Complete EvoBench-ML system with GNN and benchmarking"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cache = CacheManager(config.cache_dir)
        
        ensure_dir(config.output_dir)
        ensure_dir(config.cache_dir)
        
        # Components
        self.embedder = None
        self.gnn_trainer = None
        
        # Data
        self.papers = []
        self.edges = []
        self.scibert_embeddings = None
        self.gnn_embeddings = None
        
        # Results
        self.benchmark_results = {}
    
    def load_data(self):
        """Load and preprocess data"""
        print("="*60)
        print("LOADING DATA")
        print("="*60)
        
        # Load raw data
        raw_papers = load_all_sample_json(self.config.seed_file)
        fulltext = load_imrad_corpus(self.config.fulltext_file)
        
        print(f"Loaded {len(raw_papers)} papers")
        print(f"Loaded fulltext for {len(fulltext)} papers")
        
        # Normalize
        self.papers = []
        for paper in tqdm(raw_papers, desc="Normalizing papers"):
            try:
                normalized = normalize_paper(paper, fulltext)
                self.papers.append(normalized)
            except Exception as e:
                print(f"Error normalizing paper: {e}")
                continue
        
        print(f"Processed {len(self.papers)} papers")
        
        # Stats
        with_fulltext = sum(1 for p in self.papers if p['has_fulltext'])
        print(f"Papers with fulltext: {with_fulltext}/{len(self.papers)} ({with_fulltext/len(self.papers)*100:.1f}%)")
    
    def extract_embeddings(self):
        """Extract SciBERT embeddings"""
        print("\n" + "="*60)
        print("EXTRACTING SCIBERT EMBEDDINGS")
        print("="*60)
        
        self.embedder = SciBERTEmbedder(
            model_name=self.config.embedding_model,
            device=self.config.device,
            cache_manager=self.cache
        )
        
        self.scibert_embeddings = self.embedder.embed_papers(
            self.papers,
            text_field='full_text' if any(p['has_fulltext'] for p in self.papers) else 'title_abstract'
        )
    
    def build_temporal_graph(self):
        """Build temporal knowledge graph"""
        print("\n" + "="*60)
        print("BUILDING TEMPORAL KNOWLEDGE GRAPH")
        print("="*60)
        
        # Build edges
        edge_builder = EdgeBuilder(self.config)
        self.edges = edge_builder.build_edges(self.papers, self.scibert_embeddings)
        
        # Edge statistics
        edge_types = Counter([e['edge_type'] for e in self.edges])
        print(f"\nEdge statistics:")
        for etype, count in edge_types.items():
            print(f"  {etype}: {count}")
    
    def train_gnn(self):
        """Train GNN model"""
        if not self.config.use_gnn or not TORCH_GEOMETRIC_AVAILABLE:
            print("\nSkipping GNN training (disabled or not available)")
            return
        
        print("\n" + "="*60)
        print("TRAINING GRAPH NEURAL NETWORK")
        print("="*60)
        
        self.gnn_trainer = GNNTrainer(self.config, self.cache)
        
        # Build graph
        graph_data = self.gnn_trainer.build_graph(
            self.papers,
            self.scibert_embeddings,
            self.edges
        )
        
        print(f"Graph: {graph_data.num_nodes} nodes, {graph_data.edge_index.shape[1]} edges")
        
        # Train
        model = self.gnn_trainer.train(graph_data, num_epochs=100)
        
        # Extract embeddings
        self.gnn_embeddings = self.gnn_trainer.get_embeddings(model, graph_data)
        
        print(f"GNN embeddings shape: {self.gnn_embeddings.shape}")
    
    def run_benchmarks(self):
        """Run all baseline comparisons"""
        print("\n" + "="*60)
        print("RUNNING BENCHMARKS")
        print("="*60)
        
        # Determine k (number of clusters)
        k = max(5, min(30, int(np.sqrt(len(self.papers)))))
        print(f"Using k={k} clusters")
        
        results = {}
        
        # Baseline 1: TF-IDF + K-Means
        if 'tfidf_kmeans' in self.config.baseline_methods:
            labels = ClusteringBaselines.tfidf_kmeans(self.papers, k)
            metrics = BenchmarkMetrics.clustering_metrics(self.scibert_embeddings, labels)
            metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(self.papers, labels)
            results['tfidf_kmeans'] = {'labels': labels, 'metrics': metrics}
            print(f"\nTF-IDF + K-Means:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Baseline 2: SciBERT + K-Means
        if 'scibert_kmeans' in self.config.baseline_methods:
            labels = ClusteringBaselines.scibert_kmeans(self.scibert_embeddings, k)
            metrics = BenchmarkMetrics.clustering_metrics(self.scibert_embeddings, labels)
            metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(self.papers, labels)
            results['scibert_kmeans'] = {'labels': labels, 'metrics': metrics}
            print(f"\nSciBERT + K-Means:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Baseline 3: SciBERT + Hierarchical
        if 'scibert_hierarchical' in self.config.baseline_methods:
            labels = ClusteringBaselines.scibert_hierarchical(self.scibert_embeddings, k)
            metrics = BenchmarkMetrics.clustering_metrics(self.scibert_embeddings, labels)
            metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(self.papers, labels)
            results['scibert_hierarchical'] = {'labels': labels, 'metrics': metrics}
            print(f"\nSciBERT + Hierarchical:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Method 4: GNN + K-Means (our method)
        if self.gnn_embeddings is not None and 'gnn_clustering' in self.config.baseline_methods:
            labels = ClusteringBaselines.gnn_clustering(self.gnn_embeddings, k)
            metrics = BenchmarkMetrics.clustering_metrics(self.gnn_embeddings, labels)
            metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(self.papers, labels)
            results['gnn_clustering'] = {'labels': labels, 'metrics': metrics}
            print(f"\nGNN + K-Means (Ours):")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        self.benchmark_results = results
        
        return results
    
    def save_results(self):
        """Save all results"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        output_dir = Path(self.config.output_dir)
        
        # Save papers
        papers_output = []
        for i, paper in enumerate(self.papers):
            paper_out = {
                'paper_id': paper['paper_id'],
                'title': paper['title'],
                'year': paper['year'],
                'venue': paper['venue'],
                'abstract': paper['abstract'],
                'has_fulltext': paper['has_fulltext'],
                'citation_count': paper['citation_count']
            }
            
            # Add cluster assignments from each method
            for method_name, result in self.benchmark_results.items():
                labels = result['labels']
                paper_out[f'cluster_{method_name}'] = int(labels[i])
            
            papers_output.append(paper_out)
        
        with open(output_dir / 'papers_with_clusters.json', 'w') as f:
            json.dump(papers_output, f, indent=2)
        
        print(f"Saved papers to {output_dir / 'papers_with_clusters.json'}")
        
        # Save edges
        with open(output_dir / 'temporal_edges.json', 'w') as f:
            json.dump(self.edges, f, indent=2)
        
        print(f"Saved edges to {output_dir / 'temporal_edges.json'}")
        
        # Save embeddings
        if self.scibert_embeddings is not None:
            np.save(output_dir / 'scibert_embeddings.npy', self.scibert_embeddings)
            print(f"Saved SciBERT embeddings")
        
        if self.gnn_embeddings is not None:
            np.save(output_dir / 'gnn_embeddings.npy', self.gnn_embeddings)
            print(f"Saved GNN embeddings")
        
        # Save benchmark results
        benchmark_summary = {}
        for method_name, result in self.benchmark_results.items():
            benchmark_summary[method_name] = result['metrics']
        
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_summary, f, indent=2)
        
        print(f"Saved benchmark results")
        
        # Create comparison table
        self.create_comparison_table()
    
    def create_comparison_table(self):
        """Create comparison table of all methods"""
        
        if not self.benchmark_results:
            return
        
        # Build table
        methods = list(self.benchmark_results.keys())
        metrics = list(next(iter(self.benchmark_results.values()))['metrics'].keys())
        
        table_data = []
        for method in methods:
            row = {'Method': method}
            for metric in metrics:
                value = self.benchmark_results[method]['metrics'][metric]
                row[metric] = f"{value:.4f}"
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        output_path = Path(self.config.output_dir) / 'method_comparison.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\nMethod Comparison Table:")
        print(df.to_string(index=False))
        print(f"\nSaved to {output_path}")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("EVOBENCH-ML COMPLETE PIPELINE")
        print("="*60)
        print(f"Configuration:")
        print(f"  Seed file: {self.config.seed_file}")
        print(f"  Fulltext file: {self.config.fulltext_file}")
        print(f"  Output dir: {self.config.output_dir}")
        print(f"  Device: {self.config.device}")
        print(f"  Use GNN: {self.config.use_gnn}")
        print("="*60 + "\n")
        
        # Run pipeline
        self.load_data()
        self.extract_embeddings()
        self.build_temporal_graph()
        
        if self.config.use_gnn and TORCH_GEOMETRIC_AVAILABLE:
            self.train_gnn()
        
        self.run_benchmarks()
        self.save_results()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {self.config.output_dir}/")
        print(f"  - papers_with_clusters.json")
        print(f"  - temporal_edges.json")
        print(f"  - scibert_embeddings.npy")
        if self.gnn_embeddings is not None:
            print(f"  - gnn_embeddings.npy")
        print(f"  - benchmark_results.json")
        print(f"  - method_comparison.csv")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EvoBench-ML Advanced System")
    
    parser.add_argument('--seed', type=str, default='all_sample.json',
                       help='Path to seed file (all_sample.json)')
    parser.add_argument('--fulltext', type=str, default='imrad_corpus.json',
                       help='Path to fulltext file (imrad_corpus.json)')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--cache', type=str, default='cache',
                       help='Cache directory')
    
    parser.add_argument('--no-gnn', action='store_true',
                       help='Disable GNN training')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')
    
    parser.add_argument('--k-topics', type=int, default=0,
                       help='Number of clusters (0=auto)')
    parser.add_argument('--temporal-window', type=int, default=3,
                       help='Temporal window in years')
    
    args = parser.parse_args()
    
    # Build config
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = Config(
        seed_file=args.seed,
        fulltext_file=args.fulltext,
        output_dir=args.output,
        cache_dir=args.cache,
        use_gnn=not args.no_gnn,
        device=device,
        temporal_window=args.temporal_window
    )
    
    # Run system
    system = EvoBenchMLSystem(config)
    system.run_complete_pipeline()


if __name__ == '__main__':
    main()