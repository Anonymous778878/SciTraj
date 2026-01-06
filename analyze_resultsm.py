#!/usr/bin/env python3
"""
Visualization and Analysis Tools for EvoBench-ML
================================================

Usage:
    python3 analyze_results.py --output output/

Features:
- Visualize embeddings
- Analyze clusters
- Plot temporal evolution
- Generate reports
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
import networkx as nx

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class ResultAnalyzer:
    """Analyze EvoBench-ML results"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        
        # Load data
        self.papers = self.load_json('papers_with_clusters.json')
        self.edges = self.load_json('temporal_edges.json')
        self.benchmarks = self.load_json('benchmark_results.json')
        
        # Load embeddings
        self.scibert_emb = self.load_npy('scibert_embeddings.npy')
        self.gnn_emb = self.load_npy('gnn_embeddings.npy')
        
        print(f"Loaded {len(self.papers)} papers")
        print(f"Loaded {len(self.edges)} edges")
    
    def load_json(self, filename: str) -> Any:
        path = self.output_dir / filename
        if not path.exists():
            print(f"Warning: {path} not found")
            return {}
        with open(path) as f:
            return json.load(f)
    
    def load_npy(self, filename: str) -> np.ndarray:
        path = self.output_dir / filename
        if not path.exists():
            print(f"Warning: {path} not found")
            return None
        return np.load(path)
    
    def plot_embeddings_2d(self, method: str = 'gnn', save: bool = True):
        if method == 'gnn' and self.gnn_emb is not None:
            emb = self.gnn_emb
            cluster_key = 'cluster_gnn_clustering'
        else:
            emb = self.scibert_emb
            cluster_key = 'cluster_scibert_kmeans'
        
        if emb is None:
            return
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb) - 1))
        coords = tsne.fit_transform(emb)
        
        clusters = [p.get(cluster_key, 0) for p in self.papers]
        years = [p['year'] for p in self.papers]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        sc1 = ax1.scatter(coords[:, 0], coords[:, 1], c=clusters, cmap='tab20', alpha=0.6, s=50)
        ax1.set_title(f'Research Landscape - Colored by Cluster ({method.upper()})')
        plt.colorbar(sc1, ax=ax1, label='Cluster ID')
        
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=years, cmap='viridis', alpha=0.6, s=50)
        ax2.set_title(f'Research Landscape - Colored by Year ({method.upper()})')
        plt.colorbar(sc2, ax=ax2, label='Year')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'embedding_visualization_{method}.png', dpi=300)
        plt.show()
    
    def plot_cluster_sizes(self, save: bool = True):
        methods = [k for k in self.papers[0].keys() if k.startswith('cluster_')]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, method_key in enumerate(methods[:4]):
            clusters = [p[method_key] for p in self.papers]
            counts = Counter(clusters)
            sizes = list(counts.values())
            
            ax = axes[idx]
            ax.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(sizes), color='r', linestyle='--')
            ax.set_title(method_key.replace('cluster_', ''))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'cluster_sizes.png', dpi=300)
        plt.show()
    
    def plot_temporal_distribution(self, save: bool = True):
        df = pd.DataFrame(self.papers)
        cluster_key = 'cluster_gnn_clustering' if 'cluster_gnn_clustering' in df.columns else 'cluster_scibert_kmeans'
        top_clusters = df[cluster_key].value_counts().head(10).index
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for cid in top_clusters:
            cdf = df[df[cluster_key] == cid]
            years = cdf['year'].value_counts().sort_index()
            ax.plot(years.index, years.values, marker='o', label=f'Cluster {cid}')
        
        ax.set_title('Temporal Evolution of Top Clusters')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Papers')
        ax.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'temporal_distribution.png', dpi=300)
        plt.show()
    
    def plot_benchmark_comparison(self, save: bool = True):
        if not self.benchmarks:
            return
        
        methods = list(self.benchmarks.keys())
        metrics = ['silhouette_score', 'calinski_harabasz_score', 'temporal_coherence']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(metrics):
            if metric == 'temporal_coherence':
                values = [self.benchmarks[m][metric] * 5 for m in methods]
            else:
                values = [self.benchmarks[m][metric] for m in methods]
            
            ax = axes[idx]
            bars = ax.bar(range(len(methods)), values, edgecolor='black', alpha=0.7)
            
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')
            
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.replace('_', '\n') for m in methods])
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(True, axis='y', alpha=0.3)
            
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'benchmark_comparison.png', dpi=300)
        plt.show()
    
    def generate_report(self):
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("EVOBENCH-ML ANALYSIS REPORT\n\n")
            f.write(f"Total papers: {len(self.papers)}\n")
            f.write(f"Total edges: {len(self.edges)}\n\n")
            
            if self.benchmarks:
                for method, metrics in self.benchmarks.items():
                    f.write(f"{method}\n")
                    for k, v in metrics.items():
                        f.write(f"  {k}: {v:.4f}\n")
                    f.write("\n")
        
        print(f"Report saved to {report_path}")
    
    def run_all_analyses(self):
        if self.gnn_emb is not None:
            self.plot_embeddings_2d('gnn')
        self.plot_embeddings_2d('scibert')
        
        self.plot_cluster_sizes()
        self.plot_temporal_distribution()
        
        if self.benchmarks:
            self.plot_benchmark_comparison()
        
        self.generate_report()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='output')
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer(args.output)
    analyzer.run_all_analyses()


if __name__ == '__main__':
    main()
