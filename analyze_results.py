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
from typing import Dict, List, Any

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
        """Load JSON file"""
        path = self.output_dir / filename
        if not path.exists():
            print(f"Warning: {path} not found")
            return [] if 'json' in filename else {}
        
        with open(path) as f:
            return json.load(f)
    
    def load_npy(self, filename: str) -> np.ndarray:
        """Load numpy file"""
        path = self.output_dir / filename
        if not path.exists():
            print(f"Warning: {path} not found")
            return None
        return np.load(path)
    
    def plot_embeddings_2d(self, method: str = 'gnn', save: bool = True):
        """Plot 2D projection of embeddings"""
        
        # Select embeddings
        if method == 'gnn' and self.gnn_emb is not None:
            emb = self.gnn_emb
            cluster_key = 'cluster_gnn_clustering'
        else:
            emb = self.scibert_emb
            cluster_key = 'cluster_scibert_kmeans'
        
        if emb is None:
            print(f"Embeddings not available for {method}")
            return
        
        print(f"Computing t-SNE for {method}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(emb)-1))
        coords = tsne.fit_transform(emb)
        
        # Get clusters
        clusters = [p.get(cluster_key, 0) for p in self.papers]
        years = [p['year'] for p in self.papers]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Color by cluster
        scatter1 = ax1.scatter(
            coords[:, 0], coords[:, 1],
            c=clusters, cmap='tab20',
            alpha=0.6, s=50
        )
        ax1.set_title(f'Research Landscape - Colored by Cluster ({method.upper()})', fontsize=16)
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
        plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
        
        # Plot 2: Color by year
        scatter2 = ax2.scatter(
            coords[:, 0], coords[:, 1],
            c=years, cmap='viridis',
            alpha=0.6, s=50
        )
        ax2.set_title(f'Research Landscape - Colored by Year ({method.upper()})', fontsize=16)
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=12)
        plt.colorbar(scatter2, ax=ax2, label='Year')
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / f'embedding_visualization_{method}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_cluster_sizes(self, save: bool = True):
        """Plot cluster size distribution"""
        
        methods = [k for k in self.papers[0].keys() if k.startswith('cluster_')]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, method_key in enumerate(methods[:4]):
            clusters = [p[method_key] for p in self.papers]
            cluster_counts = Counter(clusters)
            
            # Plot
            ax = axes[idx]
            sizes = list(cluster_counts.values())
            ax.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Cluster Size', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Cluster Size Distribution - {method_key.replace("cluster_", "")}', 
                        fontsize=14)
            ax.axvline(np.mean(sizes), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(sizes):.1f}')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'cluster_sizes.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_temporal_distribution(self, save: bool = True):
        """Plot temporal distribution of clusters"""
        
        df = pd.DataFrame(self.papers)
        
        # Use GNN clusters if available
        cluster_key = 'cluster_gnn_clustering' if 'cluster_gnn_clustering' in df.columns else 'cluster_scibert_kmeans'
        
        # Get top 10 clusters by size
        top_clusters = df[cluster_key].value_counts().head(10).index
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for cluster_id in top_clusters:
            cluster_df = df[df[cluster_key] == cluster_id]
            year_counts = cluster_df['year'].value_counts().sort_index()
            
            ax.plot(year_counts.index, year_counts.values, 
                   marker='o', label=f'Cluster {cluster_id}', linewidth=2)
        
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel('Number of Papers', fontsize=14)
        ax.set_title('Temporal Evolution of Top 10 Clusters', fontsize=16)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'temporal_distribution.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def plot_benchmark_comparison(self, save: bool = True):
        """Plot benchmark comparison"""
        
        if not self.benchmarks:
            print("No benchmark results available")
            return
        
        # Prepare data
        methods = list(self.benchmarks.keys())
        metrics = ['silhouette_score', 'calinski_harabasz_score', 'temporal_coherence']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, metric in enumerate(metrics):
            values = [self.benchmarks[m][metric] for m in methods]
            
            ax = axes[idx]
            bars = ax.bar(range(len(methods)), values, edgecolor='black', alpha=0.7)
            
            # Color best method
            best_idx = np.argmax(values)
            bars[best_idx].set_color('green')
            bars[best_idx].set_alpha(1.0)
            
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([m.replace('_', '\n') for m in methods], rotation=0)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.set_title(metric.replace('_', ' ').title(), fontsize=14)
            ax.grid(True, axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Benchmark Comparison (Green = Best)', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / 'benchmark_comparison.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def analyze_clusters(self, method: str = 'gnn_clustering'):
        """Detailed cluster analysis"""
        
        df = pd.DataFrame(self.papers)
        cluster_key = f'cluster_{method}'
        
        if cluster_key not in df.columns:
            print(f"Method {method} not found")
            return
        
        print(f"\n{'='*60}")
        print(f"CLUSTER ANALYSIS: {method}")
        print(f"{'='*60}\n")
        
        for cluster_id in sorted(df[cluster_key].unique()):
            cluster_df = df[df[cluster_key] == cluster_id]
            
            print(f"Cluster {cluster_id}:")
            print(f"  Size: {len(cluster_df)} papers")
            print(f"  Year range: {cluster_df['year'].min()} - {cluster_df['year'].max()}")
            print(f"  Venues: {cluster_df['venue'].value_counts().head(3).to_dict()}")
            print(f"  Sample titles:")
            for title in cluster_df['title'].head(3):
                print(f"    - {title[:70]}...")
            print()
    
    def generate_report(self):
        """Generate comprehensive report"""
        
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVOBENCH-ML ANALYSIS REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Dataset stats
            f.write("DATASET STATISTICS\n")
            f.write("-"*60 + "\n")
            f.write(f"Total papers: {len(self.papers)}\n")
            f.write(f"Total edges: {len(self.edges)}\n")
            
            df = pd.DataFrame(self.papers)
            f.write(f"Year range: {df['year'].min()} - {df['year'].max()}\n")
            f.write(f"Papers with fulltext: {df['has_fulltext'].sum()}\n")
            f.write(f"Unique venues: {df['venue'].nunique()}\n\n")
            
            # Benchmark results
            if self.benchmarks:
                f.write("BENCHMARK RESULTS\n")
                f.write("-"*60 + "\n")
                
                for method, metrics in self.benchmarks.items():
                    f.write(f"\n{method}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
            
            # Best method
            if self.benchmarks:
                f.write("\nBEST METHOD\n")
                f.write("-"*60 + "\n")
                
                sil_scores = {m: self.benchmarks[m]['silhouette_score'] 
                             for m in self.benchmarks}
                best_method = max(sil_scores, key=sil_scores.get)
                
                f.write(f"Winner: {best_method}\n")
                f.write(f"Silhouette score: {sil_scores[best_method]:.4f}\n")
        
        print(f"Report saved to {report_path}")
        
        # Print to console
        with open(report_path) as f:
            print(f.read())
    
    def run_all_analyses(self):
        """Run all analyses"""
        
        print("Running all analyses...\n")
        
        # Plots
        if self.gnn_emb is not None:
            self.plot_embeddings_2d('gnn')
        self.plot_embeddings_2d('scibert')
        
        self.plot_cluster_sizes()
        self.plot_temporal_distribution()
        
        if self.benchmarks:
            self.plot_benchmark_comparison()
        
        # Text analysis
        self.analyze_clusters('gnn_clustering' if self.gnn_emb is not None else 'scibert_kmeans')
        
        # Report
        self.generate_report()
        
        print("\n" + "="*60)
        print("ALL ANALYSES COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}/")
        print("  - embedding_visualization_*.png")
        print("  - cluster_sizes.png")
        print("  - temporal_distribution.png")
        print("  - benchmark_comparison.png")
        print("  - analysis_report.txt")


def main():
    parser = argparse.ArgumentParser(description="Analyze EvoBench-ML results")
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory with results')
    
    args = parser.parse_args()
    
    analyzer = ResultAnalyzer(args.output)
    analyzer.run_all_analyses()


if __name__ == '__main__':
    main()
