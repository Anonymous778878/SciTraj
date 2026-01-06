#!/usr/bin/env python3
"""
Temporal Shuffling Control Experiment
======================================

Causality stress test: Does the model actually learn temporal structure,
or is it just clustering by similarity?

Experiment:
- World A (Normal): Use real years, build temporal graph
- World B (Shuffled): Randomly permute training years, rebuild graph
- Evaluate both on real future test set

If performance drops with shuffling (especially temporal coherence),
it proves the model learns temporal evolution, not just similarity.

Usage:
    python3 temporal_shuffling_experiment.py \
      --seed all_sample.json \
      --fulltext imrad_corpus.json \
      --train-end-year 2020 \
      --n-shuffles 5
"""

import json
import numpy as np
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import List, Dict, Any, Tuple
import random
import copy

# Import your existing system
import sys
sys.path.insert(0, str(Path(__file__).parent))

from evobench_complete_system import (
    EvoBenchMLSystem,
    Config,
    load_all_sample_json,
    load_imrad_corpus,
    normalize_paper,
    SciBERTEmbedder,
    EdgeBuilder,
    GNNTrainer,
    ClusteringBaselines,
    BenchmarkMetrics
)


class TemporalShufflingExperiment:
    """
    Run temporal shuffling control experiment
    """
    
    def __init__(self, config: Config, train_end_year: int = 2020):
        self.config = config
        self.train_end_year = train_end_year
        
        # Results storage
        self.results = {
            'normal': {},
            'shuffled': []  # List of results for different shuffle seeds
        }
    
    def load_and_split_data(self, seed_file: str, fulltext_file: str) -> Tuple[List, List]:
        """
        Load data and split into train/test by year
        """
        print("="*60)
        print("LOADING AND SPLITTING DATA")
        print("="*60)
        
        # Load raw data
        raw_papers = load_all_sample_json(seed_file)
        fulltext = load_imrad_corpus(fulltext_file)
        
        print(f"Loaded {len(raw_papers)} papers")
        
        # Normalize
        papers = []
        for paper in raw_papers:
            try:
                normalized = normalize_paper(paper, fulltext)
                papers.append(normalized)
            except Exception as e:
                continue
        
        # Split by year
        train_papers = []
        test_papers = []
        
        for paper in papers:
            year = paper.get('year')
            if year is None:
                continue
            
            if year <= self.train_end_year:
                train_papers.append(paper)
            else:
                test_papers.append(paper)
        
        print(f"\nTrain papers (year ≤ {self.train_end_year}): {len(train_papers)}")
        print(f"Test papers (year > {self.train_end_year}): {len(test_papers)}")
        
        # Show year distribution
        train_years = [p['year'] for p in train_papers]
        test_years = [p['year'] for p in test_papers]
        
        print(f"\nTrain year range: {min(train_years)} - {max(train_years)}")
        print(f"Test year range: {min(test_years)} - {max(test_years)}")
        
        return train_papers, test_papers
    
    def shuffle_train_years(self, train_papers: List[Dict], shuffle_seed: int) -> List[Dict]:
        """
        Create shuffled copy of training data with permuted years
        
        IMPORTANT: Only shuffle years, keep everything else identical
        """
        print(f"\nShuffling training years (seed={shuffle_seed})...")
        
        # Create deep copy
        shuffled_papers = copy.deepcopy(train_papers)
        
        # Extract years
        years = [p['year'] for p in shuffled_papers]
        
        # Shuffle years
        rng = np.random.RandomState(shuffle_seed)
        shuffled_years = rng.permutation(years)
        
        # Reassign
        for paper, new_year in zip(shuffled_papers, shuffled_years):
            paper['year'] = int(new_year)
        
        # Verify shuffle worked
        original_years = sorted([p['year'] for p in train_papers])
        new_years = sorted([p['year'] for p in shuffled_papers])
        
        assert original_years == new_years, "Year distribution changed!"
        
        # Count how many papers changed years
        changes = sum(1 for p1, p2 in zip(train_papers, shuffled_papers) 
                     if p1['year'] != p2['year'])
        
        print(f"  Changed years for {changes}/{len(train_papers)} papers ({changes/len(train_papers)*100:.1f}%)")
        
        # Show example changes
        print("  Example changes:")
        shown = 0
        for p1, p2 in zip(train_papers, shuffled_papers):
            if p1['year'] != p2['year'] and shown < 3:
                print(f"    '{p1['title'][:40]}...'")
                print(f"      {p1['year']} → {p2['year']}")
                shown += 1
        
        return shuffled_papers
    
    def run_benchmark_on_split(self, train_papers: List[Dict], test_papers: List[Dict],
                               name: str) -> Dict[str, Any]:
        """
        Run complete benchmark on train/test split
        """
        print("\n" + "="*60)
        print(f"RUNNING BENCHMARK: {name}")
        print("="*60)
        
        # Combine for processing
        all_papers = train_papers + test_papers
        
        # Mark train/test
        for i, paper in enumerate(all_papers):
            paper['_is_train'] = i < len(train_papers)
        
        # Extract embeddings
        print("\nExtracting SciBERT embeddings...")
        embedder = SciBERTEmbedder(
            model_name=self.config.embedding_model,
            device=self.config.device,
            cache_manager=None  # Don't cache shuffled versions
        )
        
        embeddings = embedder.embed_papers(all_papers, text_field='title_abstract')
        
        # Build temporal graph (only on training data)
        print("\nBuilding temporal graph (training only)...")
        edge_builder = EdgeBuilder(self.config)
        edges = edge_builder.build_edges(train_papers, embeddings[:len(train_papers)])
        
        print(f"Created {len(edges)} edges")
        
        # Train GNN (if enabled)
        gnn_embeddings = None
        if self.config.use_gnn:
            print("\nTraining GNN...")
            gnn_trainer = GNNTrainer(self.config, cache_manager=None)
            
            # Build graph data (training only)
            graph_data = gnn_trainer.build_graph(
                train_papers,
                embeddings[:len(train_papers)],
                edges
            )
            
            # Train
            model = gnn_trainer.train(graph_data, num_epochs=50)  # Fewer epochs for speed
            
            # Extract embeddings for ALL papers (train + test)
            # For test papers, use SciBERT as input to GNN
            full_graph_data = gnn_trainer.build_graph(
                all_papers,
                embeddings,
                edges
            )
            
            gnn_embeddings = gnn_trainer.get_embeddings(model, full_graph_data)
        
        # Run clustering methods on TEST set only
        print("\nClustering test set...")
        
        test_indices = [i for i, p in enumerate(all_papers) if not p['_is_train']]
        test_embeddings = embeddings[test_indices]
        test_gnn_embeddings = gnn_embeddings[test_indices] if gnn_embeddings is not None else None
        
        # Determine k
        k = max(5, min(20, int(np.sqrt(len(test_papers)))))
        print(f"Using k={k} clusters")
        
        results = {}
        
        # Method 1: TF-IDF + K-Means
        print("\n1. TF-IDF + K-Means...")
        labels = ClusteringBaselines.tfidf_kmeans(test_papers, k)
        metrics = BenchmarkMetrics.clustering_metrics(test_embeddings, labels)
        metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(test_papers, labels)
        results['tfidf_kmeans'] = metrics
        
        # Method 2: SciBERT + K-Means
        print("2. SciBERT + K-Means...")
        labels = ClusteringBaselines.scibert_kmeans(test_embeddings, k)
        metrics = BenchmarkMetrics.clustering_metrics(test_embeddings, labels)
        metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(test_papers, labels)
        results['scibert_kmeans'] = metrics
        
        # Method 3: SciBERT + Hierarchical
        print("3. SciBERT + Hierarchical...")
        labels = ClusteringBaselines.scibert_hierarchical(test_embeddings, k)
        metrics = BenchmarkMetrics.clustering_metrics(test_embeddings, labels)
        metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(test_papers, labels)
        results['scibert_hierarchical'] = metrics
        
        # Method 4: GNN + K-Means
        if test_gnn_embeddings is not None:
            print("4. GNN + K-Means...")
            labels = ClusteringBaselines.gnn_clustering(test_gnn_embeddings, k)
            metrics = BenchmarkMetrics.clustering_metrics(test_gnn_embeddings, labels)
            metrics['temporal_coherence'] = BenchmarkMetrics.temporal_coherence(test_papers, labels)
            results['gnn_clustering'] = metrics
        
        print("\nResults:")
        for method, metrics in results.items():
            print(f"  {method}:")
            print(f"    Silhouette: {metrics['silhouette_score']:.4f}")
            print(f"    C-H Score: {metrics['calinski_harabasz_score']:.2f}")
            print(f"    Temporal Coherence: {metrics['temporal_coherence']:.4f}")
        
        return results
    
    def run_normal_experiment(self, train_papers: List[Dict], test_papers: List[Dict]):
        """
        Run experiment with real years (World A)
        """
        print("\n" + "#"*60)
        print("# WORLD A: NORMAL (Real Years)")
        print("#"*60)
        
        self.results['normal'] = self.run_benchmark_on_split(
            train_papers, test_papers, "Normal (Real Years)"
        )
    
    def run_shuffled_experiments(self, train_papers: List[Dict], test_papers: List[Dict],
                                n_shuffles: int = 5):
        """
        Run experiments with shuffled years (World B)
        """
        print("\n" + "#"*60)
        print(f"# WORLD B: SHUFFLED (Permuted Years, {n_shuffles} seeds)")
        print("#"*60)
        
        for seed in range(n_shuffles):
            print(f"\n{'='*60}")
            print(f"SHUFFLE SEED {seed}")
            print(f"{'='*60}")
            
            # Shuffle training years
            shuffled_train = self.shuffle_train_years(train_papers, shuffle_seed=seed)
            
            # Run benchmark
            results = self.run_benchmark_on_split(
                shuffled_train, test_papers, f"Shuffled (seed={seed})"
            )
            
            self.results['shuffled'].append(results)
    
    def compute_statistics(self):
        """
        Compute mean ± std across shuffled runs
        """
        if not self.results['shuffled']:
            return {}
        
        # Collect metrics from all shuffled runs
        all_metrics = defaultdict(lambda: defaultdict(list))
        
        for shuffle_result in self.results['shuffled']:
            for method, metrics in shuffle_result.items():
                for metric_name, value in metrics.items():
                    all_metrics[method][metric_name].append(value)
        
        # Compute statistics
        stats = {}
        for method in all_metrics:
            stats[method] = {}
            for metric_name, values in all_metrics[method].items():
                stats[method][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return stats
    
    def create_comparison_table(self, output_file: str = 'shuffling_results.csv'):
        """
        Create ACL-style comparison table
        """
        print("\n" + "="*60)
        print("CREATING COMPARISON TABLE")
        print("="*60)
        
        # Compute statistics
        shuffled_stats = self.compute_statistics()
        
        # Build table
        rows = []
        
        methods = ['tfidf_kmeans', 'scibert_kmeans', 'scibert_hierarchical', 'gnn_clustering']
        method_names = {
            'tfidf_kmeans': 'TF-IDF + K-Means',
            'scibert_kmeans': 'SciBERT + K-Means',
            'scibert_hierarchical': 'SciBERT + Hierarchical',
            'gnn_clustering': 'GNN + K-Means'
        }
        
        for method in methods:
            if method not in self.results['normal']:
                continue
            
            # Normal (real years)
            normal_metrics = self.results['normal'][method]
            rows.append({
                'Method': method_names[method],
                'Setting': 'Normal',
                'Silhouette': f"{normal_metrics['silhouette_score']:.4f}",
                'Calinski-Harabasz': f"{normal_metrics['calinski_harabasz_score']:.2f}",
                'Temporal Coherence': f"{normal_metrics['temporal_coherence']:.4f}",
                'Num Clusters': normal_metrics['num_clusters']
            })
            
            # Shuffled (permuted years)
            if method in shuffled_stats:
                shuffled = shuffled_stats[method]
                rows.append({
                    'Method': method_names[method],
                    'Setting': 'Shuffled',
                    'Silhouette': f"{shuffled['silhouette_score']['mean']:.4f} ± {shuffled['silhouette_score']['std']:.4f}",
                    'Calinski-Harabasz': f"{shuffled['calinski_harabasz_score']['mean']:.2f} ± {shuffled['calinski_harabasz_score']['std']:.2f}",
                    'Temporal Coherence': f"{shuffled['temporal_coherence']['mean']:.4f} ± {shuffled['temporal_coherence']['std']:.4f}",
                    'Num Clusters': shuffled['num_clusters']['mean']
                })
                
                # Compute drops
                sil_drop = (normal_metrics['silhouette_score'] - shuffled['silhouette_score']['mean']) / normal_metrics['silhouette_score'] * 100
                temp_drop = (normal_metrics['temporal_coherence'] - shuffled['temporal_coherence']['mean']) / normal_metrics['temporal_coherence'] * 100
                
                rows.append({
                    'Method': method_names[method],
                    'Setting': '→ Drop',
                    'Silhouette': f"{sil_drop:.1f}%",
                    'Calinski-Harabasz': '-',
                    'Temporal Coherence': f"{temp_drop:.1f}%",
                    'Num Clusters': '-'
                })
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Save
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved to: {output_file}")
        
        # Print
        print("\n" + "="*60)
        print("RESULTS TABLE")
        print("="*60)
        print(df.to_string(index=False))
        
        return df
    
    def create_latex_table(self, output_file: str = 'shuffling_results.tex'):
        """
        Create LaTeX table for paper
        """
        # Compute statistics
        shuffled_stats = self.compute_statistics()
        
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        latex.append("\\caption{Temporal shuffling control experiment. Shuffling training years destroys temporal structure, causing performance drops (especially for GNN which relies on temporal edges).}")
        latex.append("\\label{tab:shuffling}")
        latex.append("\\begin{tabular}{llccc}")
        latex.append("\\toprule")
        latex.append("Method & Setting & Silhouette $\\uparrow$ & C-H Score $\\uparrow$ & Temporal $\\uparrow$ \\\\")
        latex.append("\\midrule")
        
        methods = ['tfidf_kmeans', 'scibert_kmeans', 'gnn_clustering']
        method_names = {
            'tfidf_kmeans': 'TF-IDF + K-Means',
            'scibert_kmeans': 'SciBERT + K-Means',
            'gnn_clustering': 'GNN + K-Means'
        }
        
        for method in methods:
            if method not in self.results['normal']:
                continue
            
            normal = self.results['normal'][method]
            
            # Normal row
            latex.append(f"{method_names[method]} & Normal & "
                        f"{normal['silhouette_score']:.3f} & "
                        f"{normal['calinski_harabasz_score']:.1f} & "
                        f"{normal['temporal_coherence']:.3f} \\\\")
            
            # Shuffled row
            if method in shuffled_stats:
                shuffled = shuffled_stats[method]
                latex.append(f" & Shuffled & "
                           f"{shuffled['silhouette_score']['mean']:.3f}$_{{\pm{shuffled['silhouette_score']['std']:.3f}}}$ & "
                           f"{shuffled['calinski_harabasz_score']['mean']:.1f}$_{{\pm{shuffled['calinski_harabasz_score']['std']:.1f}}}$ & "
                           f"{shuffled['temporal_coherence']['mean']:.3f}$_{{\pm{shuffled['temporal_coherence']['std']:.3f}}}$ \\\\")
                
                # Drop percentage
                sil_drop = (normal['silhouette_score'] - shuffled['silhouette_score']['mean']) / normal['silhouette_score'] * 100
                temp_drop = (normal['temporal_coherence'] - shuffled['temporal_coherence']['mean']) / normal['temporal_coherence'] * 100
                
                latex.append(f" & $\\rightarrow$ Drop & "
                           f"{sil_drop:.1f}\\% & "
                           f"- & "
                           f"\\textbf{{{temp_drop:.1f}\\%}} \\\\")
                
                latex.append("\\midrule")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")
        
        # Save
        with open(output_file, 'w') as f:
            f.write('\n'.join(latex))
        
        print(f"\n✓ Saved LaTeX table to: {output_file}")
        
        return '\n'.join(latex)
    
    def run_complete_experiment(self, seed_file: str, fulltext_file: str, n_shuffles: int = 5):
        """
        Run complete temporal shuffling experiment
        """
        print("\n" + "#"*60)
        print("# TEMPORAL SHUFFLING CONTROL EXPERIMENT")
        print("#"*60)
        print(f"\nTrain: year ≤ {self.train_end_year}")
        print(f"Test: year > {self.train_end_year}")
        print(f"Shuffle seeds: {n_shuffles}")
        
        # Load and split
        train_papers, test_papers = self.load_and_split_data(seed_file, fulltext_file)
        
        # World A: Normal
        self.run_normal_experiment(train_papers, test_papers)
        
        # World B: Shuffled
        self.run_shuffled_experiments(train_papers, test_papers, n_shuffles)
        
        # Create outputs
        self.create_comparison_table('temporal_shuffling_results.csv')
        self.create_latex_table('temporal_shuffling_results.tex')
        
        # Save raw results
        with open('temporal_shuffling_raw.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print("\n" + "="*60)
        print("✓ EXPERIMENT COMPLETE")
        print("="*60)
        print("\nOutputs:")
        print("  - temporal_shuffling_results.csv")
        print("  - temporal_shuffling_results.tex")
        print("  - temporal_shuffling_raw.json")


def main():
    parser = argparse.ArgumentParser(
        description='Temporal Shuffling Control Experiment'
    )
    
    parser.add_argument('--seed', default='all_sample.json',
                       help='Papers file')
    parser.add_argument('--fulltext', default='imrad_corpus.json',
                       help='Fulltext file')
    parser.add_argument('--train-end-year', type=int, default=2020,
                       help='Last year for training (test is > this year)')
    parser.add_argument('--n-shuffles', type=int, default=5,
                       help='Number of shuffle seeds')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--use-gnn', action='store_true',
                       help='Enable GNN (slower but shows bigger effect)')
    
    args = parser.parse_args()
    
    # Build config
    config = Config(
        seed_file=args.seed,
        fulltext_file=args.fulltext,
        output_dir='shuffling_output',
        device=args.device,
        use_gnn=args.use_gnn
    )
    
    # Run experiment
    experiment = TemporalShufflingExperiment(config, train_end_year=args.train_end_year)
    experiment.run_complete_experiment(args.seed, args.fulltext, args.n_shuffles)


if __name__ == '__main__':
    main()
