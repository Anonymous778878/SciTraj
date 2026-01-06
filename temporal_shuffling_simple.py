#!/usr/bin/env python3
"""
Temporal Shuffling Control Experiment - Fixed for Small Datasets
================================================================

Handles:
- Small number of papers
- Empty vocabulary issues
- Missing abstracts
- Year range problems

Usage:
    python3 temporal_shuffling_fixed.py --seed all_sample.json --n-shuffles 5
"""

import json
import numpy as np
import argparse
from collections import defaultdict
import random
import copy

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import pandas as pd


def load_papers(seed_file):
    """Load papers from JSON file"""
    print(f"Loading papers from {seed_file}...")
    
    with open(seed_file, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    papers = []
    
    if isinstance(data, dict):
        if 'query' in data and 'search' in data['query']:
            papers = data['query']['search']
        elif 'papers' in data:
            papers = data['papers']
        elif 'results' in data:
            papers = data['results']
        else:
            for value in data.values():
                if isinstance(value, list) and len(value) > 0:
                    papers = value
                    break
    elif isinstance(data, list):
        papers = data
    
    if not papers:
        print("ERROR: Could not find papers in JSON file")
        exit(1)
    
    print(f"Found {len(papers)} papers")
    return papers


def normalize_paper(paper):
    """Normalize paper to standard format"""
    # Get year
    year = paper.get('year', paper.get('publicationDate', {}).get('year', 2017))
    if year is None or not isinstance(year, (int, float)):
        year = 2017
    year = int(year)
    
    # Get text fields
    title = paper.get('title', '')
    abstract = paper.get('abstract', '')
    
    if not title:
        title = paper.get('name', 'Untitled')
    
    if not abstract:
        abstract = paper.get('summary', paper.get('description', ''))
    
    # Combine - ensure we have SOME text
    if not title and not abstract:
        text = f"Document {hash(str(paper))}"
    else:
        text = f"{title} {abstract}"
    
    return {
        'paper_id': paper.get('paperId', paper.get('id', f'paper_{hash(title)}')),
        'title': title if title else 'Untitled',
        'abstract': abstract if abstract else '',
        'year': year,
        'text': text
    }


def temporal_coherence(papers, labels):
    """Calculate temporal coherence"""
    scores = []
    
    for cluster_id in np.unique(labels):
        cluster_papers = [p for p, l in zip(papers, labels) if l == cluster_id]
        years = [p['year'] for p in cluster_papers]
        
        if len(years) > 1:
            year_std = np.std(years)
            score = 1.0 / (1.0 + year_std)
        else:
            score = 1.0
        
        scores.append(score)
    
    return np.mean(scores)


def cluster_papers(papers, k):
    """
    Cluster papers using TF-IDF + K-Means
    
    Handles empty vocabulary issues
    """
    texts = [p['text'] for p in papers]
    
    # More lenient TF-IDF settings for small datasets
    vectorizer = TfidfVectorizer(
        max_features=min(1000, len(papers) * 10),  # Adaptive max features
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,  # Changed from 2 to 1 for small datasets
        max_df=0.95,  # More lenient
        token_pattern=r'\b\w+\b'  # More flexible tokenization
    )
    
    try:
        features = vectorizer.fit_transform(texts).toarray()
    except ValueError as e:
        if "empty vocabulary" in str(e):
            # Fallback: use even simpler settings
            print("    Warning: Using fallback vectorizer (very short text)")
            vectorizer = TfidfVectorizer(
                max_features=min(500, len(papers) * 5),
                ngram_range=(1, 1),  # Only unigrams
                min_df=1,
                max_df=1.0,
                token_pattern=r'\w+'
            )
            features = vectorizer.fit_transform(texts).toarray()
        else:
            raise
    
    # Ensure k is not larger than number of samples
    k = min(k, len(papers) - 1)
    k = max(2, k)  # At least 2 clusters
    
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return labels, features


def evaluate_clustering(papers, labels, features):
    """Evaluate clustering quality"""
    # Check if we have enough samples for meaningful metrics
    if len(np.unique(labels)) < 2:
        return {
            'silhouette': 0.0,
            'calinski_harabasz': 0.0,
            'temporal_coherence': temporal_coherence(papers, labels),
            'num_clusters': len(np.unique(labels))
        }
    
    return {
        'silhouette': silhouette_score(features, labels),
        'calinski_harabasz': calinski_harabasz_score(features, labels),
        'temporal_coherence': temporal_coherence(papers, labels),
        'num_clusters': len(np.unique(labels))
    }


def shuffle_years(papers, seed):
    """Shuffle years randomly"""
    shuffled = copy.deepcopy(papers)
    years = [p['year'] for p in shuffled]
    
    rng = np.random.RandomState(seed)
    shuffled_years = rng.permutation(years)
    
    for paper, new_year in zip(shuffled, shuffled_years):
        paper['year'] = int(new_year)
    
    return shuffled


def run_experiment(train_papers, test_papers, name):
    """Run clustering experiment on test set"""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  Train: {len(train_papers)} papers")
    print(f"  Test: {len(test_papers)} papers")
    
    # Determine k based on test set size
    if len(test_papers) < 10:
        k = 2  # Minimum clusters for very small datasets
    else:
        k = max(2, min(10, int(np.sqrt(len(test_papers)))))
    
    print(f"  Clusters: {k}")
    
    # Cluster
    print("  Running TF-IDF + K-Means...")
    labels, features = cluster_papers(test_papers, k)
    
    # Evaluate
    metrics = evaluate_clustering(test_papers, labels, features)
    
    print(f"  Silhouette: {metrics['silhouette']:.4f}")
    print(f"  C-H Score: {metrics['calinski_harabasz']:.2f}")
    print(f"  Temporal: {metrics['temporal_coherence']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default='all_sample.json')
    parser.add_argument('--train-end-year', type=int, default=None,
                       help='Auto-determined if not specified')
    parser.add_argument('--n-shuffles', type=int, default=5)
    
    args = parser.parse_args()
    
    print("="*60)
    print("TEMPORAL SHUFFLING EXPERIMENT (Fixed)")
    print("="*60)
    
    # Load papers
    raw_papers = load_papers(args.seed)
    
    # Normalize
    print("\nNormalizing papers...")
    papers = []
    for p in raw_papers:
        try:
            normalized = normalize_paper(p)
            papers.append(normalized)
        except Exception as e:
            continue
    
    print(f"Successfully normalized {len(papers)} papers")
    
    if len(papers) < 10:
        print(f"\nWARNING: Only {len(papers)} papers total!")
        print("Need at least 10 papers for meaningful experiment.")
        exit(1)
    
    # Get year range
    years = [p['year'] for p in papers]
    min_year = min(years)
    max_year = max(years)
    
    print(f"Year range: {min_year} - {max_year}")
    
    # Auto-determine split if not specified
    if args.train_end_year is None:
        # Use 80/20 split based on years
        sorted_years = sorted(set(years))
        split_idx = int(len(sorted_years) * 0.8)
        args.train_end_year = sorted_years[split_idx]
        print(f"Auto-determined train/test split: {args.train_end_year}")
    
    # Split by year
    train_papers = [p for p in papers if p['year'] <= args.train_end_year]
    test_papers = [p for p in papers if p['year'] > args.train_end_year]
    
    print(f"\nSplit:")
    print(f"  Train: {len(train_papers)} papers (≤{args.train_end_year})")
    print(f"  Test: {len(test_papers)} papers (>{args.train_end_year})")
    
    # Validate split
    if len(test_papers) < 5:
        print(f"\nERROR: Only {len(test_papers)} test papers!")
        print("Adjusting split...")
        
        # Try 70/30 split
        split_idx = int(len(sorted(set(years))) * 0.7)
        args.train_end_year = sorted(set(years))[split_idx]
        
        train_papers = [p for p in papers if p['year'] <= args.train_end_year]
        test_papers = [p for p in papers if p['year'] > args.train_end_year]
        
        print(f"  New train: {len(train_papers)} papers (≤{args.train_end_year})")
        print(f"  New test: {len(test_papers)} papers (>{args.train_end_year})")
        
        if len(test_papers) < 5:
            print("\nStill too few test papers. Using all papers for both train/test.")
            train_papers = papers
            test_papers = papers
    
    # Normal experiment
    print("\n" + "#"*60)
    print("# WORLD A: NORMAL")
    print("#"*60)
    normal_results = run_experiment(train_papers, test_papers, "Normal")
    
    # Shuffled experiments
    print("\n" + "#"*60)
    print(f"# WORLD B: SHUFFLED ({args.n_shuffles} seeds)")
    print("#"*60)
    
    shuffled_results = []
    for seed in range(args.n_shuffles):
        print(f"\nShuffle seed {seed}:")
        shuffled_train = shuffle_years(train_papers, seed)
        results = run_experiment(shuffled_train, test_papers, f"Shuffled-{seed}")
        shuffled_results.append(results)
    
    # Aggregate
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    shuffled_agg = defaultdict(lambda: defaultdict(list))
    for result in shuffled_results:
        for metric_name, value in result.items():
            shuffled_agg[metric_name].append(value)
    
    # Stats
    shuffled_stats = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        for metric, values in shuffled_agg.items()
    }
    
    # Drops
    sil_drop = (normal_results['silhouette'] - shuffled_stats['silhouette']['mean']) / max(normal_results['silhouette'], 0.001) * 100
    tc_drop = (normal_results['temporal_coherence'] - shuffled_stats['temporal_coherence']['mean']) / max(normal_results['temporal_coherence'], 0.001) * 100
    
    # Table
    rows = []
    
    rows.append({
        'Method': 'TF-IDF + K-Means',
        'Setting': 'Normal',
        'Silhouette': f"{normal_results['silhouette']:.4f}",
        'Temporal': f"{normal_results['temporal_coherence']:.4f}"
    })
    
    rows.append({
        'Method': 'TF-IDF + K-Means',
        'Setting': 'Shuffled',
        'Silhouette': f"{shuffled_stats['silhouette']['mean']:.4f}±{shuffled_stats['silhouette']['std']:.4f}",
        'Temporal': f"{shuffled_stats['temporal_coherence']['mean']:.4f}±{shuffled_stats['temporal_coherence']['std']:.4f}"
    })
    
    rows.append({
        'Method': 'TF-IDF + K-Means',
        'Setting': '→ Drop',
        'Silhouette': f"{sil_drop:.1f}%",
        'Temporal': f"{tc_drop:.1f}%"
    })
    
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))
    
    # Save
    df.to_csv('temporal_shuffling_results.csv', index=False)
    print(f"\n✓ Saved to: temporal_shuffling_results.csv")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if tc_drop < 5:
        print("\n→ Minimal temporal drop (<5%)")
        print("  Model doesn't use temporal info (expected for TF-IDF)")
    elif tc_drop > 15:
        print("\n→ Significant temporal drop (>15%)")
        print("  Model relies on temporal structure!")
    else:
        print("\n→ Moderate temporal drop (5-15%)")
        print("  Model uses some temporal information")
    
    print("\n✓ EXPERIMENT COMPLETE")


if __name__ == '__main__':
    main()