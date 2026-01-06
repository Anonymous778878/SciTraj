# SciTraj: Modeling Scientific Research Trajectories with Temporal Knowledge Graphs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A GNN-based system for modeling scientific research trajectories and tracking knowledge evolution through temporal knowledge graphs.**

---

## 📋 Overview

SciTraj constructs temporal knowledge graphs from scientific research papers, enabling researchers to:
- **Model research trajectories** over time (1999-2025)
- **Discover research evolution** through GNN-based clustering
- **Identify temporal relationships** between papers and ideas
- **Visualize knowledge propagation** across research communities

### Key Features

✅ **Temporal Knowledge Graph Construction** - SciBERT embeddings + GNN clustering  
✅ **Multi-Method Comparison** - TF-IDF, SciBERT, Hierarchical, GNN approaches  
✅ **Human Validation Framework** - Single-annotator validation with web interface  
✅ **Temporal Shuffling Controls** - Validates temporal structure learning  
✅ **Visualization Tools** - Network graphs, temporal flows, cluster evolution  


---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/scitraj.git
cd scitraj

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
```
torch>=2.0.0
torch-geometric>=2.3.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1
```

### Basic Usage

```bash
# Run complete pipeline on sample data
python3 evobench_complete_system.py \
  --input all_sample.json \
  --output ./output

# Or run on full dataset
python3 run_all_papers.py
```

---

## 📁 Repository Structure

```
scitraj/
├── README.md                              # This file
├── requirements.txt                        # Python dependencies
│
├── Core System
│   ├── evobench_complete_system.py        # Main pipeline
│   ├── evobench_ml_toolkit_updated.py     # Toolkit implementation
│   └── run_all_papers.py                  # Full dataset runner
│
├── Data
│   ├── papers_with_clusters.json          # Papers + cluster assignments
│   ├── temporal_edges.json                # Temporal knowledge graph edges
│   ├── imrad_corpus.json                  # Full corpus metadata
│   ├── all_sample.json                    # Small sample (testing)
│   └── all_sample1.json                   # Alternative sample
│
├── Validation
│   ├── validation_interface.html          # Web-based validation UI
│   ├── validation_data.json               # Validation pair samples
│   ├── validator_interface.py             # Validation generator
│   ├── analyze_validator.py               # Validation analysis
│   └── analyze_results.py                 # Results analysis
│
├── Experiments
│   ├── temporal_shuffling_experiment.py   # Control experiment
│   ├── temporal_shuffling_simple.py       # Lightweight version
│   ├── temporal_shuffling_results.csv     # Experiment results
│   └── temporal_shuffling_results.tex     # LaTeX table
│
├── Visualization
│   ├── visualize_temporal_graph.py        # Graph visualization
│   └── figures/                           # Generated figures
│
├── Documentation
│   ├── FILE_INDEX.md                      # File organization guide
│   ├── ALL_PAPERS_GUIDE.md                # Full dataset guide
│   ├── README_UPDATED.md                  # Additional documentation
│   └── FINAL_UPDATE_SUMMARY.md            # System updates
│
└── Utilities
    ├── fix_edges_properly.py              # JSON repair utility
    ├── cache/                              # Cached embeddings
    ├── output/                             # Generated outputs
    └── mnt/                                # Mounted data
```

---

## 🔬 System Components

### 1. Temporal Knowledge Graph Construction

**Input:** Research papers (title, abstract, year, venue)

**Pipeline:**
```
Papers → SciBERT Embeddings → GNN Clustering → Temporal Edges
```

**Core File:** `evobench_complete_system.py`

```python
from evobench_ml_toolkit_updated import SciTrajSystem

# Initialize system
system = SciTrajSystem(
    use_cache=True,
    cache_dir='./cache'
)

# Load papers
papers = system.load_papers('papers_with_clusters.json')

# Run full pipeline
results = system.run_pipeline(
    papers=papers,
    n_clusters=25,
    temporal_threshold=0.6
)
```

### 2. Clustering Methods

Compare 4 approaches:

| Method | File | Description |
|--------|------|-------------|
| TF-IDF + K-Means | `evobench_complete_system.py` | Baseline, bag-of-words |
| SciBERT + K-Means | `evobench_complete_system.py` | Semantic embeddings |
| SciBERT + Hierarchical | `evobench_complete_system.py` | Hierarchical clustering |
| **GNN + K-Means** | `evobench_complete_system.py` | Graph Neural Network ✓ |

### 3. Human Validation

**Generate Validation Interface:**

```bash
# Create validation pairs
python3 validator_interface.py \
  --papers papers_with_clusters.json \
  --edges temporal_edges.json \
  --n-cluster-pairs 50 \
  --n-edge-pairs 50

# Open validation_interface.html in browser
# Complete validation, download results

# Analyze results
python3 analyze_validator.py \
  --results validation_results.json \
  --data validation_data.json
```

**Metrics:**
- System Agreement (cluster: 78%, edge: 84%)
- Precision/Recall/F1
- Confidence analysis

### 4. Temporal Shuffling Control

Validates that GNN learns temporal structure:

```bash
python3 temporal_shuffling_experiment.py \
  --papers papers_with_clusters.json \
  --n-trials 5
```

## 📊 Datasets

### Sample Data

**`all_sample.json`** - 1,000 papers for testing
```bash
python3 evobench_complete_system.py --input all_sample.json
```

### Full Dataset

**`papers_with_clusters.json`** - 20,000+ papers (1899-2025)

```json
{
  "paper_id": "abc123",
  "title": "Attention Is All You Need",
  "abstract": "We propose a new architecture...",
  "year": 2017,
  "venue": "NeurIPS",
  "cluster_gnn_clustering": 5
}
```

### Temporal Edges

**`temporal_edges.json`** - 150,000+ temporal relationships

```json
{
  "src_paper_id": "abc123",
  "tgt_paper_id": "def456",
  "src_year": 2017,
  "tgt_year": 2018,
  "score": 0.85,
  "edge_type": "temporal_semantic"
}
```

---

## 🎯 Use Cases

### 1. Track Research Evolution

```python
# Find papers that influenced BERT
edges = load_temporal_edges('temporal_edges.json')
bert_influences = [e for e in edges if e['tgt_title'] == 'BERT']

# Visualize
visualize_evolution(bert_influences)
```

### 2. Discover Research Trends

```python
# Analyze cluster evolution over time
clusters = analyze_cluster_evolution(papers)
plot_cluster_timeline(clusters)
```

### 3. Identify Key Papers

```python
# Find highly connected papers
influential_papers = find_influential_papers(
    edges, 
    min_connections=10
)
```

### 4. Predict Future Directions

```python
# Use temporal patterns to predict trends
predictions = predict_research_directions(
    papers, 
    edges, 
    horizon=2
)
```

---

## 📈 Visualization

### Generate Temporal Graph

```bash
python3 visualize_temporal_graph.py \
  --papers papers_with_clusters.json \
  --edges temporal_edges.json \
  --output figures/temporal_graph.png
```

**Output Types:**
- Network graph with temporal layers
- Cluster evolution timeline
- Edge type distribution
- Influence flow diagram

---

## 🔧 Utilities

### Fix JSON Files

If your JSON files are corrupted:

```bash
python3 fix_edges_properly.py
```

### Analyze Results

```bash
python3 analyze_results.py \
  --clustering-results output/clustering_results.json \
  --temporal-edges output/temporal_edges.json
```

---

## 📝 Validation Protocol

### Annotator Validation

**Tasks:**
1. **Cluster Quality** - Do papers belong in same cluster?
2. **Temporal Edge** - Does later paper build on earlier paper?

**Metrics:**
- System Agreement
- Precision, Recall, F1
- Pearson Correlation

**Interface:**
- Web-based UI (`validation_interface.html`)
- Auto-saves progress
- Exports CSV/JSON

**Time:** ~60 minutes for 100 pairs

See `ALL_PAPERS_GUIDE.md` for detailed protocol.

---



GNN's larger drop confirms temporal learning.

---


---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **SciBERT** embeddings from AllenAI
- **PyTorch Geometric** for GNN implementation
- **Sentence Transformers** for embedding utilities
- Inspired by research evolution tracking in NLP/ML

---

## 📧 Contact

- **Author:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/yourusername)
- **Paper:** [arXiv link]

---



## ⭐ Star History

If you find this useful, please star the repository! ⭐

---

**Built with ❤️ for the research community**
