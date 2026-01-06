#!/usr/bin/env python3
"""
Temporal Knowledge Graph Visualization
=======================================

Creates publication-quality figures showing:
- Temporal structure (papers arranged by year)
- Directed edges (forward in time only)
- Edge types (different relationship types)
- Network layout optimized for temporal flow

Usage:
    python3 visualize_temporal_graph.py --output figures/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import networkx as nx
from collections import defaultdict
import argparse
from pathlib import Path

# Publication-quality settings
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['figure.dpi'] = 300


class TemporalGraphVisualizer:
    """
    Visualize temporal knowledge graphs for papers
    """
    
    def __init__(self, papers_file, edges_file):
        """Load data"""
        print("Loading data...")
        
        # Load papers
        with open(papers_file) as f:
            self.papers = json.load(f)
        
        # Load edges
        with open(edges_file) as f:
            self.edges = json.load(f)
        
        # Create paper index
        self.paper_dict = {p['paper_id']: p for p in self.papers}
        
        print(f"Loaded {len(self.papers)} papers, {len(self.edges)} edges")
        
        # Classify edge types
        self.classify_edge_types()
    
    def classify_edge_types(self):
        """
        Classify edges by relationship type based on metadata
        """
        print("\nClassifying edge types...")
        
        for edge in self.edges:
            src_paper = self.paper_dict.get(edge['src_paper_id'])
            tgt_paper = self.paper_dict.get(edge['tgt_paper_id'])
            
            if not src_paper or not tgt_paper:
                edge['edge_type'] = 'unknown'
                continue
            
            # Use similarity score and temporal gap to infer type
            score = edge.get('score', 0)
            year_gap = edge['tgt_year'] - edge['src_year']
            
            # Classification heuristics
            if score > 0.8 and year_gap <= 2:
                edge['relationship_type'] = 'direct_extension'
            elif score > 0.7 and year_gap <= 3:
                edge['relationship_type'] = 'builds_on'
            elif score > 0.5 and year_gap >= 3:
                edge['relationship_type'] = 'future_realized'
            elif 0.4 <= score <= 0.5:
                edge['relationship_type'] = 'related_work'
            else:
                edge['relationship_type'] = 'temporal_semantic'
        
        # Print distribution
        type_counts = defaultdict(int)
        for edge in self.edges:
            type_counts[edge.get('relationship_type', 'unknown')] += 1
        
        print("Edge type distribution:")
        for edge_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {edge_type}: {count}")
    
    def create_figure_2_simple(self, output_file='figure_2_simple.png'):
        """
        Figure 2 - Simple Temporal Graph Illustration
        
        Clean, schematic view showing:
        - Papers as nodes (year labeled)
        - Forward-time edges
        - Different edge types
        """
        print("\n" + "="*60)
        print("Creating Figure 2: Temporal Knowledge Graph Illustration")
        print("="*60)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define example papers for illustration
        example_papers = [
            {'id': 'A', 'year': 2008, 'title': 'Paper A:\nTransformer Architecture', 'x': 1, 'y': 3},
            {'id': 'B', 'year': 2009, 'title': 'Paper B:\nAttention Mechanisms', 'x': 1, 'y': 1},
            {'id': 'C', 'year': 2012, 'title': 'Paper C:\nPre-training Methods', 'x': 4, 'y': 3},
            {'id': 'D', 'year': 2014, 'title': 'Paper D:\nLanguage Models', 'x': 4, 'y': 1},
            {'id': 'E', 'year': 2018, 'title': 'Paper E:\nBERT', 'x': 7, 'y': 2},
        ]
        
        # Define edges
        example_edges = [
            {'src': 'A', 'tgt': 'C', 'type': 'future_realized', 'label': 'future_realized'},
            {'src': 'B', 'tgt': 'D', 'type': 'limitation_addressed', 'label': 'limitation_addressed'},
            {'src': 'C', 'tgt': 'E', 'type': 'builds_on', 'label': 'builds_on'},
            {'src': 'D', 'tgt': 'E', 'type': 'builds_on', 'label': 'builds_on'},
        ]
        
        # Color scheme
        node_color = '#E8F4F8'
        node_border = '#2E86AB'
        edge_colors = {
            'future_realized': '#E63946',      # Red
            'limitation_addressed': '#06A77D', # Green
            'builds_on': '#457B9D',            # Blue
        }
        
        # Draw nodes
        for paper in example_papers:
            # Node box
            box = FancyBboxPatch(
                (paper['x'] - 0.5, paper['y'] - 0.35),
                1.0, 0.7,
                boxstyle="round,pad=0.05",
                facecolor=node_color,
                edgecolor=node_border,
                linewidth=2,
                zorder=10
            )
            ax.add_patch(box)
            
            # Paper ID
            ax.text(paper['x'], paper['y'] + 0.15, f"Paper {paper['id']}",
                   ha='center', va='center', fontweight='bold', fontsize=11, zorder=11)
            
            # Year
            ax.text(paper['x'], paper['y'] - 0.15, f"({paper['year']})",
                   ha='center', va='center', fontsize=9, color='#555', zorder=11)
        
        # Draw edges
        for edge in example_edges:
            src = next(p for p in example_papers if p['id'] == edge['src'])
            tgt = next(p for p in example_papers if p['id'] == edge['tgt'])
            
            color = edge_colors.get(edge['type'], '#999')
            
            # Arrow
            arrow = FancyArrowPatch(
                (src['x'] + 0.5, src['y']),
                (tgt['x'] - 0.5, tgt['y']),
                arrowstyle='-|>',
                mutation_scale=20,
                linewidth=2,
                color=color,
                zorder=5,
                connectionstyle="arc3,rad=0.1"
            )
            ax.add_patch(arrow)
            
            # Edge label
            mid_x = (src['x'] + tgt['x']) / 2
            mid_y = (src['y'] + tgt['y']) / 2 + 0.3
            
            ax.text(mid_x, mid_y, edge['label'],
                   ha='center', va='center',
                   fontsize=8, style='italic',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                           edgecolor=color, linewidth=1),
                   zorder=12)
        
        # Add time axis
        ax.plot([0.5, 7.5], [0, 0], 'k-', linewidth=1.5, alpha=0.3, zorder=0)
        ax.text(4, -0.5, 'Time →', ha='center', fontsize=12, style='italic')
        
        # Year markers
        for year, x in [(2008, 1), (2012, 4), (2018, 7)]:
            ax.plot([x, x], [-0.1, 0.1], 'k-', linewidth=1.5, alpha=0.3, zorder=0)
            ax.text(x, -0.3, str(year), ha='center', fontsize=9, color='#555')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=edge_colors['future_realized'], 
                          label='Future Realized'),
            mpatches.Patch(facecolor=edge_colors['limitation_addressed'], 
                          label='Limitation Addressed'),
            mpatches.Patch(facecolor=edge_colors['builds_on'], 
                          label='Builds On'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 frameon=True, fancybox=True, shadow=True)
        
        # Title
        ax.set_title('Temporal Knowledge Graph: Forward-Time Research Evolution',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Clean up
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.7, 4)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def create_figure_2_full(self, output_file='figure_2_full.png', max_nodes=50):
        """
        Figure 2 - Full Temporal Graph
        
        Shows actual data from your dataset
        """
        print("\n" + "="*60)
        print("Creating Figure 2 (Full): Real Temporal Knowledge Graph")
        print("="*60)
        
        # Build NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for paper in self.papers[:max_nodes]:
            G.add_node(paper['paper_id'], 
                      year=paper['year'],
                      title=paper['title'][:30] + '...')
        
        # Add edges (only between nodes we added)
        node_ids = set(G.nodes())
        edge_count = 0
        
        for edge in self.edges:
            if edge['src_paper_id'] in node_ids and edge['tgt_paper_id'] in node_ids:
                G.add_edge(
                    edge['src_paper_id'],
                    edge['tgt_paper_id'],
                    relationship_type=edge.get('relationship_type', 'temporal_semantic'),
                    score=edge['score']
                )
                edge_count += 1
        
        print(f"Graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Layout: temporal ordering (year on x-axis)
        pos = {}
        year_groups = defaultdict(list)
        
        for node in G.nodes():
            year = G.nodes[node]['year']
            year_groups[year].append(node)
        
        # Assign positions
        years = sorted(year_groups.keys())
        x_positions = {year: i for i, year in enumerate(years)}
        
        for year in years:
            nodes_in_year = year_groups[year]
            n = len(nodes_in_year)
            
            for i, node in enumerate(nodes_in_year):
                x = x_positions[year]
                y = i - n / 2  # Center vertically
                pos[node] = (x, y)
        
        # Edge colors by type
        edge_colors_map = {
            'direct_extension': '#E63946',
            'builds_on': '#457B9D',
            'future_realized': '#F77F00',
            'related_work': '#06A77D',
            'temporal_semantic': '#999999',
        }
        
        # Draw edges by type
        for edge_type, color in edge_colors_map.items():
            edge_list = [
                (u, v) for u, v, d in G.edges(data=True)
                if d.get('relationship_type') == edge_type
            ]
            
            if edge_list:
                nx.draw_networkx_edges(
                    G, pos, edge_list,
                    edge_color=color,
                    width=1.5,
                    alpha=0.6,
                    arrows=True,
                    arrowsize=10,
                    arrowstyle='-|>',
                    connectionstyle='arc3,rad=0.1',
                    ax=ax
                )
        
        # Draw nodes
        node_colors = []
        for node in G.nodes():
            year = G.nodes[node]['year']
            # Color gradient by year
            year_normalized = (year - min(years)) / (max(years) - min(years))
            node_colors.append(plt.cm.viridis(year_normalized))
        
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=300,
            edgecolors='black',
            linewidths=1.5,
            ax=ax
        )
        
        # Add year labels on x-axis
        for year, x in x_positions.items():
            ax.text(x, min(pos.values(), key=lambda p: p[1])[1] - 1,
                   str(year),
                   ha='center', fontsize=10, fontweight='bold')
        
        # Time arrow
        ax.annotate('Time →', xy=(len(years)-1, min(pos.values(), key=lambda p: p[1])[1] - 2),
                   fontsize=12, style='italic', ha='right')
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=color, label=edge_type.replace('_', ' ').title())
            for edge_type, color in edge_colors_map.items()
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 frameon=True, fancybox=True, shadow=True, fontsize=9)
        
        # Title
        ax.set_title(f'Temporal Knowledge Graph of Research Evolution ({len(G.nodes())} papers)',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def create_figure_2_layered(self, output_file='figure_2_layered.png'):
        """
        Figure 2 - Layered Timeline View
        
        Papers arranged in temporal layers (one per year)
        Shows clear forward-time flow
        """
        print("\n" + "="*60)
        print("Creating Figure 2 (Layered): Timeline View")
        print("="*60)
        
        # Group papers by year
        year_groups = defaultdict(list)
        for paper in self.papers:
            year_groups[paper['year']].append(paper)
        
        years = sorted(year_groups.keys())
        
        # Sample papers (max 5 per year for clarity)
        sampled_papers = []
        for year in years:
            papers_in_year = year_groups[year]
            sample_size = min(5, len(papers_in_year))
            sampled_papers.extend(np.random.choice(papers_in_year, sample_size, replace=False))
        
        # Build graph
        G = nx.DiGraph()
        paper_ids = {p['paper_id'] for p in sampled_papers}
        
        for paper in sampled_papers:
            G.add_node(paper['paper_id'], **paper)
        
        for edge in self.edges:
            if edge['src_paper_id'] in paper_ids and edge['tgt_paper_id'] in paper_ids:
                G.add_edge(
                    edge['src_paper_id'],
                    edge['tgt_paper_id'],
                    **edge
                )
        
        print(f"Sampled graph: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Layout: strict temporal layers
        pos = {}
        year_to_layer = {year: i for i, year in enumerate(years)}
        
        for year in years:
            nodes_in_year = [n for n in G.nodes() if G.nodes[n]['year'] == year]
            n = len(nodes_in_year)
            layer_y = year_to_layer[year]
            
            for i, node in enumerate(nodes_in_year):
                x = i - n/2  # Spread horizontally
                y = layer_y
                pos[node] = (x, y)
        
        # Draw year separators
        for i, year in enumerate(years):
            ax.axhline(y=i, color='lightgray', linestyle='--', linewidth=1, alpha=0.5, zorder=0)
            ax.text(-8, i, str(year), fontsize=12, fontweight='bold', va='center')
        
        # Draw edges
        edge_type_colors = {
            'direct_extension': '#E63946',
            'builds_on': '#457B9D',
            'future_realized': '#F77F00',
            'related_work': '#06A77D',
            'temporal_semantic': '#999999',
        }
        
        for u, v, data in G.edges(data=True):
            edge_type = data.get('relationship_type', 'temporal_semantic')
            color = edge_type_colors.get(edge_type, '#999')
            
            arrow = FancyArrowPatch(
                pos[u], pos[v],
                arrowstyle='-|>',
                mutation_scale=15,
                linewidth=1.5,
                color=color,
                alpha=0.6,
                zorder=5,
                connectionstyle='arc3,rad=0.2'
            )
            ax.add_patch(arrow)
        
        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            
            # Node circle
            circle = Circle((x, y), 0.3, 
                          facecolor='#E8F4F8',
                          edgecolor='#2E86AB',
                          linewidth=2,
                          zorder=10)
            ax.add_patch(circle)
            
            # Node label (first 3 words of title)
            title = G.nodes[node]['title']
            short_title = ' '.join(title.split()[:3])
            ax.text(x, y, short_title,
                   ha='center', va='center',
                   fontsize=6, zorder=11)
        
        # Legend
        legend_elements = [
            mpatches.Patch(color=color, label=etype.replace('_', ' ').title())
            for etype, color in edge_type_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 frameon=True, fancybox=True, shadow=True)
        
        # Title
        ax.set_title('Temporal Knowledge Graph: Layered Timeline View',
                    fontsize=14, fontweight='bold', pad=20)
        
        # Time axis label
        ax.text(-8, len(years)/2, 'Time →',
               fontsize=12, style='italic', rotation=90,
               ha='center', va='center')
        
        ax.set_xlim(-9, max(pos.values(), key=lambda p: p[0])[0] + 2)
        ax.set_ylim(-0.5, len(years) - 0.5)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def create_edge_type_breakdown(self, output_file='figure_2_edge_types.png'):
        """
        Supplementary figure showing edge type statistics
        """
        print("\n" + "="*60)
        print("Creating Edge Type Breakdown")
        print("="*60)
        
        # Count edge types
        type_counts = defaultdict(int)
        for edge in self.edges:
            edge_type = edge.get('relationship_type', 'unknown')
            type_counts[edge_type] += 1
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pie chart
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        colors = ['#E63946', '#457B9D', '#F77F00', '#06A77D', '#999999'][:len(types)]
        
        ax1.pie(counts, labels=[t.replace('_', ' ').title() for t in types],
               colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Edge Type Distribution', fontweight='bold')
        
        # Bar chart
        ax2.barh(range(len(types)), counts, color=colors)
        ax2.set_yticks(range(len(types)))
        ax2.set_yticklabels([t.replace('_', ' ').title() for t in types])
        ax2.set_xlabel('Count')
        ax2.set_title('Edge Type Counts', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add counts on bars
        for i, count in enumerate(counts):
            ax2.text(count + 5, i, str(count), va='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def create_all_figures(self, output_dir='figures'):
        """Create all figure variants"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("CREATING ALL TEMPORAL GRAPH FIGURES")
        print("="*60)
        
        # Figure 2a: Simple schematic
        self.create_figure_2_simple(output_dir / 'figure_2_simple.png')
        
        # Figure 2b: Full network
        self.create_figure_2_full(output_dir / 'figure_2_full.png', max_nodes=50)
        
        # Figure 2c: Layered timeline
        self.create_figure_2_layered(output_dir / 'figure_2_layered.png')
        
        # Supplementary: Edge type breakdown
        self.create_edge_type_breakdown(output_dir / 'figure_2_edge_types.png')
        
        print("\n" + "="*60)
        print("✓ ALL FIGURES CREATED")
        print("="*60)
        print(f"\nOutput directory: {output_dir}/")
        print("Files:")
        print("  - figure_2_simple.png      (Schematic illustration)")
        print("  - figure_2_full.png        (Full network view)")
        print("  - figure_2_layered.png     (Timeline layers)")
        print("  - figure_2_edge_types.png  (Edge statistics)")


def main():
    parser = argparse.ArgumentParser(
        description='Create temporal knowledge graph visualizations'
    )
    parser.add_argument('--papers', default='output/papers_with_clusters.json',
                       help='Papers file')
    parser.add_argument('--edges', default='output/temporal_edges.json',
                       help='Edges file')
    parser.add_argument('--output-dir', default='figures',
                       help='Output directory')
    parser.add_argument('--figure', choices=['simple', 'full', 'layered', 'edge-types', 'all'],
                       default='all',
                       help='Which figure to create')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = TemporalGraphVisualizer(args.papers, args.edges)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create requested figures
    if args.figure == 'simple':
        viz.create_figure_2_simple(output_dir / 'figure_2_simple.png')
    elif args.figure == 'full':
        viz.create_figure_2_full(output_dir / 'figure_2_full.png')
    elif args.figure == 'layered':
        viz.create_figure_2_layered(output_dir / 'figure_2_layered.png')
    elif args.figure == 'edge-types':
        viz.create_edge_type_breakdown(output_dir / 'figure_2_edge_types.png')
    else:  # all
        viz.create_all_figures(output_dir)
    
    print(f"\n✅ Done! Figures saved to: {output_dir}/")


if __name__ == '__main__':
    main()
