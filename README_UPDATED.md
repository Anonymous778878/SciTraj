# EvoBench-ML Toolkit - Updated for New JSON Format

## ⚡ Quick Start - Process ALL Papers

The toolkit is configured to process **ALL papers** by default - no filtering!

```bash
# Simple one-command build (processes ALL 200 papers)
python run_all_papers.py
```

Or run manually:
```bash
python evobench_ml_toolkit_updated.py build \
  --seed /mnt/user-data/uploads/all_sample.json \
  --fulltext_file /mnt/user-data/uploads/imrad_corpus.json \
  --out_dir output/ \
  --store_sections
```

**No conference filtering required!** All papers in your dataset will be processed.

---

## Key Changes from Original Code

### 1. **Input Format Changes**

#### Original Format:
- **Seed file**: JSONL with fields like `paper_id`, `acl_id`, `conference`, `year`, `abstract`
- **Fulltext**: Directory with individual JSON files named `{paper_id}.json`

#### New Format:
- **Seed file** (`all_sample.json`): Single JSON array with query results
  ```json
  [
    {
      "query": "natural language processing",
      "data": [
        {
          "paperId": "28692beece311a90f5fa1ca2ec9d0c2ce293d069",
          "title": "...",
          "abstract": "...",
          "year": 2021,
          "venue": "ACM Computing Surveys",
          "citationCount": 4733,
          "referenceCount": 223
        }
      ]
    }
  ]
  ```

- **Fulltext file** (`imrad_corpus.json`): Single JSON array with IMRAD sections
  ```json
  [
    {
      "paperId": "2f5102ec3f70d0dea98c957cc2cab4d15d83a2da",
      "title": "...",
      "sections": {
        "Introduction": ["paragraph 1", "paragraph 2"],
        "Methods": ["..."],
        "Results": ["..."],
        "Discussion": ["..."],
        "Conclusion": ["..."]
      }
    }
  ]
  ```

### 2. **New Functions Added**

1. **`load_all_sample_json(path)`**: Loads papers from nested JSON structure
   - Extracts papers from `data` arrays within query results
   - Returns flat list of paper dictionaries

2. **`load_imrad_corpus_json(path)`**: Loads fulltext sections
   - Returns dict mapping `paperId` -> sections
   - Handles list-based section paragraphs

3. **`normalize_seed_record_new(rec)`**: Normalizes new paper format
   - Maps `paperId` -> `paper_id`
   - Maps `venue` -> `conference`
   - Handles missing fields gracefully

4. **`load_fulltext_sections_new(corpus, paper_id)`**: Retrieves sections
   - Joins list paragraphs with double newlines
   - Normalizes section keys to lowercase

### 3. **Modified Functions**

- **`cmd_build()`**: Updated to use new loading functions
  - Uses `load_all_sample_json()` instead of `read_jsonl()`
  - Uses `--fulltext_file` instead of `--fulltext_dir`
  - Added progress indicators for large datasets
  - More informative console output

## Usage Examples

### Basic Build - Process ALL Papers (RECOMMENDED)
```bash
# Processes all 200 papers in your dataset
python evobench_ml_toolkit_updated.py build \
  --seed /path/to/all_sample.json \
  --fulltext_file /path/to/imrad_corpus.json \
  --out_dir output/ \
  --store_sections
```

### Using the Simple Wrapper Script
```bash
# Even easier - just run this!
python run_all_papers.py
```

### Without Fulltext (Faster, Basic Features Only)
```bash
python evobench_ml_toolkit_updated.py build \
  --seed /path/to/all_sample.json \
  --out_dir output/
```

### Custom Parameters (Still Processes All Papers)
```bash
python evobench_ml_toolkit_updated.py build \
  --seed /path/to/all_sample.json \
  --fulltext_file /path/to/imrad_corpus.json \
  --out_dir output/ \
  --k_topics 25 \
  --top_k_edges 8 \
  --max_year_ahead 5 \
  --store_sections \
  --segment_gap_chars 400
```

### Validation
```bash
python evobench_ml_toolkit_updated.py validatepp \
  --data_dir output/
```

### Create Review Pack
```bash
python evobench_ml_toolkit_updated.py review_pack \
  --data_dir output/ \
  --out_dir review/ \
  --n_extractions 150 \
  --n_edges 150 \
  --strategy mixed
```

### LLM Judge (Optional)
```bash
python evobench_ml_toolkit_updated.py llm_judge \
  --data_dir output/ \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device auto
```

### Compute Stats
```bash
python evobench_ml_toolkit_updated.py stats \
  --data_dir output/ \
  --judge_file output/llm_judgments.jsonl \
  --human_csv review/review_extractions.csv
```

## Output Files (Same as Original)

```
output/
├── raw_ml_papers.jsonl              # Papers with extracted features
├── evobench_ml_units.jsonl          # Topic clusters
├── evobench_ml_events.jsonl         # Sentence-level events
├── evobench_ml_edges.jsonl          # Temporal relationships
├── evobench_ml_segments.jsonl       # Grounded text segments
├── evobench_ml_segment_events.jsonl # Segment-level events
├── validation_report_pp.json        # Validation metrics
└── splits/
    ├── train_papers.jsonl
    ├── train_events.jsonl
    ├── train_edges.jsonl
    ├── val_papers.jsonl
    ├── val_events.jsonl
    ├── val_edges.jsonl
    ├── test_papers.jsonl
    ├── test_events.jsonl
    └── test_edges.jsonl
```

## Data Structure Mapping

| Original Field | New Field | Notes |
|---------------|-----------|-------|
| `paper_id` / `acl_id` | `paperId` | Unique identifier |
| `conference` / `booktitle` | `venue` | Conference/journal name |
| `year` | `year` | Publication year |
| `title` | `title` | Paper title |
| `abstract` / `summary` | `abstract` | Paper abstract |
| `references` | N/A | Not in new format (yet) |

## Important Notes

1. **No References**: The new format doesn't include reference lists, so citation-based edges won't be created unless you add them to the data.

2. **Venue Filtering**: Use exact venue names as they appear in the JSON (e.g., "Annual Meeting of the Association for Computational Linguistics" not just "ACL").

3. **Section Names**: The IMRAD format uses capitalized section names (Introduction, Methods, Results, Discussion, Conclusion). The code normalizes these to lowercase internally.

4. **Missing Fulltext**: Papers without fulltext sections will have contributions extracted from abstracts only.

5. **Performance**: For large datasets (1000+ papers), the build process may take several minutes due to TF-IDF computation and clustering.

## Debugging Tips

### Check loaded papers:
```python
import json
with open('output/raw_ml_papers.jsonl', 'r') as f:
    papers = [json.loads(line) for line in f]
print(f"Total papers: {len(papers)}")
print(f"Sample: {papers[0].keys()}")
```

### Check if fulltext was loaded:
```python
papers_with_sections = sum(1 for p in papers if p.get('sections'))
print(f"Papers with fulltext: {papers_with_sections}/{len(papers)}")
```

### Inspect a specific paper:
```python
paper = papers[0]
print(f"Paper ID: {paper['paper_id']}")
print(f"Contributions: {len(paper.get('contributions', []))}")
print(f"Limitations: {len(paper.get('limitations', []))}")
print(f"Future work: {len(paper.get('future_work', []))}")
print(f"Method tags: {paper.get('method_tags', [])}")
```

## Differences in Behavior

1. **Progress Indicators**: New version shows progress during processing
2. **Error Handling**: Skips papers with errors instead of crashing
3. **Informative Output**: Better console messages showing what's happening
4. **Default Conference Filter**: Empty by default (processes all venues)

## Migration from Original Code

If you have scripts using the original code:

1. Change `--seed` to point to the new JSON file (all_sample.json)
2. Change `--fulltext_dir` to `--fulltext_file` (single JSON file)
3. Update any hardcoded field names in post-processing scripts
4. **Remove `--conference_filter`** - all papers are processed by default

The new version processes **ALL papers** automatically!
