# PROCESS ALL PAPERS - Final Configuration

## 🎯 What Changed

The toolkit is now configured to **process ALL papers by default** with NO filtering.

### Before (Original)
```bash
# Required conference filter
--conference_filter "ACL,NeurIPS,ICLR"
```

### After (Updated)
```bash
# NO conference filter needed
# All 200 papers are processed automatically
```

---

## ✅ Three Ways to Run

### Method 1: Simple Wrapper Script (EASIEST)
```bash
python run_all_papers.py
```

This automatically:
- Processes ALL 200 papers
- Uses fulltext when available
- Stores sections
- No configuration needed!

### Method 2: Direct Command (RECOMMENDED)
```bash
python evobench_ml_toolkit_updated.py build \
  --seed /mnt/user-data/uploads/all_sample.json \
  --fulltext_file /mnt/user-data/uploads/imrad_corpus.json \
  --out_dir output/ \
  --store_sections
```

### Method 3: Custom Parameters
```bash
python evobench_ml_toolkit_updated.py build \
  --seed /mnt/user-data/uploads/all_sample.json \
  --fulltext_file /mnt/user-data/uploads/imrad_corpus.json \
  --out_dir output/ \
  --k_topics 25 \
  --top_k_edges 8 \
  --store_sections
```

---

## 📊 What You'll Get

Processing **ALL 200 papers** from your dataset:

```
Papers processed: 200
  ├─ With fulltext: 51 papers (full features + segments)
  └─ Without fulltext: 149 papers (basic features)

Topics: ~20-25 clusters
Events: ~600-800 events
Edges: ~400-600 temporal relationships
Segments: ~100-200 text segments (from 51 papers with fulltext)

Splits:
  ├─ Train: ~140 papers (70%)
  ├─ Val: ~30 papers (15%)
  └─ Test: ~30 papers (15%)
```

---

## 🗂️ Output Files

```
output/
├── raw_ml_papers.jsonl              # All 200 papers
├── evobench_ml_events.jsonl         # ~600-800 events
├── evobench_ml_edges.jsonl          # ~400-600 edges
├── evobench_ml_segments.jsonl       # ~100-200 segments
├── evobench_ml_segment_events.jsonl # ~100-200 events
├── evobench_ml_units.jsonl          # ~20-25 topics
├── validation_report_pp.json        # Quality metrics
└── splits/
    ├── train_papers.jsonl           # ~140 papers
    ├── train_events.jsonl
    ├── train_edges.jsonl
    ├── val_papers.jsonl             # ~30 papers
    ├── val_events.jsonl
    ├── val_edges.jsonl
    ├── test_papers.jsonl            # ~30 papers
    ├── test_events.jsonl
    └── test_edges.jsonl
```

---

## 📋 Complete Workflow

### Step 1: Diagnose (Optional but Recommended)
```bash
python diagnose_issues.py
```

Expected output:
- ✅ 200 papers loaded
- ✅ 51 papers with fulltext
- ✅ All diagnostics pass

### Step 2: Build Dataset (ALL PAPERS)
```bash
python run_all_papers.py
```

Or:
```bash
python evobench_ml_toolkit_updated.py build \
  --seed /mnt/user-data/uploads/all_sample.json \
  --fulltext_file /mnt/user-data/uploads/imrad_corpus.json \
  --out_dir output/ \
  --store_sections
```

### Step 3: Validate Results
```bash
python evobench_ml_toolkit_updated.py validatepp \
  --data_dir output/
```

### Step 4: Create Review Pack (Optional)
```bash
python evobench_ml_toolkit_updated.py review_pack \
  --data_dir output/ \
  --out_dir review/ \
  --n_extractions 150 \
  --n_edges 150
```

### Step 5: Analyze Results
```python
import json

# Load all papers
papers = [json.loads(line) for line in open('output/raw_ml_papers.jsonl')]

print(f"Total papers: {len(papers)}")
print(f"Papers with fulltext: {sum(1 for p in papers if p.get('sections'))}")
print(f"Papers with limitations: {sum(1 for p in papers if p.get('limitations'))}")
print(f"Papers with future work: {sum(1 for p in papers if p.get('future_work'))}")

# Check topics
from collections import Counter
topics = Counter(p['topic_id'] for p in papers)
print(f"\nPapers per topic:")
for topic, count in topics.most_common(10):
    print(f"  {topic}: {count} papers")
```

---

## 🔍 Verification Checklist

After running, verify:

- [ ] `output/raw_ml_papers.jsonl` has 200 lines
- [ ] `output/evobench_ml_segments.jsonl` has >0 lines
- [ ] `output/evobench_ml_events.jsonl` has 400+ lines
- [ ] `output/evobench_ml_edges.jsonl` has 200+ lines
- [ ] `output/splits/` directory contains train/val/test files
- [ ] `output/validation_report_pp.json` exists
- [ ] Console shows "BUILD COMPLETE" message

Quick check:
```bash
# Count outputs
wc -l output/*.jsonl

# Expected:
#     200 raw_ml_papers.jsonl
# 600-800 evobench_ml_events.jsonl
# 400-600 evobench_ml_edges.jsonl
# 100-200 evobench_ml_segments.jsonl
# 100-200 evobench_ml_segment_events.jsonl
#   20-25 evobench_ml_units.jsonl
```

---

## 📈 Expected Performance

- **Time**: ~2-5 minutes for 200 papers
- **Memory**: ~500MB-1GB peak
- **CPU**: Single-threaded (uses 1 core)
- **Disk**: ~10-20MB output

---

## 🎓 Understanding Your Dataset

### Venue Distribution (Top 10)
```
arXiv.org: 11 papers
Neural Information Processing Systems: 5 papers
ACM Computing Surveys: 4 papers
Conference on Empirical Methods in NLP: 3 papers
IEEE Access: 3 papers
... (115 unique venues total)
```

### Fulltext Coverage
```
Papers with fulltext: 51 (25.8%)
Papers without fulltext: 149 (74.2%)
```

### Feature Coverage
```
Papers with:
  ✓ Method tags: ~200 (100%)
  ✓ Task tags: ~200 (100%)
  ✓ Dataset tags: ~200 (100%)
  ✓ Contributions: ~180 (90%)
  ✓ Metrics: ~51 (25.8% - fulltext only)
  ✓ Limitations: ~51 (25.8% - fulltext only)
  ✓ Future work: ~51 (25.8% - fulltext only)
  ✓ Segments: ~51 (25.8% - fulltext only)
```

---

## 💡 Key Points

1. ✅ **ALL 200 papers** are processed - no filtering
2. ✅ **No conference filter** needed or available
3. ✅ **51 papers** get full features (with fulltext)
4. ✅ **149 papers** get basic features (without fulltext)
5. ✅ **This is normal** and expected behavior
6. ✅ **Dataset is valid** with mixed coverage

---

## 🚨 Common Questions

**Q: Why do some papers have empty limitations?**  
A: Normal! Only 51/200 papers have fulltext. Limitations are extracted from Discussion sections.

**Q: Can I filter to specific conferences?**  
A: Not in this version. It's designed to process ALL papers. If you need filtering, modify the code or filter the output files.

**Q: Why are there only ~100 segments for 200 papers?**  
A: Segments require fulltext. Only 51 papers have fulltext, so only ~100-200 segments are created.

**Q: Should I get more fulltext?**  
A: Optional! The toolkit works fine with 25% coverage. More fulltext = richer features, but 200 papers is already a good dataset.

**Q: How do I process only certain years?**  
A: Build the full dataset, then filter the output files by year in post-processing.

---

## 📚 Next Steps

1. **Run the build** using one of the methods above
2. **Validate** the output with `validatepp`
3. **Explore** the generated files
4. **Analyze** your research evolution graph!

For help:
- See `TROUBLESHOOTING_GUIDE.md`
- See `README_UPDATED.md`
- See `QUICK_REFERENCE.md`
- Run `python diagnose_issues.py`

---

**Ready to start? Just run:**
```bash
python run_all_papers.py
```
