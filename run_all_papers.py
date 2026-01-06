#!/usr/bin/env python3
"""
Simple wrapper to run EvoBench-ML toolkit on ALL papers
No filtering, processes everything
"""

import subprocess
import sys
import os

def main():
    # File paths
    seed_file = "/mnt/user-data/uploads/all_sample.json"
    fulltext_file = "/mnt/user-data/uploads/imrad_corpus.json"
    output_dir = "output"
    
    # Check if files exist
    if not os.path.exists(seed_file):
        print(f"❌ Error: Seed file not found: {seed_file}")
        sys.exit(1)
    
    if not os.path.exists(fulltext_file):
        print(f"⚠️  Warning: Fulltext file not found: {fulltext_file}")
        print(f"   Will process papers with metadata only")
        fulltext_file = ""
    
    print("╔" + "="*58 + "╗")
    print("║" + " "*8 + "EVOBENCH-ML: PROCESS ALL PAPERS" + " "*17 + "║")
    print("╚" + "="*58 + "╝")
    print()
    print("Configuration:")
    print(f"  📄 Seed file: {seed_file}")
    print(f"  📚 Fulltext file: {fulltext_file if fulltext_file else 'None (basic features only)'}")
    print(f"  📁 Output directory: {output_dir}")
    print(f"  🎯 Processing: ALL PAPERS (no filtering)")
    print()
    
    # Build command
    cmd = [
        "python", "evobench_ml_toolkit_updated.py", "build",
        "--seed", seed_file,
        "--out_dir", output_dir,
    ]
    
    # Add fulltext if available
    if fulltext_file:
        cmd.extend(["--fulltext_file", fulltext_file])
        cmd.append("--store_sections")
    
    print("Running command:")
    print("  " + " \\\n    ".join(cmd))
    print()
    print("="*60)
    print()
    
    # Run the build
    try:
        result = subprocess.run(cmd, check=True)
        
        print()
        print("="*60)
        print("✅ SUCCESS! Dataset built for ALL papers")
        print("="*60)
        print()
        print("Next steps:")
        print("  1. Validate: python evobench_ml_toolkit_updated.py validatepp --data_dir output/")
        print("  2. Review: python evobench_ml_toolkit_updated.py review_pack --data_dir output/ --out_dir review/")
        print("  3. Check files: ls -lh output/*.jsonl")
        
    except subprocess.CalledProcessError as e:
        print()
        print("="*60)
        print("❌ Build failed!")
        print("="*60)
        print()
        print("Troubleshooting:")
        print("  1. Run: python diagnose_issues.py")
        print("  2. Check: TROUBLESHOOTING_GUIDE.md")
        sys.exit(1)

if __name__ == "__main__":
    main()
