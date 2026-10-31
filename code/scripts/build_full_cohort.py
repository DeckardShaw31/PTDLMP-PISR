"""
Downloads (if needed) the full ALMRRC 2021 training set and splits it into per-route shards.

  python scripts/build_full_cohort.py            # uses code/data/s3_cache -> code/data/full_cohort
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.expand_amazon_cohort import S3_BASE, download_file_with_progress
from src.data.cohort import OFFICIAL_FILES, split_official_dataset

CODE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", default=os.path.join(CODE_DIR, "data", "s3_cache"))
    parser.add_argument("--cohort-dir", default=os.path.join(CODE_DIR, "data", "full_cohort"))
    args = parser.parse_args()

    os.makedirs(args.source_dir, exist_ok=True)
    for fname in OFFICIAL_FILES:
        path = os.path.join(args.source_dir, fname)
        if not os.path.exists(path):
            download_file_with_progress(S3_BASE + fname, path)
    split_official_dataset(args.source_dir, args.cohort_dir)
