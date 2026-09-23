import os
import sys
import json
import time
import urllib.request
from typing import Dict, List, Any, Optional

S3_BASE = "https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/almrrc2021-data-training/model_build_inputs/"

def download_file_with_progress(url: str, dest_path: str, chunk_size: int = 1024 * 1024):
    print(f"[Download] Starting download from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Antigravity/1.0"})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_size * 100) if total_size > 0 else 0
            speed = downloaded / (1024 * 1024 * max(0.001, time.time() - t0))
            print(f"\r[Download] {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({pct:.1f}%) @ {speed:.2f} MB/s", end="", flush=True)
    print(f"\n[Download] Finished in {time.time() - t0:.1f}s.")

if __name__ == "__main__":
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "s3_cache"))
    os.makedirs(cache_dir, exist_ok=True)
    route_data_path = os.path.join(cache_dir, "route_data.json")
    if not os.path.exists(route_data_path):
        download_file_with_progress(S3_BASE + "route_data.json", route_data_path)
    else:
        print(f"[Cache] route_data.json already exists at {route_data_path} ({os.path.getsize(route_data_path)/(1024*1024):.1f} MB).")
