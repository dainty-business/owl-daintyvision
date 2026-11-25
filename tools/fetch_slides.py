"""Re-download a fixed random sample of OWL training slides (SURF WebDAV) for exemplar-tile galleries.

usage: python fetch_slides.py <out_dir> <n> [seed]
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_features import SLIDE_SHARES, fetch, listing  # noqa: E402

out, n = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(out, exist_ok=True)
items = sorted((tok, name, size) for tok in SLIDE_SHARES for name, size in listing(tok).items() if name.endswith(".tif"))
for tok, name, size in random.Random(seed).sample(items, n):
    fetch(tok, name, size, os.path.join(out, name))
    print("fetched", name, size, flush=True)
print("ALL FETCHED", flush=True)
