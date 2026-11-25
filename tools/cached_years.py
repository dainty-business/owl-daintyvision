"""Predicted years of a packaged model for every case of a cached feature dir, under the container's tile cap
(lowest-key cells) and near-white drop: compares host-to-host predictions (e.g. with GC job outputs) without slides.

usage: python cached_years.py --feats feats --meta meta --model-dir pkg\\v4 --out runs\\years_v4_leopard.json
       [--find 7.721 6.312 9.584]   (print the cases whose years round to these values)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
import owl_wsi  # noqa: E402
from faithfulness_sim import Head  # noqa: E402
from inference import TILE_BUDGET  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("feats", "meta", "model-dir", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--max-tiles", type=int, default=TILE_BUDGET)
ap.add_argument("--find", type=float, nargs="*", default=[])
a = ap.parse_args()
head = Head(a.model_dir, "cuda")
years = {}
for p in sorted(glob.glob(os.path.join(a.feats, "*.npz"))):
    case = os.path.basename(p)[:-4]
    z, meta = np.load(p), json.load(open(os.path.join(a.meta, case + ".json")))
    if "mpp" not in meta:  # tools/extract_features.py layout (LEOPARD at 0.97 um/px)
        meta = {**meta["10"], "mpp0": meta["levels"][0][2]}
    f, w = z["f10"], z["w10"]
    if len(f) > a.max_tiles:  # owl_wsi.iter_tiles: the max_tiles lowest-key cells
        g = np.round(z["c10"] / (owl_wsi.TILE * meta["mpp"] / meta["mpp0"])).astype(int)
        key = np.random.default_rng(0).random(meta["grid"]).ravel()[g[:, 1] * meta["grid"][1] + g[:, 0]]
        sel = np.sort(np.argsort(key, kind="stable")[:a.max_tiles])
        f, w = f[sel], w[sel]
    keep = w <= owl_wsi.MAX_WHITE
    years[case] = head.years(head(f[keep] if keep.any() else f)[0])
json.dump(years, open(a.out, "w"), indent=1)
for v in a.find:
    near = sorted(years, key=lambda c: abs(years[c] - v))[:2]
    print(v, [(c, round(years[c], 6)) for c in near], flush=True)
