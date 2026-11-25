"""Extract Midnight features for the organizer-curated TCGA-PRAD BCR cohort (HF dataset waticlems/tcga-prad-bcr)
with exactly the training tiling (container/owl_wsi.py). Only bcr.csv labels are used; metadata.csv (grading) is not.

Resumable: a case is done when <out>/feats/<case>.npz exists; each raw slide is deleted after its features save.
Writes <out>/labels.csv (case_id, event, follow_up_years) in the training-label format for tools/train_mil.py.

usage: python extract_tcga.py --out work\\tcga --weights work\\weights\\midnight

Encoding variants (a measurement of the frozen model): --level 0 forces the container's
resample path from level 0; --raw-resample drops that path's JPEG q90 round trip; --tile-jpeg Q adds a JPEG round trip
at quality Q to every tile (the tile set and white fractions stay those of the unmodified tiles).
"""
import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
import owl_wsi  # noqa: E402

BASE = "https://huggingface.co/datasets/waticlems/tcga-prad-bcr/resolve/main/"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fetch(rel, dest):
    if os.path.exists(dest):
        return dest
    for attempt in range(6):
        if subprocess.run(["curl", "-s", "-f", "-L", "-C", "-", "-o", dest + ".part", BASE + rel]).returncode == 0:
            os.replace(dest + ".part", dest)
            return dest
        time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"download failed: {rel}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--encoder", default="midnight", help="owl_wsi.ENCODERS name of the weights (e.g. hoptimus0)")
    ap.add_argument("--mpp", type=float, default=owl_wsi.TRAIN_MPP, help="tile spacing (owl_wsi.extract tile_mpp)")
    ap.add_argument("--max-tiles", type=int, default=0, help="keep the container's lowest-key cells (0 = all)")
    ap.add_argument("--feats", default="feats", help="feature and meta subdirectory name (meta goes to <feats>_meta)")
    ap.add_argument("--skip", help="file of case ids done on another host")
    ap.add_argument("--limit", type=int, default=0, help="cases per run (short GPU queue jobs)")
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th case from i (one process per GPU)")
    ap.add_argument("--level", type=int, default=None, help="force this pyramid level (0 = resample path)")
    ap.add_argument("--raw-resample", action="store_true", help="resample by area resize only (no JPEG q90 step)")
    ap.add_argument("--tile-jpeg", type=int, default=0, help="extra JPEG round trip at this quality on every tile")
    ap.add_argument("--keep-raw", action="store_true", help="keep each raw slide (another variant reads it next)")
    a = ap.parse_args()
    if a.raw_resample:
        owl_wsi.resample = lambda t: np.asarray(Image.fromarray(t).resize((owl_wsi.TILE, owl_wsi.TILE), Image.BOX))
    if a.tile_jpeg:
        tiles = owl_wsi.iter_tiles

        def jpeg_tiles(*args, **kw):
            for gx, gy, t, w in tiles(*args, **kw):
                b = io.BytesIO()
                Image.fromarray(t).save(b, "JPEG", quality=a.tile_jpeg)
                yield gx, gy, np.asarray(Image.open(b).convert("RGB")), w
        owl_wsi.iter_tiles = jpeg_tiles
    skip = set(open(a.skip).read().split()) if a.skip else set()
    fd, md = a.feats, "meta" if a.feats == "feats" else a.feats + "_meta"
    for d in ("raw", "masks", fd, md):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    rows = list(csv.DictReader(open(fetch("bcr.csv", os.path.join(a.out, "bcr.csv")))))
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case_id", "event", "follow_up_years"])
        w.writerows((r["case_id"], r["event"], r["follow_up_y"]) for r in rows)
    i, n = map(int, a.shard.split("/"))  # shard the fixed case list, so runs of several variants pick the same cases
    todo = [c for c in [r["case_id"] for r in rows if r["case_id"] not in skip][i::n]
            if not os.path.exists(os.path.join(a.out, fd, c + ".npz"))]
    todo = todo[:a.limit] if a.limit else todo
    log(f"cases {len(rows)}, to do {len(todo)}")

    def get(case):
        m = fetch(f"tissue-masks/{case}.tif", os.path.join(a.out, "masks", case + ".tif"))
        return fetch(f"whole-slide-images/{case}.tif", os.path.join(a.out, "raw", case + ".tif")), m

    enc = owl_wsi.Encoder(a.weights, name=a.encoder)
    pool, dl = ThreadPoolExecutor(12), ThreadPoolExecutor(2)
    futs = {c: dl.submit(get, c) for c in todo[:2]}
    failed = 0
    for i, case in enumerate(todo):
        if i + 2 < len(todo):
            futs[todo[i + 2]] = dl.submit(get, todo[i + 2])
        try:
            slide, mask = futs.pop(case).result()
            t0 = time.time()
            ex = owl_wsi.extract(slide, mask, enc, pool, log=lambda *x: None, max_tiles=a.max_tiles, tile_mpp=a.mpp,
                                 level=a.level)
            assert ex["meta"]["n"] > 0 and np.isfinite(ex["f"].astype(np.float32)).all(), "no tiles or bad features"
            np.savez(os.path.join(a.out, fd, case + ".tmp.npz"), f10=ex["f"], c10=ex["c"], w10=ex["w"])
            os.replace(os.path.join(a.out, fd, case + ".tmp.npz"), os.path.join(a.out, fd, case + ".npz"))
            meta = {**ex["meta"], "case": case, "seconds": round(time.time() - t0, 1)}
            json.dump(meta, open(os.path.join(a.out, md, case + ".json"), "w"))
            if not a.keep_raw:
                os.remove(slide)
            log(f"done {i + 1}/{len(todo)} {case} n={meta['n']} mpp={meta['mpp']:.3f} {meta['seconds']}s")
        except Exception as e:  # keep going; the raw slide stays for a retry
            failed += 1
            log(f"FAILED {case}: {type(e).__name__}: {e}")
    log(f"ALL DONE ({failed} failed)")
    raise SystemExit(3 if failed else 0)


if __name__ == "__main__":
    main()
