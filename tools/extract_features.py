"""Stream OWL (= LEOPARD) training slides from SURF, extract Midnight tile features, keep a thumbnail, delete the raw slide.

Resumable: a case is done when <out>/feats/<case>.npz exists. Raw slides live only in <out>/raw while being processed.
Per case it writes feats/<case>.npz with, for each tiling (native ~0.97 um/px level, 224 px tiles):
  f{tag}: (N, 3072) float16, Midnight [CLS, mean(patch tokens)]  c{tag}: (N, 2) int32 tile top-left (x, y) in level-0 px
  w{tag}: (N,) uint8, 255 x fraction of near-white pixels (min RGB > 220), so inference-time background filters can be matched
and thumbs/<case>.jpg (~15.5 um/px), lowres/<case>.jpg (~3.9 um/px, the mask spacing) plus meta/<case>.json (levels, um/px, tile counts, timings).

usage: python extract_features.py --out work --weights work\\weights\\midnight [--limit N]
"""
import argparse
import json
import os
import queue
import subprocess
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tifffile
import torch
import zarr
from PIL import Image

DAV = "https://surfdrive.surf.nl/public.php/webdav/"
SLIDE_SHARES = ["MZlTzMmbDYu2nZp", "951GvilginOZbCR"]
MASK_SHARE = "stHcZPHBB5VpNFN"
TILE = 224
TARGETS = {"10": 1.0}  # tag -> wanted um/px; nearest pyramid level used as-is (levels step 4x per side: 0.24, 0.97, 3.9 um/px)
BLOCK = 8  # tiles per side read as one region (fewer JPEG tile decodes)
MIN_TISSUE = 0.5


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def listing(token):
    out = subprocess.run(["curl", "-s", "-f", "-u", token + ":", "-X", "PROPFIND", "-H", "Depth: 1", DAV],
                         capture_output=True, check=True).stdout
    ns = {"d": "DAV:"}
    items = {}
    for r in ET.fromstring(out).findall("d:response", ns):
        name = urllib.parse.unquote(r.find("d:href", ns).text.rsplit("/", 1)[-1])
        size = r.find(".//d:getcontentlength", ns)
        if name and size is not None:
            items[name] = int(size.text)
    return items


def fetch(token, name, size, dest):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    part = dest + ".part"
    for attempt in range(8):
        subprocess.run(["curl", "-s", "-C", "-", "-u", token + ":", "-o", part, DAV + urllib.parse.quote(name)])
        if os.path.exists(part) and os.path.getsize(part) == size:
            os.replace(part, dest)
            return dest
        if os.path.exists(part) and os.path.getsize(part) > size:
            os.remove(part)
        time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"download failed: {name}")


def mpp_of(page):
    res, unit = page.tags.get("XResolution"), page.tags.get("ResolutionUnit")
    num, den = res.value
    per = {2: 25400.0, 3: 10000.0}[int(unit.value)]
    return per * den / num


def tissue_grid(mask_path, mpp_level, shape_level):
    """Fraction of tissue in each TILE x TILE cell of a level, from the binary mask (physical coordinates)."""
    with tifffile.TiffFile(mask_path) as mt:
        lv0 = mt.series[0].levels[0]
        m = (lv0.asarray() > 0).astype(np.float32)
        s = mpp_level / mpp_of(lv0.keyframe)  # level px -> mask px
        tile_um = TILE * mpp_level
    gh, gw = shape_level[0] // TILE, shape_level[1] // TILE
    hm, wm = int(round(gh * TILE * s)), int(round(gw * TILE * s))
    m = m[:hm, :wm]
    if m.shape != (hm, wm):  # mask smaller than the slide grid: pad with background
        m = np.pad(m, ((0, hm - m.shape[0]), (0, wm - m.shape[1])))
    frac = np.asarray(Image.fromarray(m).resize((gw, gh), Image.BOX))
    return frac, tile_um


def level_for(tf, want):
    levels = tf.series[0].levels
    mpps = [mpp_of(lv.keyframe) for lv in levels]
    i = int(np.argmin([abs(np.log(m / want)) for m in mpps]))
    return i, mpps


def iter_tiles(store_arr, frac, pool):
    """Yield (gx, gy, tile uint8 HxWx3, white fraction 0-255) for every grid cell with tissue >= MIN_TISSUE, reading BLOCKxBLOCK regions."""
    sel = frac >= MIN_TISSUE
    gh, gw = sel.shape
    blocks = [(by, bx) for by in range(0, gh, BLOCK) for bx in range(0, gw, BLOCK)
              if sel[by:by + BLOCK, bx:bx + BLOCK].any()]

    def read(b):
        by, bx = b
        reg = np.asarray(store_arr[by * TILE:(by + BLOCK) * TILE, bx * TILE:(bx + BLOCK) * TILE])
        out = []
        for ty, tx in zip(*np.nonzero(sel[by:by + BLOCK, bx:bx + BLOCK])):
            t = reg[ty * TILE:(ty + 1) * TILE, tx * TILE:(tx + 1) * TILE, :3]
            if t.shape[:2] == (TILE, TILE):
                out.append((bx + tx, by + ty, t, int(255 * (t.min(-1) > 220).mean())))
        return out

    for res in pool.map(read, blocks):
        yield from res


class Encoder:
    def __init__(self, weights, batch=96):
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(weights, torch_dtype=torch.float16).cuda().eval()
        self.batch = batch

    @torch.inference_mode()
    def __call__(self, tiles):
        x = torch.from_numpy(np.stack(tiles)).cuda().permute(0, 3, 1, 2).half().div_(255).sub_(0.5).div_(0.5)
        h = self.m(pixel_values=x).last_hidden_state
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], -1).float().cpu().numpy().astype(np.float16)


def process(case, slide, mask, enc, out, pool):
    t0 = time.time()
    meta = {"case": case, "tile": TILE, "min_tissue": MIN_TISSUE}
    res = {}
    with tifffile.TiffFile(slide) as tf:
        levels = tf.series[0].levels
        meta["levels"] = [[int(lv.shape[0]), int(lv.shape[1]), mpp_of(lv.keyframe)] for lv in levels]
        mpp0 = meta["levels"][0][2]
        for tag, want in TARGETS.items():
            li, mpps = level_for(tf, want)
            arr = zarr.open(tf.aszarr(level=li), mode="r")
            frac, _ = tissue_grid(mask, mpps[li], arr.shape)
            feats, coords, white, buf = [], [], [], []
            for gx, gy, t, w in iter_tiles(arr, frac, pool):
                buf.append(t)
                coords.append((gx, gy))
                white.append(w)
                if len(buf) == enc.batch:
                    feats.append(enc(buf)); buf = []
            if buf:
                feats.append(enc(buf))
            f = np.concatenate(feats) if feats else np.zeros((0, 3072), np.float16)
            scale = mpps[li] / mpp0
            c = (np.asarray(coords, np.float64).reshape(-1, 2) * TILE * scale).round().astype(np.int32)
            res["f" + tag], res["c" + tag], res["w" + tag] = f, c, np.asarray(white, np.uint8)
            meta[tag] = {"level": li, "mpp": mpps[li], "n": int(len(f)), "grid": list(frac.shape)}
            if len(f) == 0 or not np.isfinite(f.astype(np.float32)).all():
                raise RuntimeError(f"{case}: bad features at {tag} (n={len(f)})")
        for name, want, q in (("thumbs", 16.0, 90), ("lowres", 4.0, 85)):  # lowres = tissue-mask spacing, for maps/reports
            ti, mpps = level_for(tf, want)
            img = levels[ti].asarray()[..., :3]
            Image.fromarray(img).save(os.path.join(out, name, case + ".jpg"), quality=q)
            meta[name] = {"level": ti, "mpp": mpps[ti], "shape": list(img.shape[:2])}
    meta["seconds"] = round(time.time() - t0, 1)
    tmp = os.path.join(out, "feats", case + ".tmp.npz")
    np.savez(tmp, **res)
    os.replace(tmp, os.path.join(out, "feats", case + ".npz"))
    with open(os.path.join(out, "meta", case + ".json"), "w") as fh:
        json.dump(meta, fh)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--keep-raw", action="store_true")
    a = ap.parse_args()
    for d in ("raw", "masks", "feats", "thumbs", "lowres", "meta"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    masks = listing(MASK_SHARE)
    todo = []
    lists = [sorted(listing(t).items()) for t in SLIDE_SHARES]
    for i in range(max(map(len, lists))):  # interleave the two shares
        for tok, lst in zip(SLIDE_SHARES, lists):
            if i < len(lst) and lst[i][0].endswith(".tif"):
                todo.append((tok, *lst[i]))
    log(f"slides listed: {len(todo)}, masks listed: {len(masks)}")
    todo = [t for t in todo if not os.path.exists(os.path.join(a.out, "feats", t[1][:-4] + ".npz"))]
    if a.limit:
        todo = todo[:a.limit]
    log(f"to do: {len(todo)}")

    q = queue.Queue(maxsize=a.prefetch)

    def downloader():
        with ThreadPoolExecutor(a.prefetch) as dl:
            futs = []
            for tok, name, size in todo:
                case = name[:-4]
                mname = case + "_tissue.tif"
                mdest = os.path.join(a.out, "masks", mname)

                def job(tok=tok, name=name, size=size, mname=mname, mdest=mdest):
                    if not os.path.exists(mdest):
                        fetch(MASK_SHARE, mname, masks[mname], mdest)
                    return fetch(tok, name, size, os.path.join(a.out, "raw", name)), mdest
                futs.append((name[:-4], dl.submit(job)))
                while len([f for _, f in futs if not f.done()]) >= a.prefetch:
                    time.sleep(2)
                while futs and futs[0][1].done():
                    q.put(futs.pop(0))
            for item in futs:
                q.put(item)
        q.put(None)

    threading.Thread(target=downloader, daemon=True).start()
    enc = Encoder(a.weights)
    pool = ThreadPoolExecutor(12)
    done = failed = 0
    while (item := q.get()) is not None:
        case, fut = item
        try:
            slide, mask = fut.result()
            meta = process(case, slide, mask, enc, a.out, pool)
            done += 1
            log(f"done {done}/{len(todo)} {case} n10={meta['10']['n']} {meta['seconds']}s")
            if not a.keep_raw:
                os.remove(slide)
        except Exception as e:  # keep going; the raw slide stays for a retry
            failed += 1
            torch.cuda.empty_cache()
            log(f"FAILED {case}: {type(e).__name__}: {e}")
    log(f"ALL DONE ({failed} failed)")
    raise SystemExit(3 if failed else 0)


if __name__ == "__main__":
    main()
