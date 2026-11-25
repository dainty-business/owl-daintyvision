"""WSI tiling and Midnight tile encoding, shared by training-feature extraction (tools/extract_features.py) and the container.

Tiles: 224 px at the pyramid level nearest 1.0 um/px (OWL slides: 0.97 um/px), kept when the official tissue mask covers
>= 50% of the tile. Features: Kaiko Midnight [CLS, mean(patch tokens)], 3072-d.
"""
import time

import io

import numpy as np
import tifffile
import torch
import zarr
from PIL import Image

TILE = 224
WANT_MPP = 1.0
TRAIN_MPP = 0.97  # native level of the OWL training tiles; slides with no level within 15% of it are resampled
BLOCK = 8  # tiles per side read as one region (fewer JPEG tile decodes)
MIN_TISSUE = 0.5
MAX_WHITE = 230  # tiles with > 90% near-white pixels are dropped at training and inference (w array, 0-255)
FEAT_DIM = 3072


def mpp_of(page):
    res, unit = page.tags.get("XResolution"), page.tags.get("ResolutionUnit")
    num, den = res.value
    per = {2: 25400.0, 3: 10000.0}[int(unit.value)]
    return per * den / num


def tag_mpp(page, lo=0.0, hi=np.inf):
    """um/px from the page's resolution tags, or None if they are missing or outside [lo, hi]."""
    try:
        m = mpp_of(page)
    except Exception:
        return None
    return m if lo <= m <= hi else None


def level_mpps(tf, m0=None):
    """um/px per pyramid level. Level 0: its resolution tag if plausible (0.05-5 um/px), else m0 (e.g. from the mask),
    else 0.2425. A reduced level keeps its own tag only when it agrees within 2% with level 0 scaled by the width ratio:
    some writers repeat the level-0 tag on every page, which would make every level look full resolution."""
    levels = tf.series[0].levels
    t0 = tag_mpp(levels[0].keyframe, 0.05, 5.0)
    if t0 is not None:
        m0 = t0
    elif m0 is None:
        m0 = 0.2425  # OWL/LEOPARD scanner resolution; used only if the file has no usable resolution tag
        print("WARNING: slide has no usable resolution tag, assuming level 0 = 0.2425 um/px", flush=True)
    out = []
    for lv in levels:
        r = m0 * levels[0].shape[1] / lv.shape[1]
        t = tag_mpp(lv.keyframe)
        out.append(t if t is not None and abs(np.log(t / r)) <= np.log(1.02) else r)
    return out


def level_for(tf, want, m0=None):
    mpps = level_mpps(tf, m0)
    i = int(np.argmin([abs(np.log(m / want)) for m in mpps]))
    return i, mpps


def thumbnail(tf, want=16.0, max_side=8192):
    """RGB overview near `want` um/px. A file without a small enough level (e.g. one saved without reduced-page flags)
    is area-downsampled block by block instead of being loaded whole."""
    ti, _ = level_for(tf, want)
    lv = tf.series[0].levels[ti]
    if max(lv.shape[:2]) <= max_side:
        return lv.asarray()[..., :3]
    arr = zarr.open(tf.aszarr(level=ti), mode="r")
    k = -(-max(arr.shape[:2]) // 4096)  # downsample factor: longest side <= 4096
    h, w, b = arr.shape[0] // k * k, arr.shape[1] // k * k, k * 128
    out = np.zeros((h // k, w // k, 3), np.uint8)
    for y in range(0, h, b):
        for x in range(0, w, b):
            reg = np.asarray(arr[y:min(y + b, h), x:min(x + b, w)])[..., :3]
            out[y // k:(y + len(reg)) // k, x // k:(x + reg.shape[1]) // k] = np.asarray(
                Image.fromarray(np.ascontiguousarray(reg)).resize((reg.shape[1] // k, len(reg) // k), Image.BOX))
    return out


def tiling_for(tf, level=None, m0=None, mpp=TRAIN_MPP):
    """(level, mpps, src): read `level` in src x src px regions resized to TILE. The native path (src = TILE) is used
    when a level lies within 15% of `mpp` (the model's tile spacing); otherwise the coarsest finer level is resampled to
    `mpp`, so an unseen scanner or pyramid (e.g. 0.5 um/px with 4x steps) does not change the magnification the model
    was trained on."""
    li, mpps = level_for(tf, WANT_MPP * mpp / TRAIN_MPP, m0)
    if level is None and abs(np.log(mpps[li] / mpp)) <= np.log(1.15):
        return li, mpps, TILE
    if level is None:
        finer = [i for i, m in enumerate(mpps) if m < mpp]
        level = max(finer, key=lambda i: mpps[i]) if finer else li
    return level, mpps, max(1, round(TILE * mpp / mpps[level]))


def read_mask(mask_path):
    """Binary mask level 0 as float32 and its um/px (None if the file has no resolution tags)."""
    with tifffile.TiffFile(mask_path) as mt:
        lv0 = mt.series[0].levels[0]
        m = lv0.asarray()
        if m.ndim == 3:
            m = m[..., 0]
        return (m > 0).astype(np.float32), tag_mpp(lv0.keyframe, 0.05, 100.0)


def tissue_grid(mask, mpp_mask, mpp_level, shape_level):
    """Fraction of tissue in each TILE x TILE cell of a level, from the binary mask (physical coordinates)."""
    s = mpp_level / mpp_mask  # level px -> mask px
    gh, gw = shape_level[0] // TILE, shape_level[1] // TILE
    hm, wm = int(round(gh * TILE * s)), int(round(gw * TILE * s))
    m = mask[:hm, :wm]
    if m.shape != (hm, wm):  # mask smaller than the slide grid: pad with background
        m = np.pad(m, ((0, hm - m.shape[0]), (0, wm - m.shape[1])))
    return np.asarray(Image.fromarray(m).resize((gw, gh), Image.BOX))


def resample(t):
    """A src x src region as a TILE px tile like a scanner's own pyramid level: area resize, then a JPEG q90 round trip
    (pyramid levels are JPEG; without it Midnight features of the same area reach only cosine 0.79 with the native
    level, with it 0.94: tools/check_resample_filters.py)."""
    b = io.BytesIO()
    Image.fromarray(t).resize((TILE, TILE), Image.BOX if len(t) > TILE else Image.BILINEAR).save(b, "JPEG", quality=90)
    return np.asarray(Image.open(b).convert("RGB"))


def iter_tiles(store_arr, frac, pool, min_tissue=MIN_TISSUE, max_tiles=0, seed=0, shuffle=False, src=TILE):
    """Yield (gx, gy, tile uint8 HxWx3, white fraction 0-255) for every grid cell with tissue >= min_tissue, reading
    BLOCKxBLOCK regions. max_tiles > 0 keeps the eligible cells with the lowest fixed random key (one key per grid cell,
    inference time cap): a mask edit then only drops the edited cells and admits the next-ranked ones, instead of
    redrawing the subset (the faithfulness test re-runs the model on edited masks)."""
    sel = frac >= min_tissue
    key = np.random.default_rng(seed).random(sel.shape) if max_tiles or shuffle else None
    if max_tiles and sel.sum() > max_tiles:
        idx = np.flatnonzero(sel)
        keep = idx[np.argsort(key.flat[idx], kind="stable")[:max_tiles]]
        sel = np.zeros_like(sel)
        sel.flat[keep] = True
    gh, gw = sel.shape
    blocks = [(by, bx) for by in range(0, gh, BLOCK) for bx in range(0, gw, BLOCK)
              if sel[by:by + BLOCK, bx:bx + BLOCK].any()]
    if shuffle:  # fixed-key block order: a run stopped early still holds a random spatial sample
        blocks.sort(key=lambda b: key[b])

    def read(b):
        by, bx = b
        reg = np.asarray(store_arr[by * src:(by + BLOCK) * src, bx * src:(bx + BLOCK) * src])
        out = []
        for ty, tx in zip(*np.nonzero(sel[by:by + BLOCK, bx:bx + BLOCK])):
            t = reg[ty * src:(ty + 1) * src, tx * src:(tx + 1) * src, :3].copy()  # don't pin the region
            if t.shape[:2] == (src, src):
                if src != TILE:
                    t = resample(t)
                out.append((bx + tx, by + ty, t, int(255 * (t.min(-1) > 220).mean())))
        return out

    ahead = 64  # bounded read-ahead: decoded tiles waiting for the GPU stay below ~0.6 GB
    for i in range(0, len(blocks), ahead):
        for res in pool.map(read, blocks[i:i + ahead]):
            yield from res


IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# name -> (normalization mean/std, pooling); weights are a local HF snapshot dir. Midnight is the deployed default.
# "cls_mean" = [CLS, mean of patch tokens], skipping CLS and any register tokens.
ENCODERS = {
    "midnight": (((0.5,) * 3, (0.5,) * 3), "cls_mean"),  # kaiko-ai/midnight, 3072-d
    "phikon2": (IMAGENET, "cls"),  # owkin/phikon-v2, 1024-d
    "virchow2": (IMAGENET, "cls_mean"),  # paige-ai/Virchow2 (timm), 2560-d
    "hoptimus0": (((0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)), "cls"),  # bioptimus/H-optimus-0 (timm), 1536-d
}


def _timm_extra(name):
    """Constructor arguments the model cards pass explicitly (not resolvable from config.json)."""
    import timm
    return {"virchow2": dict(mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU),
            "hoptimus0": dict(img_size=224, init_values=1e-5, dynamic_img_size=False)}[name]


def _timm_model(weights, name):
    import json
    import os
    import timm
    cfg = json.load(open(os.path.join(weights, "config.json")))
    m = timm.create_model(cfg["architecture"], pretrained=False, **{**cfg.get("model_args", {}), "num_classes": 0},
                          **_timm_extra(name))
    st = os.path.join(weights, "model.safetensors")
    if os.path.exists(st):
        from safetensors.torch import load_file
        sd = load_file(st)
    else:
        sd = torch.load(os.path.join(weights, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m


class Encoder:
    def __init__(self, weights, batch=96, name="midnight"):
        self.timm = name in ("virchow2", "hoptimus0")
        if self.timm:
            self.m = _timm_model(weights, name).half().cuda().eval()
            self.prefix = self.m.num_prefix_tokens  # CLS + 4 registers
        else:
            from transformers import AutoModel
            self.m = AutoModel.from_pretrained(weights, dtype=torch.float16).cuda().eval()
            self.prefix = 1
        self.batch, self.name = batch, name
        (mean, std), self.pool = ENCODERS[name]
        self.mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1).half()
        self.std = torch.tensor(std, device="cuda").view(1, 3, 1, 1).half()
        self.dim = self([np.zeros((TILE, TILE, 3), np.uint8)]).shape[1]  # also a warm-up

    @torch.inference_mode()
    def __call__(self, tiles):
        x = torch.from_numpy(np.stack(tiles)).cuda().permute(0, 3, 1, 2).half().div_(255)
        if self.name == "midnight":
            x = x.sub_(0.5).div_(0.5)  # exactly the training-extraction arithmetic (parity with stored features)
        else:
            x = (x - self.mean) / self.std
        h = self.m.forward_features(x) if self.timm else self.m(pixel_values=x).last_hidden_state
        out = torch.cat([h[:, 0], h[:, self.prefix:].mean(1)], -1) if self.pool == "cls_mean" else h[:, 0]
        return out.float().cpu().numpy().astype(np.float16)


class MultiEncoder:
    """Several encoders on the same tiles, features concatenated per tile in the given order."""

    def __init__(self, encs):
        self.encs, self.batch, self.dim = encs, encs[0].batch, sum(e.dim for e in encs)

    def __call__(self, tiles):
        return np.concatenate([e(tiles) for e in self.encs], 1)


def extract(slide_path, mask_path, enc, pool, log=print, min_tissue=MIN_TISSUE, max_tiles=0, deadline=None,
            level=None, tile_mpp=TRAIN_MPP):
    """Tile and encode one slide. Returns dict f (N,3072) fp16, c (N,2) level-0 px, w (N,) uint8, g (N,2) grid idx, meta.
    deadline (time.time() value): visit blocks in random order and stop encoding once it passes."""
    mask, mpp_mask = read_mask(mask_path)
    with tifffile.TiffFile(slide_path) as tf:
        w0 = tf.series[0].levels[0].shape[1]
        m0 = None
        if tag_mpp(tf.series[0].levels[0].keyframe, 0.05, 5.0) is None and mpp_mask is not None:
            m0 = mpp_mask * mask.shape[1] / w0  # the mask covers the same area as the slide
            log(f"slide has no usable resolution tag; using {m0:.4f} um/px from the mask width")
        li, mpps, src = tiling_for(tf, level, m0, tile_mpp)
        if mpp_mask is not None and abs(np.log(mpp_mask * mask.shape[1] / (mpps[0] * w0))) > np.log(1.05):
            log(f"WARNING slide and mask extents differ by more than 5% ({mpps[0] * w0:.0f} vs "
                f"{mpp_mask * mask.shape[1]:.0f} um wide); using both files' own resolution tags")
        mpp = mpps[li] * src / TILE  # um/px of the encoded tiles
        if src != TILE:
            log(f"resampling level {li} ({mpps[li]:.3f} um/px): {src} px regions -> {TILE} px tiles at {mpp:.3f} um/px")
        if mpp_mask is None:  # e.g. a rewritten mask without resolution tags: it covers the same area as the slide
            mpp_mask = mpps[0] * tf.series[0].levels[0].shape[1] / mask.shape[1]
            log(f"mask has no resolution tags; using {mpp_mask:.4f} um/px from the slide width")
        arr = zarr.open(tf.aszarr(level=li), mode="r")
        frac = tissue_grid(mask, mpp_mask, mpp, (arr.shape[0] * TILE // src, arr.shape[1] * TILE // src))
        feats, grid, white, buf = [], [], [], []
        for gx, gy, t, w in iter_tiles(arr, frac, pool, min_tissue, max_tiles, shuffle=deadline is not None, src=src):
            buf.append(t)
            grid.append((gx, gy))
            white.append(w)
            if len(buf) == enc.batch:
                feats.append(enc(buf))
                buf = []
                if len(grid) % (enc.batch * 50) == 0:
                    log(f"  {len(grid)} tiles encoded")
                if deadline is not None and time.time() > deadline:
                    log(f"  WARNING time budget reached after {len(grid)} tiles: the tile set now depends on GPU speed")
                    break
        if buf:
            feats.append(enc(buf))
        f = np.concatenate(feats) if feats else np.zeros((0, enc.dim), np.float16)
        g = np.asarray(grid, np.int32).reshape(-1, 2)
        c = (g.astype(np.float64) * TILE * mpp / mpps[0]).round().astype(np.int32)
        meta = {"level": li, "src": src, "mpp": mpp, "mpp0": mpps[0], "grid": list(frac.shape), "n": int(len(f)),
                "mask_shape": list(mask.shape), "mpp_mask": mpp_mask, "n_candidates": int((frac >= min_tissue).sum())}
    return {"f": f, "c": c, "w": np.asarray(white, np.uint8), "g": g, "meta": meta, "mask": mask}
