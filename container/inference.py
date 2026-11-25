"""OWL algorithm (team DaintyVision): frozen-encoder tile features -> additive attention MIL Cox model -> time to BCR,
risk class, uint8 attribution TIFF at the tissue-mask spacing and a PDF report.

init_model() runs once at server start (app.py); run(model) handles one case per /invoke.
The organizers' faithfulness notebook instead calls run(get_args_parser().parse_args([...])) once per perturbed mask;
run() accepts that namespace too and caches the model between calls.
Model files (GC algorithm model tarball, extracted to /opt/ml/model):
  <encoder>/           tile-encoder weights (config.json, model.safetensors) for each name in calibration.json
                       "encoder" ("midnight" when absent; "a+b" concatenates features per tile)
  mil/model_s*.pt      AdditiveABMIL state dicts (seed ensemble)
  mil/calibration.json Breslow baseline, RMST tau, risk-class time thresholds, training risk quantiles
Any failure raises: the case fails rather than writing a placeholder.
"""
import argparse
import glob
import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import owl_wsi
from owl_mil import AdditiveABMIL, rmst

INPUT_PATH = Path(os.environ.get("OWL_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("OWL_OUTPUT", "/output"))
MODEL_DIR = Path(os.environ.get("OWL_MODEL_DIR", "/opt/ml/model"))
MAX_TILES = int(os.environ.get("OWL_MAX_TILES", "0"))  # tile cap; 0 = TILE_BUDGET / T4 cost of the model's encoders
# Safety net only. The tile cap must bind first: a run stopped by the clock keeps a speed-dependent tile set, and the
# faithfulness notebook aborts when a repeat run moves the prediction by more than 0.01 SD.
ENCODE_BUDGET_S = float(os.environ.get("OWL_ENCODE_BUDGET_S", "2400"))
# Midnight-equivalent tiles per case. SUB-002 measured 32-35 tiles/s end to end on the T4, so 28000 tiles take ~15 min.
# Kept, not raised: on LEOPARD the cap moves predictions by <= 0.025 SD (tools/check_cap.py), so more tiles buy nothing.
TILE_BUDGET = 28000
# Per-tile encode cost relative to Midnight (4080, uncontended); phikon2 is an unmeasured estimate (ViT-L/16).
T4_COST = {"midnight": 1.0, "hoptimus0": 1.02, "virchow2": 0.64, "phikon2": 0.3}
HEATMAP_DIR = "images/prostatectomy-overall-survival-attribution-heatmap"
PCTS = (1, 2, 4, 8, 16, 32, 64)  # removal steps (% of tissue) of the organizers' faithfulness test


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Model:
    def __init__(self, model_dir=MODEL_DIR):
        model_dir = Path(model_dir)
        assert torch.cuda.is_available(), "CUDA is required"
        log("GPU:", torch.cuda.get_device_name(0))
        self.cal = json.load(open(model_dir / "mil" / "calibration.json"))
        names = self.cal.get("encoder", "midnight").split("+")
        encs = [owl_wsi.Encoder(str(model_dir / n), batch=64, name=n) for n in names]
        self.enc = encs[0] if len(encs) == 1 else owl_wsi.MultiEncoder(encs)
        # cheaper encoders keep the Midnight cap: tile reading, not the GPU, may bind on GC
        self.max_tiles = (MAX_TILES or self.cal.get("max_tiles")  # set by train_mil for a non-default tile spacing
                          or int(TILE_BUDGET / max(1.0, sum(T4_COST[n] for n in names))))
        self.mils = []
        for p in sorted(glob.glob(str(model_dir / "mil" / "model_s*.pt"))):
            m = AdditiveABMIL(d_in=self.enc.dim).cuda().eval()
            m.load_state_dict(torch.load(p, map_location="cuda"))
            self.mils.append(m)
        assert self.mils, f"no MIL weights in {model_dir / 'mil'}"
        self.atlas = None  # optional tissue-pattern atlas (tools/build_atlas.py): mil/atlas.npz, atlas.json, atlas_descriptions.json
        if (model_dir / "mil" / "atlas.npz").exists():
            z = np.load(model_dir / "mil" / "atlas.npz")
            self.atlas = {"mean": z["pca_mean"], "comps": z["pca_components"], "centers": z["centers"],
                          "stats": json.load(open(model_dir / "mil" / "atlas.json"))["clusters"],
                          "desc": json.load(open(model_dir / "mil" / "atlas_descriptions.json"))["clusters"]}
        self.ts, self.H0 = np.array(self.cal["ts"]), np.array(self.cal["H0"])
        self.mpp = self.cal.get("tile_mpp", owl_wsi.TRAIN_MPP)  # tile spacing the MIL members were trained on
        self.pool = ThreadPoolExecutor(max(4, (os.cpu_count() or 4)))
        log(f"model ready: encoder {'+'.join(names)} ({self.enc.dim}-d), {len(self.mils)} MIL members, "
            f"tile cap {self.max_tiles}, tiles at {self.mpp} um/px")

    @torch.no_grad()
    def mil(self, f):
        """Ensemble risk and per-tile signed influence s_i (mean over members): removing tile i changes the risk by
        -s_i, since dropping it renormalizes attention: new risk - risk = -a_i (r_i - risk) / (1 - a_i)."""
        x = torch.from_numpy(f).cuda().float()
        risk, signed = [], 0
        for m in self.mils:
            rk, a, r = m(x)
            risk.append(rk.item())
            signed = signed + (a * (r - rk) / (1 - a.clamp(max=0.999))).cpu().numpy()
        return float(np.mean(risk)), signed / len(self.mils)

    def time_of(self, risk):
        return float(rmst(risk, self.ts, self.H0, self.cal["tau"])[0])

    def risk_class(self, t):
        lo, hi = self.cal["time_tertiles"]
        return "high" if t <= lo else ("medium" if t <= hi else "low")


def init_model():
    return Model()


def get_args_parser():
    """Arguments for notebook-style calls (organizers' faithfulness_blackbox.ipynb); defaults are the GC paths."""
    p = argparse.ArgumentParser("OWL DaintyVision inference")
    p.add_argument("--wsi-dir", default=str(INPUT_PATH / "images/prostatectomy-wsi"))
    p.add_argument("--mask-dir", default=str(INPUT_PATH / "images/prostatectomy-tissue-mask"))
    p.add_argument("--output-json", default=str(OUTPUT_PATH / "overall-survival-years.json"))
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--prediction-only", action="store_true", help="write only the predicted years (faster)")
    return p


def find_one(folder):
    files = [f for ext in ("tif", "tiff", "mha") for f in glob.glob(str(Path(folder) / f"*.{ext}"))]
    assert len(files) == 1, f"expected one file in {folder}, found {files}"
    return files[0]


def map_values(score):
    """uint8 attribution value per tile: 1..255 = rank(score)^4 (0 is left for no tile). Keeps the order that removal
    tests use, with the values concentrated on the top tiles."""
    ranks = np.argsort(np.argsort(score, kind="stable"), kind="stable")
    return (1 + np.round(254 * (ranks / max(1, len(score) - 1)) ** 4)).astype(np.uint8)


def write_heatmap(path, arr, mpp):
    """uint8 pyramidal tiled TIFF shaped like the official masks: 512 tiles, LZW, reduced pages halving, um -> px/cm."""
    import tifffile
    with tifffile.TiffWriter(path) as tw:
        level, k = arr, 0
        while True:  # each reduced page halves the size, so its spacing doubles (as in the official masks)
            res = 1e4 / (mpp * 2 ** k)
            tw.write(level, tile=(512, 512), compression="lzw", photometric="minisblack", resolution=(res, res),
                     resolutionunit="CENTIMETER", subfiletype=0 if k == 0 else 1)
            k += 1
            if max(level.shape) <= 512:
                break
            h, w = (level.shape[0] + 1) // 2 * 2, (level.shape[1] + 1) // 2 * 2
            level = np.pad(level, ((0, h - level.shape[0]), (0, w - level.shape[1])))
            level = level.reshape(h // 2, 2, w // 2, 2).max((1, 3))  # max-pool keeps hotspots visible


def heat_array(mask_shape, g, values, cell):
    """Paint per-tile values (uint8) onto the mask grid; cell = tile size in mask px (float)."""
    out = np.zeros(mask_shape, np.uint8)
    for (gx, gy), v in zip(g, values):
        y0, x0 = int(round(gy * cell)), int(round(gx * cell))
        out[y0:int(round((gy + 1) * cell)), x0:int(round((gx + 1) * cell))] = v
    return out


def removal_auc(model, f, score):
    """Mean |change in predicted years| when the top PCTS% tiles by score are removed: a tile-level stand-in for the
    organizers' mask-patch test (trapezoid over linear percent, y = 0 at 0%, divided by the x range, as theirs)."""
    base = model.time_of(model.mil(f)[0])
    order = np.argsort(-score, kind="stable")
    ys = [abs(model.time_of(model.mil(f[np.sort(order[min(len(f) - 1, math.ceil(p * len(f) / 100)):])])[0]) - base)
          for p in PCTS]
    x, y = np.r_[0.0, np.asarray(PCTS) / 100], np.r_[0.0, ys]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))


def faithfulness_curve(model, f, infl, fracs=tuple(p / 100 for p in PCTS), n_random=10, seed=0):
    """Predicted-time change when the top-ranked tiles (by the map's ranking score) vs random tiles are removed."""
    rng = np.random.default_rng(seed)
    base = model.time_of(model.mil(f)[0])
    order = np.argsort(-infl)
    top, rand = [], []
    for q in fracs:
        k = min(len(f) - 1, max(1, math.ceil(q * len(f))))
        keep = np.sort(order[k:])
        top.append(abs(model.time_of(model.mil(f[keep])[0]) - base))
        rs = []
        for _ in range(n_random):
            keep = np.sort(rng.choice(len(f), len(f) - k, replace=False))
            rs.append(abs(model.time_of(model.mil(f[keep])[0]) - base))
        rand.append(float(np.mean(rs)))
    return list(fracs), top, rand


_MODELS = {}


def run(model):
    """GC /invoke: run(Model). Notebook: run(argparse.Namespace from get_args_parser())."""
    if isinstance(model, argparse.Namespace):
        a = model
        if a.model_dir not in _MODELS:
            _MODELS[a.model_dir] = Model(a.model_dir)
        out = Path(a.output_json)
        return predict_case(_MODELS[a.model_dir], find_one(a.wsi_dir), find_one(a.mask_dir), out.parent,
                            years_json=out, full=not a.prediction_only)
    return predict_case(model, find_one(INPUT_PATH / "images/prostatectomy-wsi"),
                        find_one(INPUT_PATH / "images/prostatectomy-tissue-mask"), OUTPUT_PATH)


def minimal_report(path, stem, years, klass, risk, err):
    """One-page PDF written without matplotlib when the full report fails."""
    lines = ["OWL biochemical-recurrence report (research use only)", f"Slide: {stem}",
             f"Predicted BCR-free time: {years:.2f} years (restricted mean over 10 years).",
             f"Risk class: {str(klass).upper()}. Model risk score {risk:+.2f} (higher = earlier predicted recurrence).",
             f"The full report could not be produced for this slide ({err}).",
             "The predicted time, the risk class and the attribution map are unaffected."]
    esc = lambda t: t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")  # noqa: E731
    text = "BT /F1 11 Tf 16 TL 56 790 Td " + " ".join(f"({esc(t)}) '" for t in lines) + " ET"
    stream = text.encode("latin-1", "replace")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    Path(path).write_bytes(bytes(out))


def predict_case(model, wsi, mask, out_dir, years_json=None, full=True):
    t0 = time.time()
    out_dir = Path(out_dir)
    years_json = Path(years_json) if years_json else out_dir / "overall-survival-years.json"
    log("input", wsi, mask)
    ex = owl_wsi.extract(wsi, mask, model.enc, model.pool, log=log, max_tiles=model.max_tiles,
                         deadline=t0 + ENCODE_BUDGET_S, tile_mpp=model.mpp)
    if ex["meta"]["n"] == 0:  # degenerate mask: fall back to any tissue rather than failing the case
        log("no tiles at 50% tissue; retrying at 1%")
        ex = owl_wsi.extract(wsi, mask, model.enc, model.pool, log=log, min_tissue=0.01, max_tiles=model.max_tiles,
                             deadline=t0 + ENCODE_BUDGET_S, tile_mpp=model.mpp)
    keep = ex["w"] <= owl_wsi.MAX_WHITE
    if keep.sum() == 0:
        keep[:] = True
    f, g, c = ex["f"][keep], ex["g"][keep], ex["c"][keep]
    ok = np.isfinite(f.astype(np.float32)).all(1)
    if not ok.all():
        log(f"WARNING dropping {int((~ok).sum())} tiles with non-finite features")
        f, g, c = f[ok], g[ok], c[ok]
    assert len(f) > 0, "no usable tiles"
    log(f"tiles: {len(f)} used of {ex['meta']['n']} (candidates {ex['meta']['n_candidates']}), {time.time() - t0:.0f}s")

    risk, signed = model.mil(f)
    # rank tiles by signed influence in one direction (ranking by |signed| mixes directions whose removals cancel);
    # pick the direction whose removal moves the predicted years more (offline replica on TCGA: gap 0.96 vs 0.91 for
    # the risk-median rule, tools/faithfulness_sim.py)
    direction = max((1.0, -1.0), key=lambda d: removal_auc(model, f, d * signed))
    infl = direction * signed
    years = model.time_of(risk)
    klass = model.risk_class(years)
    log(f"risk {risk:.4f} -> {years:.3f} years, class {klass}")
    if not full:
        years_json.parent.mkdir(parents=True, exist_ok=True)
        years_json.write_text(json.dumps(years))
        return 0

    meta = ex["meta"]
    cell = owl_wsi.TILE * meta["mpp"] / meta["mpp_mask"]
    heat = heat_array(ex["mask"].shape, g, map_values(infl), cell)
    (out_dir / HEATMAP_DIR).mkdir(parents=True, exist_ok=True)
    write_heatmap(str(out_dir / HEATMAP_DIR / f"{Path(wsi).stem}.tif"), heat, meta["mpp_mask"])

    pdf = out_dir / "pathology-report.pdf"
    try:
        curve = faithfulness_curve(model, f, infl)
        import report  # matplotlib import only when needed
        report.make(str(pdf), wsi=wsi, model=model, f=f, g=g, c=c, signed=signed, infl=infl, direction=direction,
                    risk=risk, years=years, klass=klass, meta=meta, curve=curve, mask=ex["mask"], cell=cell)
    except Exception as e:  # prediction and map are final here; a report bug must not fail the case
        traceback.print_exc()
        log(f"WARNING report failed ({type(e).__name__}); writing the minimal report")
        minimal_report(pdf, Path(wsi).stem, years, klass, risk, type(e).__name__)

    years_json.write_text(json.dumps(years))
    with open(out_dir / "prostate-cancer-risk-classification.json", "w") as fh:
        json.dump(klass, fh)
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
