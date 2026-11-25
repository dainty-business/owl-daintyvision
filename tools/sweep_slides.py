"""Run the shipped container code end to end (tiling, encoding, map, report) on every slide of a folder and compare each
predicted years with the cached-feature prediction (tools/cached_years.py), to catch per-slide crashes and
train/inference drift on more slides than the Sanity Checks saw.

usage: python sweep_slides.py --code test\\v41code --model-dir pkg\\v4 --slides gallery_raw --masks masks
       --cached runs\\years_v4_leopard.json --out test\\sweep_v41
Writes <out>/<case>/ (the container's outputs) and <out>/sweep.json; exits non-zero if any slide fails or drifts.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path

ap = argparse.ArgumentParser()
for k in ("code", "model-dir", "slides", "masks", "cached", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--tol", type=float, default=0.0005, help="years; the cross-host agreement measured on 4 cases")
a = ap.parse_args()
os.environ["OWL_MODEL_DIR"] = os.path.abspath(a.model_dir)
sys.path.insert(0, os.path.abspath(a.code))
import inference  # noqa: E402
from pypdf import PdfReader  # noqa: E402

cached = json.load(open(a.cached))
model = inference.Model(a.model_dir)
rows, bad = [], 0
for wsi in sorted(glob.glob(os.path.join(a.slides, "*.tif"))):
    case = Path(wsi).stem
    out = Path(a.out) / case
    out.mkdir(parents=True, exist_ok=True)
    row = {"case": case}
    t0 = time.time()
    try:
        inference.predict_case(model, wsi, os.path.join(a.masks, case + "_tissue.tif"), out)
        row["years"] = json.load(open(out / "overall-survival-years.json"))
        row["class"] = json.load(open(out / "prostate-cancer-risk-classification.json"))
        row["pages"] = len(PdfReader(out / "pathology-report.pdf").pages)
        row["diff"] = row["years"] - cached[case]
        row["ok"] = abs(row["diff"]) < a.tol and row["pages"] <= 5
    except Exception:
        row["error"] = traceback.format_exc()[-800:]
        row["ok"] = False
    row["seconds"] = round(time.time() - t0, 1)
    bad += not row["ok"]
    rows.append(row)
    print(json.dumps({k: v for k, v in row.items() if k != "error"}), row.get("error", ""), flush=True)
json.dump(rows, open(Path(a.out) / "sweep.json", "w"), indent=1)
print(f"SWEEP {len(rows) - bad} of {len(rows)} ok", flush=True)
sys.exit(1 if bad else 0)
