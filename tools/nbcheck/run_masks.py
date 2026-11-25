"""Model side of the organizers' faithfulness notebook: run our inference.py on every mask gen_masks.py wrote, the way
the notebook does (one parse of MODEL_ARGS, then per mask: link slide and mask into the input folders, delete the old
JSON, run(args), read the years), and score the curves with the notebook's own curve_auc (cell 11). Runs in our image.

usage: python run_masks.py <notebook.ipynb> <slide.tif> <run_dir> <baseline_years> <z_mean> <z_sd> [MODEL_ARGS...]
"""
import json
import sys
from pathlib import Path

import numpy as np

import inference

nb_path, wsi, run_dir = sys.argv[1:4]
baseline, z_mean, z_sd = (float(v) for v in sys.argv[4:7])
run_dir, wsi = Path(run_dir), Path(wsi)
g = {"np": np}
exec("".join(json.load(open(nb_path, encoding="utf-8"))["cells"][11]["source"]), g)
tmp = Path("/tmp/nbcheck")  # links inside the container: bind mounts from Windows may not take symlinks
wsi_dir, mask_dir, out_json = tmp / "model_in/slides", tmp / "model_in/masks", tmp / "model_out/years.json"
args = inference.get_args_parser().parse_args(
    ["--wsi-dir", str(wsi_dir), "--mask-dir", str(mask_dir), "--output-json", str(out_json)] + sys.argv[7:])
plan = json.load(open(run_dir / "plan.json"))
z = lambda y: (y - z_mean) / z_sd  # noqa: E731
rows = []
for st in plan["steps"]:
    for d in (wsi_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.iterdir():
            f.unlink()
    (wsi_dir / wsi.name).symlink_to(wsi.resolve())
    (mask_dir / f"{wsi.stem}_tissue.tif").symlink_to((run_dir / st["mask"]).resolve())
    out_json.unlink(missing_ok=True)
    inference.run(args)
    years = float(json.loads(out_json.read_text()))
    rows.append({**st, "years": years, "abs_delta_z": abs(z(years) - z(baseline))})
    print(st["order"], st["pct"], f"{years:.6f}", f"{rows[-1]['abs_delta_z']:.5f}", flush=True)

curve = {o: [r["abs_delta_z"] for r in rows if r["order"] == o] for o in ("attribution", "seed33", "seed79", "seed98")}
pcts = [1, 2, 4, 8, 16, 32, 64]
auc = {o: g["curve_auc"](pcts, v) for o, v in curve.items()}
auc["random_3seeds"] = float(np.mean([auc[f"seed{s}"] for s in (33, 79, 98)]))
res = {"zero_check_abs_delta_z": next(r["abs_delta_z"] for r in rows if r["order"] == "zero_check"), "auc": auc,
       "gap_1seed": auc["attribution"] - auc["seed33"], "gap_3seeds": auc["attribution"] - auc["random_3seeds"],
       "rows": rows}
json.dump(res, open(run_dir / "result.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1), flush=True)
