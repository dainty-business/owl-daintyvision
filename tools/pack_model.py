"""Assemble a GC algorithm model tarball: one weights dir per encoder named in the run's calibration.json, mil/ (MIL
ensemble + calibration [+ atlas files]), code/.

usage: python pack_model.py <mil_run_dir> <code_dir> <out.tar.gz> <name>=<weights_dir> [...] [--atlas <dir>] [--vendor-timm]
  midnight=<dir> must hold the fp16 save_pretrained copy (config.json, model.safetensors). Timm encoders (virchow2,
  hoptimus0) are converted to fp16 model.safetensors next to their config.json. --atlas adds atlas.npz, atlas.json and
  atlas_descriptions.json to mil/. --vendor-timm copies the installed timm package into code/ (app.py puts code/ on
  sys.path), for an image built without timm.
The tarball extracts to /opt/ml/model; app.py imports code/ from there when present.
"""
import glob
import json
import os
import shutil
import sys
import tarfile

args = sys.argv[1:]
vendor = "--vendor-timm" in args
args = [x for x in args if x != "--vendor-timm"]
atlas = args.pop(args.index("--atlas") + 1) if "--atlas" in args else None
args = [x for x in args if x != "--atlas"]
mil, code, out = args[:3]
weights = dict(x.split("=", 1) for x in args[3:])
names = json.load(open(os.path.join(mil, "calibration.json"))).get("encoder", "midnight").split("+")
assert set(names) <= set(weights), f"calibration.json needs encoders {names}; given {sorted(weights)}"

stage = out + ".stage"
members = []
for n in names:
    if n in ("virchow2", "hoptimus0"):
        import torch
        from safetensors.torch import load_file, save_file
        src = weights[n]
        st = os.path.join(src, "model.safetensors")
        sd = load_file(st) if os.path.exists(st) else torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu",
                                                                 weights_only=True)
        os.makedirs(os.path.join(stage, n), exist_ok=True)
        save_file({k: v.half().contiguous() if v.is_floating_point() else v.contiguous() for k, v in sd.items()},
                  os.path.join(stage, n, "model.safetensors"))
        members += [(os.path.join(src, "config.json"), f"{n}/config.json"),
                    (os.path.join(stage, n, "model.safetensors"), f"{n}/model.safetensors")]
    else:
        members += [(os.path.join(weights[n], f), f"{n}/{f}") for f in ("config.json", "model.safetensors")]
members += [(p, "mil/" + os.path.basename(p)) for p in sorted(glob.glob(os.path.join(mil, "model_s*.pt")))]
members += [(os.path.join(mil, "calibration.json"), "mil/calibration.json")]
if atlas:
    members += [(os.path.join(atlas, f), "mil/" + f) for f in ("atlas.npz", "atlas.json", "atlas_descriptions.json")]
members += [(os.path.join(code, f), f"code/{f}") for f in ("inference.py", "owl_wsi.py", "owl_mil.py", "report.py")]
if vendor:
    import timm
    root = os.path.dirname(timm.__file__)
    members += [(p, "code/timm/" + os.path.relpath(p, root).replace(os.sep, "/"))
                for p in glob.glob(os.path.join(root, "**", "*.py"), recursive=True)]
assert all(os.path.exists(p) for p, _ in members), [p for p, _ in members if not os.path.exists(p)]
assert sum(a.startswith("mil/model_s") for _, a in members) >= 1
with tarfile.open(out + ".part", "w:gz", compresslevel=1) as t:
    for p, arc in members:
        t.add(p, arc)
os.replace(out + ".part", out)
shutil.rmtree(stage, ignore_errors=True)
print("\n".join(arc for _, arc in members if not arc.startswith("code/timm/")),
      f"\n(+ vendored timm)" if vendor else "", f"\n{os.path.getsize(out) / 1e9:.2f} GB -> {out}")
