"""Mask side of the organizers' faithfulness notebook for one slide, using the notebook's own functions (cells 9, 11 and
13 are executed from the .ipynb, unchanged) with its default settings: the perturbed masks it would hand to the model,
written and read back with ASAP. Runs in Dockerfile.asap; run_masks.py then runs our model on them.

usage: python3 gen_masks.py <notebook.ipynb> <slide.tif> <mask.tif> <attribution_map.tif> <out_dir>
Writes <out_dir>/masks/<order>/<slide>_tissue_pct<NN>.tif and <out_dir>/plan.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import wholeslidedata as wsd
import multiresolutionimageinterface as mir

nb_path, wsi, mask_fp, map_fp, out = sys.argv[1:6]
out = Path(out)
g = {"np": np, "math": math, "wsd": wsd, "mir": mir, "Path": Path, "BACKEND": "asap", "SPACING": 0.5,
     "PATCH_SIZE": 256, "MASK_SUFFIX": "_tissue", "REMOVAL_MODE": "cumulative", "RUN_NAME": "check",
     "PERTURBED_MASK_DIR": out / "masks"}
cells = json.load(open(nb_path, encoding="utf-8"))["cells"]
for i in (9, 11, 13):
    exec("import contextlib, os, sys\n" + "".join(cells[i]["source"]), g)

PATCH_TISSUE_MIN, STEPS, SEEDS, TIE_SEED = 0.10, [1, 2, 4, 8, 16, 32, 64], (33, 79, 98), 0
slide, mask_image = wsd.WholeSlideImage(wsi, backend="asap"), wsd.WholeSlideImage(mask_fp, backend="asap")
tile = g["mask_tile_size"](slide, mask_image)
spacing = float(mask_image.spacings[0])
shapes = g["level_shapes"](mask_fp)
mask = g["read_level"](mask_fp, 0)
values = g["read_level"](map_fp, 0)
assert values.shape == mask.shape, (values.shape, mask.shape)
tissue = mask == 1
rows, cols, counts = g["find_tissue_patches"](tissue, tile, PATCH_TISSUE_MIN)
medians = g["patch_medians"](values, tissue, rows, cols, tile)
sel = {("attribution", p): g["attribution_order"](medians, TIE_SEED)[:g["patches_to_remove"](p, len(rows))] for p in STEPS}
for s in SEEDS:
    for p in STEPS:
        sel[(f"seed{s}", p)] = g["random_selection"](len(rows), s, p, "cumulative")
sel[("zero_check", 0)] = np.array([], np.int64)
plan = {"tile": tile, "spacing": spacing, "shapes": shapes, "n_patches": int(len(rows)),
        "pyramid_problem": g["original_pyramid_problem"](mask_fp, mask, shapes), "steps": []}
stem = Path(wsi).stem
for (order, p), s in sel.items():
    path = out / "masks" / order / g["mask_name"](stem, p)
    g["ensure_mask"](path, g["remove_patches"](mask, rows[s], cols[s], tile), shapes, spacing)  # raises if ASAP's file differs
    plan["steps"].append({"order": order, "pct": p, "mask": str(path.relative_to(out)), "n_removed": int(len(s)),
                          "tissue_removed_pct": 100 * float(counts[s].sum()) / float(tissue.sum())})
json.dump(plan, open(out / "plan.json", "w"), indent=1)
print(f"tile {tile} mask px, {len(rows)} patches, {len(plan['steps'])} masks written and verified; "
      f"pyramid problem: {plan['pyramid_problem']}", flush=True)
