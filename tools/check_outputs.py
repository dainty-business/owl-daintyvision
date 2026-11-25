"""Summarize one container output folder: the two JSON values, the attribution TIFF pyramid, the report's page count.

usage: python check_outputs.py <output_dir>
"""
import glob
import json
import os
import re
import sys

import tifffile

out = sys.argv[1]
for p in sorted(glob.glob(os.path.join(out, "*.json"))):
    print(os.path.basename(p), json.load(open(p)))
for p in glob.glob(os.path.join(out, "images", "*", "*.tif*")):
    with tifffile.TiffFile(p) as tf:
        lv = tf.series[0].levels
        a = lv[0].asarray()
        print(os.path.relpath(p, out), [(x.shape, str(x.dtype)) for x in lv], f"painted {(a > 0).mean():.3f}, max {a.max()}")
pdf = open(os.path.join(out, "pathology-report.pdf"), "rb").read()
print("report pages:", len(re.findall(rb"/Type\s*/Page[^s]", pdf)), f"{len(pdf) / 1e6:.1f} MB")
