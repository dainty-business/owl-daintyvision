"""Bootstrap the Harrell C-index of saved external predictions (tools/eval_external.py JSON with "cases" and "risk"):
95% interval over patients, and C per TCGA tissue source site (barcode TCGA-<site>-<patient>) with its event count.

usage: python boot_cindex.py runs/EXT-006_v3_risks.json tcga/labels.csv [--boot 2000]
"""
import argparse
import csv
import json

import numpy as np
from sksurv.metrics import concordance_index_censored

ap = argparse.ArgumentParser()
ap.add_argument("pred")
ap.add_argument("labels")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
p = json.load(open(a.pred))
lab = {r["case_id"]: (int(r["event"]), float(r["follow_up_years"])) for r in csv.DictReader(open(a.labels))}
cases, r = p["cases"], np.array(p["risk"])
E = np.array([lab[c][0] for c in cases], bool)
T = np.array([lab[c][1] for c in cases])


def c(idx):
    return concordance_index_censored(E[idx], T[idx], r[idx])[0]


full = c(np.arange(len(r)))
assert abs(full - p["abmil_ensemble"]["harrell"]) < 1e-9, "risks do not reproduce the saved C-index"
rng = np.random.default_rng(a.seed)
bs = []
for _ in range(a.boot):
    idx = rng.integers(0, len(r), len(r))
    if E[idx].any():
        bs.append(c(idx))
bs = np.array(bs)
print(f"n {len(r)}, events {E.sum()}, Harrell C {full:.3f}, 95% CI {np.percentile(bs, 2.5):.3f}-"
      f"{np.percentile(bs, 97.5):.3f}, bootstrap sd {bs.std():.3f}, share of resamples below 0.60 {np.mean(bs < 0.60):.3f}")
sites = np.array([x.split("-")[1] for x in cases])
print("site  n  events  C")
for s in sorted(set(sites), key=lambda s: -E[sites == s].sum()):
    idx = np.flatnonzero(sites == s)
    if E[idx].sum() >= 3:
        print(f"{s:>4} {len(idx):3d} {E[idx].sum():3d}  {c(idx):.3f}")
