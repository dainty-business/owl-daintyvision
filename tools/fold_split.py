"""Write one CV repeat's folds as files (FAITH-004): per fold, the training labels (for train_mil.py --final) and the
held-out case ids, split exactly as tools/train_mil.py's cross-validation (StratifiedKFold on the event indicator over
the label rows that have features, in label-file order).

usage: python fold_split.py --feats feats --labels training_labels.csv --out runs\\folds_r0 [--repeat 0]
"""
import argparse
import csv
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
for k in ("feats", "labels", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--repeat", type=int, default=0)
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rows = list(csv.DictReader(open(a.labels)))
fields = list(rows[0])
rows = [r for r in rows if os.path.exists(os.path.join(a.feats, r["case_id"].replace(".tif", "") + ".npz"))]
E = np.array([int(r["event"]) for r in rows])
split = StratifiedKFold(a.folds, shuffle=True, random_state=a.repeat).split(np.zeros(len(E)), E)
for k, (tr, va) in enumerate(split):
    with open(os.path.join(a.out, f"train_k{k}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows[i] for i in tr)
    with open(os.path.join(a.out, f"heldout_k{k}.txt"), "w") as fh:
        fh.write("\n".join(rows[i]["case_id"].replace(".tif", "") for i in va) + "\n")
    print(f"fold {k}: train {len(tr)} ({E[tr].sum()} events), held out {len(va)} ({E[va].sum()} events)")
