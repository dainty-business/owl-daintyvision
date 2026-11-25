"""Score models trained on all LEOPARD cases on an external cohort (TCGA-PRAD BCR): Harrell and Uno C-index.

usage: python eval_external.py --train-feats feats --train-labels training_labels.csv --ext-feats tcga\\feats
       --ext-labels tcga\\labels.csv --model-dir runs\\final_e10_lr5e-5_s5 --out runs\\EXT-...json
Optional --encoder/--train-feats2/--ext-feats2 select pass-2 encoders (train_mil.py --encoder).
Scores (1) the ABMIL seed ensemble in --model-dir (mean risk over members) and (2) the mean-pool Cox baseline
refit on all training cases.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
from owl_mil import AdditiveABMIL  # noqa: E402
from train_mil import atlas_drop, load, meanpool_fit, predict  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("ext-feats", "ext-labels", "model-dir", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--train-feats", default="", help="LEOPARD features for Uno C and the mean-pool baseline (omit: Harrell only)")
ap.add_argument("--train-labels", default="")
ap.add_argument("--encoder", default="midnight")
ap.add_argument("--train-feats2", default="")
ap.add_argument("--ext-feats2", default="")
ap.add_argument("--drop-clusters", type=int, nargs="*", default=[], help="as train_mil.py: tiles removed in both cohorts")
ap.add_argument("--atlas", default="")
ap.add_argument("--max-tiles", type=int, default=8192,
                help="larger bags get a random subsample whose draw depends on the cases loaded before; 10**9 = full bags")
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
drop = atlas_drop(a.atlas, a.drop_clusters)
ce, Xe, Te, Ee = load(a.ext_feats, a.ext_labels, a.max_tiles, encoder=a.encoder, feats2=a.ext_feats2, drop=drop)
ye = np.array([(bool(e), t) for e, t in zip(Ee, Te)], dtype=[("e", bool), ("t", float)])
tau = None
if a.train_feats:
    _, Xt, Tt, Et = load(a.train_feats, a.train_labels, a.max_tiles, encoder=a.encoder, feats2=a.train_feats2, drop=drop)
    yt = np.array([(bool(e), t) for e, t in zip(Et, Tt)], dtype=[("e", bool), ("t", float)])
    tau = min(10.0, Te[Ee == 1].max() - 1e-6, Tt.max() - 1e-6)


def scores(r):
    out = {"harrell": float(concordance_index_censored(Ee.astype(bool), Te, r)[0])}
    if tau is not None:
        out["uno"] = float(concordance_index_ipcw(yt, ye, r, tau=tau)[0])
    return out


risks = []
for p in sorted(glob.glob(os.path.join(a.model_dir, "model_s*.pt"))):
    m = AdditiveABMIL(d_in=Xe[0].shape[1]).to(dev)
    m.load_state_dict(torch.load(p, map_location=dev))
    risks.append(predict(m, Xe, range(len(Xe)), dev))
res = {"n_ext": len(ce), "events_ext": int(Ee.sum()), "tau": tau, "model_dir": a.model_dir,
       "abmil_ensemble": scores(np.mean(risks, 0)), "abmil_members": [scores(r)["harrell"] for r in risks],
       "cases": list(ce), "risk": np.mean(risks, 0).tolist()}  # per-case ensemble risk for tools/boot_cindex.py
if a.train_feats:
    X = Xt + Xe
    res["meanpool"] = scores(meanpool_fit(X, np.concatenate([Tt, Te]), np.concatenate([Et, Ee]),
                                          np.arange(len(Xt)), np.arange(len(Xt), len(X))))
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res), flush=True)
