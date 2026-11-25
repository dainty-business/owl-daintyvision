"""Cross-validated training of the additive ABMIL Cox model (or a mean-pool Cox baseline) on extracted Midnight features.

usage:
  python train_mil.py --feats work\\feats --labels data/raw_labels/leopard_labels/training_labels.csv --out runs/x
  [--model abmil|meanpool] [--folds 5 --repeats 3] [--epochs 30] [--final --seeds 3]
  [--encoder virchow2 --feats2 work\\feats2]  (pass-2 encoders; "midnight+virchow2" concatenates per tile)

CV writes out/oof_r{repeat}.csv (case, time, event, risk, fold), out/curve.csv (epoch x fold val C-index) and
out/results.json (Harrell C per repeat on pooled OOF risks, mean per-fold C, Uno C at tau). --final trains on all
cases and writes out/model_s{seed}.pt plus out/calibration.json (Breslow baseline, RMST tau, risk-class thresholds).
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
from owl_mil import AdditiveABMIL, MultiAdditiveABMIL, breslow, cox_loss, nll_loss, risk_of, rmst  # noqa: E402
from owl_wsi import MAX_WHITE  # noqa: E402
from report import assign_clusters  # noqa: E402

TAU = 10.0


def atlas_drop(path, ids):
    """(atlas, cluster ids) for load(drop=...), or None when no clusters are dropped."""
    if not ids:
        return None
    z = np.load(path)
    return {"mean": z["pca_mean"], "comps": z["pca_components"], "centers": z["centers"]}, list(ids)


def tile_mpp(feats_dir, cases):
    """Tile spacing recorded by the extractor in <feats>_meta/<case>.json (tools/extract_hires.py), or None for pass-1
    features (0.97 um/px, no such dir). Requires one spacing for every case."""
    md = feats_dir.rstrip("/\\") + "_meta"
    if not os.path.isdir(md):
        return None
    m = [json.load(open(os.path.join(md, c + ".json")))["mpp"] for c in cases]
    assert max(m) / min(m) < 1.02, f"mixed tile spacings in {md}: {min(m)}-{max(m)}"
    return round(float(np.mean(m)), 4)


def cidx(E, T, r):
    try:
        return concordance_index_censored(E.astype(bool), T, r)[0]
    except Exception:  # too few cases (smoke tests): no comparable pairs
        return float("nan")


def tile_features(feats_dir, feats2_dir, case, encoder):
    """Per-tile features of `encoder` with near-white tiles dropped, or None if any part is missing. "midnight" comes
    from pass 1 (f10, w10), other names from pass 2 (f_<name>, w; same tiles), "a+b" concatenates per tile."""
    parts, white = [], None
    for name in encoder.split("+"):
        fk, wk = ("f10", "w10") if name == "midnight" else ("f_" + name, "w")
        p = os.path.join(feats_dir if name == "midnight" else feats2_dir, case + ".npz")
        if not os.path.exists(p):
            return None
        z = np.load(p)
        if fk not in z.files:
            return None
        parts.append(z[fk])
        if wk in z.files:
            white = z[wk]
    assert len({len(x) for x in parts}) == 1, f"{case}: encoders disagree on the tile count"
    f = np.concatenate(parts, 1) if len(parts) > 1 else parts[0]
    return f if white is None else f[white <= MAX_WHITE]


def load(feats_dir, labels, max_tiles, seed=0, encoder="midnight", feats2="", views=(), drop=None):
    """drop = (atlas dict, cluster ids): tiles nearest those atlas clusters are removed before subsampling."""
    rng = np.random.default_rng(seed)
    cases, X, T, E = [], [], [], []
    for row in csv.DictReader(open(labels)):
        case = row["case_id"].replace(".tif", "")
        f = tile_features(feats_dir, feats2, case, encoder)
        if f is None:
            continue
        if drop:
            assert not views, "--drop-clusters with --views is not implemented"
            cl = np.concatenate([assign_clusters(drop[0], f[i:i + 8192]) for i in range(0, len(f), 8192)])
            f = f[~np.isin(cl, drop[1])]
        if views:  # (1 + len(views), n, d): original first, then the same tiles' extra views (pass-2 keys)
            with np.load(os.path.join(feats2, case + ".npz")) as z:
                keep = z["w"] <= MAX_WHITE
                f = np.stack([f] + [z[k][keep] for k in views])
        if f.shape[-2] > max_tiles:
            idx = np.sort(rng.choice(f.shape[-2], max_tiles, replace=False))
            f = f[..., idx, :]
        cases.append(case)
        X.append(torch.from_numpy(np.ascontiguousarray(f)))
        T.append(float(row["follow_up_years"]))
        E.append(int(row["event"]))
    return cases, X, np.array(T), np.array(E)


class TwoScale(torch.nn.Module):
    """Fine (0.97 um/px) and context (3.9 um/px) additive MIL branches; slide risk = sum of the two branch risks."""

    def __init__(self, dropout, d_fine=3072):
        super().__init__()
        self.fine, self.ctx = AdditiveABMIL(d_in=d_fine, dropout=dropout), AdditiveABMIL(dropout=dropout)

    def forward(self, pair):
        return self.fine(pair[0])[0] + self.ctx(pair[1])[0], None, None


def prep(b, dev, rng=None, bag=0):
    """A bag (or a (fine, context) pair of bags) on the device, optionally subsampled to `bag` tiles."""
    if isinstance(b, tuple):
        return tuple(prep(x, dev, rng, bag) for x in b)
    if b.dim() == 3:  # views of one bag: a random view in training, the original (view 0) otherwise
        b = b[int(rng.integers(len(b)))] if rng is not None else b[0]
    if rng is not None and len(b) > bag:
        b = b[torch.from_numpy(rng.choice(len(b), bag, replace=False))]
    return b.to(dev).float()


def load_ctx(ctx_dir, cases):
    out = []
    for case in cases:
        z = np.load(os.path.join(ctx_dir, case + ".npz"))
        fa, w = (z["fc"], z["wc"]) if "fc" in z.files else (z["f"], z["w"])  # pass 2 (WSI level) or lowres JPEGs
        f = fa[w <= MAX_WHITE]
        out.append(torch.from_numpy(np.ascontiguousarray(f if len(f) else fa)))
    return out


def predict(model, X, idx, dev):
    model.eval()
    with torch.no_grad():
        return np.array([risk_of(model(prep(X[i], dev))[0]).item() for i in idx])


def site_folds(sites, E, k, seed):
    """Assign whole sites to k folds: sites in random order, each to the fold with the fewest events (then cases)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(sorted(set(sites)))
    ev, n, fold = np.zeros(k), np.zeros(k), {}
    for s in order:
        f = int(np.lexsort((n, ev))[0])
        fold[s] = f
        ev[f] += E[sites == s].sum()
        n[f] += (sites == s).sum()
    return np.array([fold[s] for s in sites])


def strat_cox(risk, T, E, S):
    """Cox partial likelihood with a separate risk set per cohort (stratified), weighted by each cohort's events."""
    loss = 0
    for c in np.unique(S):
        m = torch.from_numpy(S == c).to(risk.device)
        loss = loss + cox_loss(risk[m], T[m], E[m]) * E[m].sum()
    return loss / E.sum()


def rank_loss(risk, time, event):
    """Pairwise logistic surrogate of Harrell's C over the batch's comparable pairs (i's event before j's time)."""
    comp = (event[:, None] > 0) & (time[:, None] < time[None, :])
    d = (risk[:, None] - risk[None, :])[comp]
    return torch.nn.functional.softplus(-d).mean() if d.numel() else risk.sum() * 0


def train_abmil(X, T, E, tr, va, a, dev, seed, curve=None, S=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    two = isinstance(X[0], tuple)
    d_in = (X[0][0] if two else X[0]).shape[-1]
    model = (TwoScale(a.dropout, d_in) if two else
             MultiAdditiveABMIL(d_in=d_in, d=a.hidden, d_att=a.hidden // 2, dropout=a.dropout, branches=a.branches)
             if getattr(a, "branches", 1) > 1 else
             AdditiveABMIL(d_in=d_in, d=a.hidden, d_att=a.hidden // 2, dropout=a.dropout,
                           n_out=a.bins if a.loss == "nll" else 1)).to(dev)
    # discrete-time bins: quantiles of the training fold's event times (fold-internal, no leakage)
    edges = np.quantile(T[tr][E[tr] == 1], np.linspace(0, 1, a.bins + 1)[1:-1]) if a.loss == "nll" else None
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(tr)
        for b in range(0, len(perm), a.batch):
            bi = perm[b:b + a.batch]
            if E[bi].sum() == 0:
                continue
            risks = []
            for i in bi:
                risks.append(model(prep(X[i], dev, rng, a.bag))[0])
            if a.loss in ("cox", "rank"):
                rt, tt, et = torch.stack(risks), torch.tensor(T[bi], device=dev), torch.tensor(E[bi], device=dev, dtype=torch.float32)
                loss = (rank_loss(rt, tt, et) if a.loss == "rank" else
                        cox_loss(rt, tt, et) if S is None else strat_cox(rt, tt, et, S[bi]))
            else:
                yb = torch.tensor(np.searchsorted(edges, T[bi], side="right"), device=dev)
                loss = nll_loss(torch.stack(risks), yb, torch.tensor(1 - E[bi], device=dev, dtype=torch.float32))
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if curve is not None and va is not None and len(va):
            r = predict(model, X, va, dev)
            curve.append(cidx(E[va], T[va], r))
    return model


def meanpool_fit(X, T, E, tr, va):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    M = np.stack([x.float().mean(0).numpy() for x in X])
    sc = StandardScaler().fit(M[tr])
    pca = PCA(min(32, len(tr) - 1), random_state=0).fit(sc.transform(M[tr]))
    y = np.array([(bool(e), t) for e, t in zip(E[tr], T[tr])], dtype=[("e", bool), ("t", float)])
    cox = CoxPHSurvivalAnalysis(alpha=10.0).fit(pca.transform(sc.transform(M[tr])), y)
    return cox.predict(pca.transform(sc.transform(M[va])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="abmil", choices=["abmil", "meanpool"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--bag", type=int, default=2048)
    ap.add_argument("--max-tiles", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--loss", default="cox", choices=["cox", "nll", "rank"])
    ap.add_argument("--views", nargs="*", default=[], help="pass-2 keys of extra views of the same tiles (e.g. "
                    "f_midnight_hed0 f_midnight_hed1): each training bag uses a random view; validation uses the original")
    ap.add_argument("--drop-clusters", type=int, nargs="*", default=[], help="atlas clusters whose tiles are removed")
    ap.add_argument("--atlas", default="", help="atlas.npz (tools/build_atlas.py) for --drop-clusters")
    ap.add_argument("--hidden", type=int, default=256, help="MIL embedding width (attention width is half)")
    ap.add_argument("--branches", type=int, default=1, help="attention branches; 1 = AdditiveABMIL")
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--ctx", default="", help="context-feature dir (extract_lowres.py): two-scale model")
    ap.add_argument("--ctx-only", action="store_true", help="train on the context features alone")
    ap.add_argument("--encoder", default="midnight", help='tile encoder(s): midnight (pass 1) or pass-2 names, "a+b" concatenates')
    ap.add_argument("--feats2", default="", help="pass-2 feature dir (extract_pass2.py)")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0, help="shift the CV training seeds (folds unchanged); fuse runs with different offsets to score a seed ensemble")
    ap.add_argument("--add-feats", default="", help="second cohort (TCGA-PRAD): its sites are split into folds; each "
                                                     "fold's held-out sites are predicted by that fold's model")
    ap.add_argument("--add-labels", default="")
    ap.add_argument("--add-train", action="store_true", help="also train on the second cohort's other sites "
                                                             "(Cox risk sets stratified by cohort)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    cases, X, T, E = load(a.feats, a.labels, a.max_tiles, encoder=a.encoder, feats2=a.feats2, views=a.views,
                          drop=atlas_drop(a.atlas, a.drop_clusters))
    print(f"loaded {len(cases)} cases, {int(E.sum())} events, {sum(len(x) for x in X)} tiles in {time.time() - t0:.0f}s", flush=True)
    if a.ctx:
        Xc = load_ctx(a.ctx, cases)
        X = Xc if a.ctx_only else list(zip(X, Xc))
        print(f"context tiles {sum(len(x) for x in Xc)} ({'only' if a.ctx_only else 'two-scale'})", flush=True)
    json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    y = np.array([(bool(e), t) for e, t in zip(E, T)], dtype=[("e", bool), ("t", float)])
    n1 = len(cases)
    if a.add_feats:  # second cohort appended after the first, so first-cohort indices and folds stay unchanged
        assert a.model == "abmil" and a.loss == "cox" and not a.ctx and not a.final
        acases, Xa, Ta, Ea = load(a.add_feats, a.add_labels, a.max_tiles, encoder=a.encoder)
        sites = np.array([c.split("-")[1] for c in acases])
        print(f"added cohort: {len(acases)} cases, {int(Ea.sum())} events, {len(set(sites))} sites, "
              f"{'trained on' if a.add_train else 'evaluated only'}", flush=True)
        Xall, Tall, Eall = X + Xa, np.r_[T, Ta], np.r_[E, Ea]
        S = np.r_[np.zeros(n1, int), np.ones(len(acases), int)] if a.add_train else None

    if a.final:
        all_idx = np.arange(len(cases))
        risks = []
        for s in range(a.seeds):
            m = train_abmil(X, T, E, all_idx, None, a, dev, seed=1000 + s)
            torch.save(m.state_dict(), os.path.join(a.out, f"model_s{s}.pt"))
            risks.append(predict(m, X, all_idx, dev))
        r = np.mean(risks, 0)
        ts, H0 = breslow(r, T, E)
        pt = rmst(r, ts, H0, TAU)
        cal = {"encoder": a.encoder, "tau": TAU, "ts": ts.tolist(), "H0": H0.tolist(), "seeds": a.seeds,
               "time_tertiles": np.quantile(pt, [1 / 3, 2 / 3]).tolist(),
               "train_risk_quantiles": np.quantile(r, np.linspace(0, 1, 101)).tolist(),
               "train_c_index_insample": cidx(E, T, r)}
        mpp = tile_mpp(a.feats, cases)
        if mpp is not None:  # the container tiles at this spacing (inference.py reads "tile_mpp"; absent = 0.97)
            cal["tile_mpp"] = mpp
            cal["max_tiles"] = a.max_tiles  # deploy on the tile budget CV and eval_external score (rule change 2026-11-15)
        json.dump(cal, open(os.path.join(a.out, "calibration.json"), "w"), indent=1)
        print("final done", cal["train_c_index_insample"], flush=True)
        return

    res = {"harrell_pooled": [], "harrell_fold_mean": [], "uno_pooled": []}
    curves = []
    for rep in range(a.repeats):
        oof = np.zeros(len(cases))
        fold_of = np.zeros(len(cases), int)
        fold_c = []
        if a.add_feats:
            afold, aoof = site_folds(sites, Ea, a.folds, seed=rep), np.zeros(len(acases))
        for k, (tr, va) in enumerate(StratifiedKFold(a.folds, shuffle=True, random_state=rep).split(np.zeros(len(E)), E)):
            curve = []
            if a.model == "abmil" and a.add_feats:
                ava = n1 + np.flatnonzero(afold == k)
                trk = np.r_[tr, n1 + np.flatnonzero(afold != k)] if a.add_train else tr
                m = train_abmil(Xall, Tall, Eall, trk, va, a, dev, seed=rep * 100 + k + 10000 * a.seed_offset, curve=curve, S=S)
                aoof[ava - n1] = predict(m, Xall, ava, dev)
                torch.save(m.state_dict(), os.path.join(a.out, f"fold_r{rep}_k{k}.pt"))
                oof[va] = predict(m, X, va, dev)
            elif a.model == "abmil":
                m = train_abmil(X, T, E, tr, va, a, dev, seed=rep * 100 + k + 10000 * a.seed_offset, curve=curve)
                oof[va] = predict(m, X, va, dev)
            else:
                oof[va] = meanpool_fit(X, T, E, tr, va)
            fold_of[va] = k
            c = cidx(E[va], T[va], oof[va])
            fold_c.append(c)
            curves.append(curve)
            print(f"rep {rep} fold {k} C={c:.4f} ({time.time() - t0:.0f}s)", flush=True)
        # pooled OOF C mixes fold-model scales; the per-fold mean is the cleaner number
        res["harrell_pooled"].append(cidx(E, T, oof))
        res["harrell_fold_mean"].append(float(np.mean(fold_c)))
        tau = min(TAU, T[E == 1].max() - 1e-6)
        res["uno_pooled"].append(concordance_index_ipcw(y, y, oof, tau=tau)[0])
        with open(os.path.join(a.out, f"oof_r{rep}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case", "time", "event", "risk", "fold"])
            w.writerows(zip(cases, T, E, oof, fold_of))
        print(f"rep {rep}: pooled {res['harrell_pooled'][-1]:.4f} fold-mean {res['harrell_fold_mean'][-1]:.4f} uno {res['uno_pooled'][-1]:.4f}", flush=True)
        if a.add_feats:  # held-out sites: risks standardized within each fold model before pooling
            z = np.zeros(len(acases))
            for k in range(a.folds):
                f = afold == k
                z[f] = (aoof[f] - aoof[f].mean()) / (aoof[f].std() + 1e-8)
            ya = np.array([(bool(e), t) for e, t in zip(Ea, Ta)], dtype=y.dtype)
            res.setdefault("add_harrell_std", []).append(cidx(Ea, Ta, z))
            res.setdefault("add_fold_mean", []).append(float(np.nanmean(
                [cidx(Ea[afold == k], Ta[afold == k], aoof[afold == k]) for k in range(a.folds)])))
            res.setdefault("add_uno_std", []).append(concordance_index_ipcw(ya, ya, z, tau=min(TAU, Ta[Ea == 1].max() - 1e-6))[0])
            with open(os.path.join(a.out, f"oof_add_r{rep}.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["case", "site", "time", "event", "risk", "z", "fold"])
                w.writerows(zip(acases, sites, Ta, Ea, aoof, z, afold))
            print(f"rep {rep} added cohort, held-out sites: pooled-std {res['add_harrell_std'][-1]:.4f} fold-mean "
                  f"{res['add_fold_mean'][-1]:.4f} uno-std {res['add_uno_std'][-1]:.4f} (fold events "
                  f"{[int(Ea[afold == k].sum()) for k in range(a.folds)]})", flush=True)
    if curves and curves[0]:
        np.savetxt(os.path.join(a.out, "curve.csv"), np.array(curves).T, delimiter=",", fmt="%.4f")
        res["curve_mean_by_epoch"] = np.array(curves).mean(0).round(4).tolist()
    res["summary"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in res.items() if k != "curve_mean_by_epoch"}
    res["n_cases"], res["n_events"] = len(cases), int(E.sum())
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(json.dumps(res["summary"]), flush=True)


if __name__ == "__main__":
    main()
