"""Offline replica of the organizers' faithfulness test (owl-challenge faithfulness_blackbox.ipynb, 2026-11-05) on cached
features, to choose how the attribution map ranks tiles.

Removing mask patches only changes which tiles pass the >= 50% tissue rule (tile pixels, hence features, are unchanged),
so each perturbed prediction is the MIL head on the surviving cached tiles: exact for slides under the tile cap.
Replicated from the notebook: 256 px patches at 0.5 um/px on the mask grid from (0, 0), patches with >= 10% tissue,
per-patch median of the map, stable sort after a TIE_SEED shuffle, ceil(pct * n / 100) patches per step, cumulative
random orders for seeds 33/79/98, y = |dz| with z from all unperturbed predictions (ddof 1), trapezoid AUC over linear
percent divided by the x range, gap = AUC_map - mean AUC_random.

With --max-tiles (a binding deployment cap, rule change 2026-11-15) the features must store cells beyond the cap
(extract_tcga --max-tiles 28000): after each mask edit the replica takes the container's max_tiles lowest-key eligible
cells, then drops near-white tiles; a step needing a cell that is not stored is counted as inexact.

usage: python faithfulness_sim.py --feats tcga\\feats --meta tcga\\meta --masks tcga\\masks --model-dir pkg\\v1test
       --out runs\\FAITH-001.json [--max-cases N]
       [--max-tiles 8192 --z-json runs\\EXT-007_mpp0485_s10.json --subset-labels tcga\\labels.csv]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
import owl_wsi  # noqa: E402
from inference import heat_array, map_values  # noqa: E402
from owl_mil import AdditiveABMIL, rmst  # noqa: E402

PCTS = [1, 2, 4, 8, 16, 32, 64]
VARIANTS = ["direction", "direction_linear", "abs_signed", "attention", "risk_up", "risk_down", "own_check"]
SEEDS = (33, 79, 98)
TIE_SEED = 0


def find_tissue_patches(tissue, tile, min_fraction=0.10):
    h, w = tissue.shape
    counts = np.add.reduceat(np.add.reduceat(tissue.astype(np.int64), np.arange(0, h, tile), axis=0),
                             np.arange(0, w, tile), axis=1)
    rows, cols = np.nonzero((counts > 0) & (counts >= min_fraction * tile ** 2))
    return rows, cols


def patch_medians(values, tissue, rows, cols, tile):
    out = np.empty(len(rows))
    for i, (r, c) in enumerate(zip(rows, cols)):
        win = (slice(r * tile, (r + 1) * tile), slice(c * tile, (c + 1) * tile))
        out[i] = np.median(values[win][tissue[win]])
    return out


def n_remove(pct, n):
    return math.ceil(round(pct * n / 100, 6))


def attribution_order(medians):
    shuffled = np.random.default_rng(TIE_SEED).permutation(len(medians))
    return shuffled[np.argsort(-medians[shuffled], kind="stable")]


def curve_auc(values):
    x = np.r_[0.0, np.asarray(PCTS) / 100]
    y = np.r_[0.0, np.asarray(values)]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))


class Head:
    def __init__(self, model_dir, dev):
        self.dev = dev
        self.mils = []
        for p in sorted(glob.glob(os.path.join(model_dir, "mil", "model_s*.pt"))):
            sd = torch.load(p, map_location=dev)
            m = AdditiveABMIL(d_in=sd["embed.0.weight"].shape[0]).to(dev).eval()
            m.load_state_dict(sd)
            self.mils.append(m)
        cal = json.load(open(os.path.join(model_dir, "mil", "calibration.json")))
        self.ts, self.H0, self.tau, self.median_risk = np.array(cal["ts"]), np.array(cal["H0"]), cal["tau"], cal["train_risk_quantiles"][50]

    @torch.no_grad()
    def __call__(self, f):
        x = torch.from_numpy(f).to(self.dev).float()
        risk, signed, att = [], 0, 0
        for m in self.mils:
            rk, a, r = m(x)
            risk.append(rk.item())
            signed = signed + (a * (r - rk) / (1 - a.clamp(max=0.999))).cpu().numpy()
            att = att + a.cpu().numpy()
        n = len(self.mils)
        return float(np.mean(risk)), signed / n, att / n

    def years(self, risk):
        return float(rmst(risk, self.ts, self.H0, self.tau)[0])

    @torch.no_grad()
    def tiles(self, f):
        """Per-member attention logits and tile risks, (members, tiles) float64."""
        x = torch.from_numpy(f).to(self.dev).float()
        out = [m.tiles(x) for m in self.mils]
        return (torch.stack([o[0] for o in out]).double().cpu().numpy(),
                torch.stack([o[1] for o in out]).double().cpu().numpy())


def greedy_score(logit, r, sign, steps=500):
    """FAITH-005: exact greedy removal order of the ensemble's softmax-attention bags. Each step removes the remaining
    ~N/steps tiles whose single removal moves the mean member risk furthest in `sign`'s direction, then renormalizes.
    Returns a score that ranks tiles by removal step (earlier = higher)."""
    w = np.exp(logit - logit.max(1, keepdims=True))
    n = w.shape[1]
    k, alive, score, pos = max(1, -(-n // steps)), np.ones(n, bool), np.empty(n), 0
    while pos < n:
        idx = np.flatnonzero(alive)
        wi, ri = w[:, idx], r[:, idx]
        rest = wi.sum(1, keepdims=True) - wi
        with np.errstate(divide="ignore", invalid="ignore"):
            g = sign * (((wi * ri).sum(1, keepdims=True) - wi * ri) / rest).mean(0)
        g[~np.isfinite(g)] = -np.inf  # removing it would empty a member's bag
        take = idx[np.argsort(-g, kind="stable")[:k]]
        score[take] = n - pos - np.arange(len(take))
        alive[take] = False
        pos += len(take)
    return score


def tile_removal_auc(head, f, score, base_years, sd):
    """Our own cheap check: remove the top pct% tiles by score (tile level, no mask)."""
    order = np.argsort(-score, kind="stable")
    ys = []
    for pct in PCTS:
        keep = np.sort(order[n_remove(pct, len(f)):])
        ys.append(abs(head.years(head(f[keep])[0]) - base_years) / sd if len(keep) else 0.0)
    return curve_auc(ys)


def main():
    ap = argparse.ArgumentParser()
    for k in ("feats", "meta", "masks", "model-dir", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--max-cases", type=int, default=0)
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="the deployed tile cap (calibration max_tiles); --feats must store cells beyond it")
    ap.add_argument("--z-json", default="", help="eval_external output: z statistics from all its per-case risks")
    ap.add_argument("--subset-labels", default="", help="labels.csv: --feats must hold exactly faith_subset's cases")
    ap.add_argument("--cases", default="", help="file of case ids: run only these (applied after --max-cases)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS + ["greedy_own"])
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    head = Head(a.model_dir, dev)
    cases = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.feats, "*.npz")))
    if a.subset_labels:
        from faith_subset import subset
        assert set(cases) == subset(a.subset_labels), "--feats does not hold exactly the fixed subset"
    if a.max_cases:
        cases = cases[:a.max_cases]
    if a.cases:
        keep = set(open(a.cases).read().split())
        cases = [c for c in cases if c in keep]

    # pass 1: unperturbed predictions for the z statistics (the notebook uses all of predictions.csv)
    data, base = {}, {}
    for case in cases:
        z = np.load(os.path.join(a.feats, case + ".npz"))
        meta = json.load(open(os.path.join(a.meta, case + ".json")))
        if "mpp" not in meta:  # tools/extract_features.py layout (LEOPARD at 0.97 um/px)
            meta = {**meta["10"], "mpp0": meta["levels"][0][2]}
        f, w = z["f10"], z["w10"]
        g = np.round(z["c10"] / (owl_wsi.TILE * meta["mpp"] / meta["mpp0"])).astype(int)
        flat = g[:, 1] * meta["grid"][1] + g[:, 0]
        key = np.random.default_rng(0).random(meta["grid"]).ravel()  # owl_wsi.iter_tiles: one fixed key per cell
        dep = np.argsort(key[flat], kind="stable")[:a.max_tiles] if a.max_tiles else np.arange(len(f))
        dep = np.sort(dep[w[dep] <= owl_wsi.MAX_WHITE])
        data[case] = (f, g, w, flat, key, dep, meta)
        base[case] = head.years(head(f[dep])[0])
    yrs = np.array(list(base.values()))
    if a.z_json:  # the notebook's z uses every prediction, not only this subset's
        ext = json.load(open(a.z_json))
        zy = dict(zip(ext["cases"], (head.years(r) for r in ext["risk"])))
        yrs = np.array(list(zy.values()))
    z_mean, z_sd = float(yrs.mean()), float(yrs.std(ddof=1))
    print(f"{len(cases)} cases, z from {len(yrs)} predictions: years mean {z_mean:.3f} sd {z_sd:.3f}", flush=True)
    if a.z_json:
        diff = max(abs(zy[c] - base[c]) for c in cases)
        print(f"largest base difference from {a.z_json}: {diff:.5f} years", flush=True)
        assert diff < 0.01 * z_sd, "base predictions differ from eval_external's (the notebook's repeat tolerance)"

    variants = a.variants
    rows = []
    for ci, case in enumerate(cases):
        fs, gs, w, flat, key, dep, meta = data[case]
        mask_fp = next(p for p in (os.path.join(a.masks, case + ".tif"), os.path.join(a.masks, case + "_tissue.tif"))
                       if os.path.exists(p))
        mask, mpp_mask = owl_wsi.read_mask(mask_fp)
        tissue = mask > 0
        factor = round(0.5 / meta["mpp0"])
        tile = int(round(256 * factor * meta["mpp0"] / mpp_mask))
        prow, pcol = find_tissue_patches(tissue, tile)
        cell = owl_wsi.TILE * meta["mpp"] / mpp_mask
        gh, gw = meta["grid"]
        pos = {int(k): i for i, k in enumerate(flat)}
        steps = [0, 0]  # perturbed predictions; of those, inexact (a cell the container would encode is not stored)

        def select(m):
            """Stored rows the container uses for this mask (inference.predict_case), and its cells not stored."""
            frac = owl_wsi.tissue_grid(m, mpp_mask, meta["mpp"], (gh * owl_wsi.TILE, gw * owl_wsi.TILE))
            for thr in (owl_wsi.MIN_TISSUE, 0.01):  # the container retries at 1% tissue when no cell passes 50%
                cells = np.flatnonzero(frac >= thr)
                if len(cells):
                    break
            if a.max_tiles and len(cells) > a.max_tiles:
                cells = cells[np.argsort(key[cells], kind="stable")[:a.max_tiles]]
            idx = np.array([pos[c] for c in cells if c in pos], int)
            keep = idx[w[idx] <= owl_wsi.MAX_WHITE]
            return np.sort(keep if len(keep) else idx), len(cells) - len(idx)

        sel0, miss0 = select(mask)
        assert miss0 == 0 and np.array_equal(sel0, dep), f"{case}: the mask does not reproduce the stored tile set"

        def predict(removed):
            m = mask.copy()
            for r, cc in zip(prow[removed], pcol[removed]):
                m[r * tile:(r + 1) * tile, cc * tile:(cc + 1) * tile] = 0
            sel, missing = select(m)
            steps[0] += 1
            steps[1] += missing > 0
            # no cell left: the container would fail this run; the notebook counts it as failed
            return abs(head.years(head(fs[sel])[0]) - base[case]) / z_sd if len(sel) else 0.0

        f, g = fs[dep], gs[dep]  # the deployed tiles carry the map
        risk, signed, att = head(f)
        d = 1.0 if risk >= head.median_risk else -1.0
        own = max((d * signed, -d * signed), key=lambda s: tile_removal_auc(head, f, s, base[case], z_sd))
        g_prot = None
        if "greedy_own" in variants:  # protective-first: removals raise the risk; risk-first: they lower it
            lg, rg = head.tiles(f)
            g_prot, g_risk = greedy_score(lg, rg, 1.0), greedy_score(lg, rg, -1.0)
            first = np.argsort(-g_prot)[:max(1, -(-len(f) // 500))]
            rest = np.setdiff1d(np.arange(len(f)), first)  # names differ from select()'s closure variables (w)
            ew = np.exp(lg - lg.max(1, keepdims=True))[:, rest]
            got = ((ew * rg[:, rest]).sum(1) / ew.sum(1)).mean()
            assert abs(got - head(f[rest])[0]) < 1e-4, f"{case}: greedy's risk bookkeeping does not match the model"
            prot_auc, risk_auc = (tile_removal_auc(head, f, s, base[case], z_sd) for s in (g_prot, g_risk))
        scores = {"direction": d * signed, "direction_linear": d * signed, "abs_signed": np.abs(signed),
                  "attention": att, "risk_up": signed, "risk_down": -signed, "own_check": own}
        if g_prot is not None:
            scores["greedy_own"] = g_prot if prot_auc >= risk_auc else g_risk
        row = {"case": case, "n_tiles": len(f), "n_stored": len(flat), "n_patches": len(prow), "tile_mask_px": tile,
               "years": base[case], "own_check_protective": bool(np.array_equal(own, -signed))}
        if g_prot is not None:
            row["greedy_protective"] = bool(prot_auc >= risk_auc)
        rand = []
        for seed in SEEDS:
            perm = np.random.default_rng(seed).permutation(len(prow))
            rand.append(curve_auc([predict(perm[:n_remove(p, len(prow))]) for p in PCTS]))
        row["auc_random"] = float(np.mean(rand))
        row["auc_random_seed33"] = rand[0]
        for v in variants:
            vals = map_values(scores[v]) if v != "direction_linear" else (
                1 + np.round(254 * np.argsort(np.argsort(scores[v])) / max(1, len(f) - 1))).astype(np.uint8)
            heat = heat_array(mask.shape, g, vals, cell)
            order = attribution_order(patch_medians(heat, tissue, prow, pcol, tile))
            row["auc_" + v] = curve_auc([predict(order[:n_remove(p, len(prow))]) for p in PCTS])
            row["gap_" + v] = row["auc_" + v] - row["auc_random"]
        row["steps"], row["inexact_steps"] = steps
        rows.append(row)
        if ci % 20 == 0:
            print(ci, {k: round(np.mean([r[k] for r in rows]), 4) for k in rows[0] if k.startswith("gap_")}, flush=True)

    rng = np.random.default_rng(0)
    summary = {}
    for v in variants:
        gaps = np.array([r["gap_" + v] for r in rows])
        boot = gaps[rng.integers(0, len(gaps), (2000, len(gaps)))].mean(1)
        summary[v] = {"gap_mean": float(gaps.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                      "auc_mean": float(np.mean([r["auc_" + v] for r in rows]))}
    summary["auc_random_mean"] = float(np.mean([r["auc_random"] for r in rows]))
    summary["inexact_steps"] = [int(sum(r["inexact_steps"] for r in rows)), int(sum(r["steps"] for r in rows))]
    json.dump({"summary": summary, "z_mean": z_mean, "z_sd": z_sd, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
