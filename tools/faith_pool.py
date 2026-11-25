"""Pool tools/faithfulness_sim.py outputs whose z statistics differ (FAITH-004: one file per CV fold, each with z from
its own ~30 cases) by converting every gap to years (gap x that file's z_sd), then compare map rankings per case.

usage: python faith_pool.py --a runs/FAITH-004_r0k*.json --b runs/FAITH-003c_leopard150_v4_cap.json
       (--b optional: a second pooled set on the same cases, e.g. in-sample v4, compared case by case)
       [--pair greedy_own own_check]   (the paired difference to report; default own_check risk_up)
Prints, per set: cases, the share where own_check picked the protective direction, the own_check and risk_up gaps in
years with bootstrap CIs, and the paired own_check - risk_up difference; with --b, the paired a - b difference of each.
"""
import argparse
import glob
import json

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--a", nargs="+", required=True)
ap.add_argument("--b", nargs="*", default=[])
ap.add_argument("--boot", type=int, default=5000)
ap.add_argument("--pair", nargs=2, default=["own_check", "risk_up"])
a = ap.parse_args()
VARIANTS = ("own_check", "risk_up", "risk_down", "attention", "greedy_own")


def pooled(paths):
    rows = {}
    for p in sorted({q for g in paths for q in glob.glob(g)}):
        d = json.load(open(p))
        for r in d["rows"]:
            assert r["case"] not in rows, f"{r['case']} in two files"
            rows[r["case"]] = {v: r["gap_" + v] * d["z_sd"] for v in VARIANTS if "gap_" + v in r}
            rows[r["case"]]["protective"] = r["own_check_protective"] if "own_check_protective" in r else bool(
                np.isclose(r["gap_own_check"], r["gap_risk_down"]) and not np.isclose(r["gap_own_check"], r["gap_risk_up"]))
            rows[r["case"]]["greedy_protective"] = r.get("greedy_protective")
    return rows


def ci(x, rng):
    b = x[rng.integers(0, len(x), (a.boot, len(x)))].mean(1)
    return f"{x.mean():.3f} ({np.percentile(b, 2.5):.3f} to {np.percentile(b, 97.5):.3f})"


rng = np.random.default_rng(0)
sets = {"a": pooled(a.a)} | ({"b": pooled(a.b)} if a.b else {})
for name, rows in sets.items():
    cases = sorted(rows)
    present = [v for v in VARIANTS if all(v in rows[c] for c in cases)]
    g = {v: np.array([rows[c][v] for c in cases]) for v in present}
    print(f"set {name}: {len(cases)} cases, own_check protective in {sum(rows[c]['protective'] for c in cases)}")
    for v in present:
        print(f"  gap {v} (years): {ci(g[v], rng)}")
    x, y = a.pair
    d = g[x] - g[y]
    print(f"  {x} - {y} (years): {ci(d, rng)}; as a share of {x}'s gap {d.mean() / g[x].mean():.3f}; of {y}'s "
          f"{d.mean() / g[y].mean():.3f}; higher in {int((d > 0).sum())} of {len(d)}")
    if "greedy_own" in present:
        gp = [rows[c]["greedy_protective"] for c in cases]
        print(f"  greedy_own protective in {sum(gp)}, same direction as own_check in "
              f"{sum(p == rows[c]['protective'] for p, c in zip(gp, cases))}")
if "b" in sets:
    common = sorted(set(sets["a"]) & set(sets["b"]))
    print(f"a - b on {len(common)} common cases (years):")
    for v in [v for v in VARIANTS if all(v in sets[s][c] for s in "ab" for c in common)]:
        print(f"  {v}: {ci(np.array([sets['a'][c][v] - sets['b'][c][v] for c in common]), rng)}")
    flips = sum(sets["a"][c]["protective"] != sets["b"][c]["protective"] for c in common)
    print(f"  own_check direction differs between a and b in {flips} cases")
