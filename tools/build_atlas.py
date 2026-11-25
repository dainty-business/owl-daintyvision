"""Cohort atlas of tissue patterns: cluster Midnight tile features and relate each cluster to the deployed model and to
outcome, so reports can say which recurring patterns drive a case and how they behave in an independent cohort.

usage: python build_atlas.py --model-dir pkg\\v1test --feats feats --labels training_labels.csv
       --ext-feats tcga\\feats --ext-labels tcga\\labels.csv --gallery-raw gallery_raw --masks masks --out atlas [--k 16]
Writes <out>/atlas.npz (PCA mean/components, centers), <out>/atlas.json (per-cluster stats) and
<out>/gallery/cluster_XX.png (exemplar tiles nearest each center, from the re-downloaded slides).
Unsupervised: clusters come from image features only; outcome is used afterwards to describe them, never to form them.
"""
import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
import owl_wsi  # noqa: E402
from faithfulness_sim import Head  # noqa: E402

PER_CASE = 256


def labels(path):
    return {r["case_id"].replace(".tif", ""): (float(r["follow_up_years"]), int(r["event"])) for r in csv.DictReader(open(path))}


def cases_of(feats_dir, lab):
    """Yield (case, usable tile features) one case at a time (all tiles of all cases would not fit in memory)."""
    for p in sorted(glob.glob(os.path.join(feats_dir, "*.npz"))):
        case = os.path.basename(p)[:-4]
        if case in lab:
            z = np.load(p)
            yield case, z["f10"][z["w10"] <= owl_wsi.MAX_WHITE].astype(np.float32)


def tile_risk(head, f):
    import torch
    with torch.no_grad():
        x = torch.from_numpy(f).to(head.dev).float()
        return np.mean([m.tiles(x)[1].cpu().numpy() for m in head.mils], 0)


def main():
    ap = argparse.ArgumentParser()
    for k in ("model-dir", "feats", "labels", "ext-feats", "ext-labels", "gallery-raw", "masks", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "gallery"), exist_ok=True)
    import torch
    head = Head(a.model_dir, "cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    lab, ext_lab = labels(a.labels), labels(a.ext_labels)
    sample = np.concatenate([f[np.sort(rng.choice(len(f), min(PER_CASE, len(f)), replace=False))]
                             for _, f in cases_of(a.feats, lab)])
    print(f"fitting on {len(sample)} sampled tiles", flush=True)
    pca = PCA(64, random_state=0).fit(sample)
    km = KMeans(a.k, n_init=4, random_state=0).fit(pca.transform(sample))
    del sample
    np.savez(os.path.join(a.out, "atlas.npz"), pca_mean=pca.mean_, pca_components=pca.components_,
             centers=km.cluster_centers_)

    def assign(f):
        return km.predict(pca.transform(f))

    def assoc(prop, lab_):
        cases = sorted(prop)
        T = np.array([lab_[c][0] for c in cases])
        E = np.array([lab_[c][1] for c in cases]).astype(bool)
        P = np.stack([prop[c] for c in cases])
        return [float(concordance_index_censored(E, T, P[:, j])[0]) for j in range(a.k)]

    # stream training cases: cluster shares per patient and model behaviour per cluster
    n, r_sum, s_sum, top_n, prop_tr = np.zeros(a.k), np.zeros(a.k), np.zeros(a.k), np.zeros(a.k), {}
    for case, f in cases_of(a.feats, lab):
        cl = assign(f)
        _, signed, _ = head(f)
        r = tile_risk(head, f)
        top = signed >= np.quantile(signed, 0.9)
        prop_tr[case] = np.bincount(cl, minlength=a.k) / len(f)
        n += np.bincount(cl, minlength=a.k)
        r_sum += np.bincount(cl, weights=r, minlength=a.k)
        s_sum += np.bincount(cl, weights=signed, minlength=a.k)
        top_n += np.bincount(cl, weights=top.astype(float), minlength=a.k)
    prop_ext = {case: np.bincount(assign(f), minlength=a.k) / len(f) for case, f in cases_of(a.ext_feats, ext_lab)}
    c_tr, c_ext = assoc(prop_tr, lab), assoc(prop_ext, ext_lab)
    stats = []
    for j in range(a.k):
        stats.append(dict(cluster=j, share_of_tiles=float(n[j] / n.sum()), mean_tile_risk=float(r_sum[j] / n[j]),
                          mean_signed_influence_x1e4=float(s_sum[j] / n[j] * 1e4),
                          share_in_top10pct_risk_raising=float(top_n[j] / n[j]),
                          c_index_of_proportion_train=c_tr[j], c_index_of_proportion_tcga=c_ext[j]))
    ext = prop_ext
    json.dump({"k": a.k, "per_case_sample": PER_CASE, "n_train_cases": len(prop_tr), "n_tcga_cases": len(ext),
               "note": "c_index_of_proportion > 0.5: a larger share of this cluster goes with earlier recurrence",
               "clusters": stats}, open(os.path.join(a.out, "atlas.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1), flush=True)

    # exemplar galleries from re-downloaded training slides
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tifffile
    import zarr
    best = {j: [] for j in range(a.k)}  # (distance, slide, gx, gy)
    for slide in sorted(glob.glob(os.path.join(a.gallery_raw, "*.tif"))):
        case = os.path.basename(slide)[:-4]
        z = np.load(os.path.join(a.feats, case + ".npz"))
        keep = z["w10"] <= owl_wsi.MAX_WHITE
        q = pca.transform(z["f10"][keep].astype(np.float32))
        cl = km.predict(q)
        dist = np.linalg.norm(q - km.cluster_centers_[cl], axis=1)
        c = z["c10"][keep]
        for j in range(a.k):
            idx = np.flatnonzero(cl == j)
            for i in idx[np.argsort(dist[idx])[:3]]:  # at most 3 per slide so galleries span patients
                best[j].append((dist[i], slide, *c[i]))
    for j in range(a.k):
        picks = sorted(best[j])[:12]
        fig, axes = plt.subplots(3, 4, figsize=(8, 6.3))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, slide, x0, y0) in zip(axes.flat, picks):
            with tifffile.TiffFile(slide) as tf:
                li, mpps = owl_wsi.level_for(tf, owl_wsi.WANT_MPP)
                arr = zarr.open(tf.aszarr(level=li), mode="r")
                sx = int(round(x0 * mpps[0] / mpps[li]))
                sy = int(round(y0 * mpps[0] / mpps[li]))
                ax.imshow(np.asarray(arr[sy:sy + owl_wsi.TILE, sx:sx + owl_wsi.TILE])[..., :3])
            ax.set_title(os.path.basename(slide)[13:17], fontsize=7)
        st = stats[j]
        fig.suptitle(f"cluster {j}: {100 * st['share_of_tiles']:.1f}% of tiles, tile risk {st['mean_tile_risk']:+.2f}, "
                     f"C(prop) train {st['c_index_of_proportion_train']:.2f} / TCGA {st['c_index_of_proportion_tcga']:.2f}",
                     fontsize=9)
        fig.savefig(os.path.join(a.out, "gallery", f"cluster_{j:02d}.png"), dpi=110)
        plt.close(fig)
    print("ATLAS DONE", flush=True)


if __name__ == "__main__":
    main()
