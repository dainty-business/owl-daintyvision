"""Per-case PDF interpretability report (4 pages, matplotlib only)."""
import os
import re
import textwrap
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import zarr  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

import owl_wsi  # noqa: E402

A4 = (8.27, 11.69)


def _text(fig, y, s, size=9, width=105, x=0.07, **kw):
    for para in s.split("\n"):
        for line in textwrap.wrap(para, width) or [""]:
            fig.text(x, y, line, fontsize=size, va="top", **kw)
            y -= 0.0165 * size / 9
    return y


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _rank_phrase(risk, q, pct):
    if risk > q[-1]:
        return "higher than every one of the 508 training patients"
    if risk < q[0]:
        return "lower than every one of the 508 training patients"
    return f"at the {_ordinal(pct)} percentile of the 508 training patients"


def _removal_curve(model, f, score):
    """|change in predicted years| at each PCTS step when the top-ranked tiles by score are removed, and its area: the
    same computation as inference.removal_auc, which chose the map's direction."""
    from inference import PCTS
    base = model.time_of(model.mil(f)[0])
    order = np.argsort(-score, kind="stable")
    ys = [abs(model.time_of(model.mil(f[np.sort(order[min(len(f) - 1, int(np.ceil(p * len(f) / 100))):])])[0]) - base)
          for p in PCTS]
    x, y = np.r_[0.0, np.asarray(PCTS) / 100], np.r_[0.0, ys]
    return list(PCTS), ys, float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))


def _member_terms(model, f):
    """Each ensemble member's own predicted time, and the members' mean attention weight and tile risk per tile."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(f)).to(next(model.mils[0].parameters()).device).float()
    years, att, rsk = [], 0.0, 0.0
    with torch.no_grad():
        for m in model.mils:
            rk, a, r = m(x)
            years.append(model.time_of(rk.item()))
            att, rsk = att + a.cpu().numpy(), rsk + r.cpu().numpy()
    return years, att / len(model.mils), rsk / len(model.mils)


def _steps(pcts, a, b):
    """Removal steps (%) at which curve a is above b, and at which it is not, as short text lists."""
    win = [str(p) for p, u, v in zip(pcts, a, b) if u > v]
    lose = [str(p) for p, u, v in zip(pcts, a, b) if u <= v]
    return ", ".join(win) or "none", ", ".join(lose) or "none"


def _mean_upto(pcts, ys, top):
    """Mean |change| over removal steps up to top% (trapezoids over linear percent, as the 1-64% mean)."""
    keep = [i for i, p in enumerate(pcts) if p <= top]
    x, y = np.r_[0.0, [pcts[i] / 100 for i in keep]], np.r_[0.0, [ys[i] for i in keep]]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / (x[-1] - x[0]))


def _step_text(pcts, a, b):
    win, lose = _steps(pcts, a, b)
    if lose == "none":
        return "at every step"
    if win == "none":
        return "at no single step (only the average differs)"
    return f"at {win}% removed but not at {lose}%"


def _where(gs, ga):
    """Rough position of tiles gs within the tissue bounding box of ga, in words."""
    (x0, y0), (x1, y1) = ga.min(0), ga.max(0)
    cx, cy = (gs[:, 0].mean() - x0) / max(1, x1 - x0), (gs[:, 1].mean() - y0) / max(1, y1 - y0)
    v = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else ""
    h = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else ""
    return f"{v} {h}".strip() + " part" if (v or h) else "middle"


def _noise_note(ctr, ctc, risk):
    """Short note for a pattern row; within noise = |C - 0.5| below 0.06 (508 training patients) or 0.08 (298 TCGA)."""
    in_tr, in_tc = abs(ctr - 0.5) < 0.06, abs(ctc - 0.5) < 0.08
    if in_tr and in_tc:
        return "no clear association"
    notes = [f"TCGA only ({ctc:.2f})"] if in_tr else ["not replicated in TCGA"] if in_tc else []
    if not in_tr and not in_tc:
        notes.append("replicated in TCGA" if (ctr - 0.5) * (ctc - 0.5) > 0 else "cohorts disagree")
    if not in_tr and (risk > 0) != (ctr > 0.5):
        notes.append("training link vs tile risk opposite")
    return "; ".join(notes) or "consistent"


def _off_tissue(mask, gi, cell):
    """Share of a tile outside the provided tissue mask (glass or background at the tissue edge)."""
    if mask is None:
        return 0.0
    x0, y0 = int(gi[0] * cell), int(gi[1] * cell)
    m = np.asarray(mask[y0:int((gi[1] + 1) * cell) + 1, x0:int((gi[0] + 1) * cell) + 1])
    return float(1 - (m > 0).mean()) if m.size else 0.0


def _art_dir(cl, art, signed, short=False):
    """Which way the artefact-pattern tiles move this slide's score, and how many sit in each 10% set."""
    t = art[cl]
    n10 = max(1, len(signed) // 10)
    n_up, n_dn = int(t[np.argsort(-signed)[:n10]].sum()), int(t[np.argsort(signed)[:n10]].sum())
    way = "raise" if signed[t].mean() > 0 else "lower"
    if short:
        return f"In this slide they {way} the score on average\n({n_up} in the 10% most raising, {n_dn} in the 10% most lowering)"
    return (f"In this slide they {way} the score on average ({n_up} of them among the 10% most risk-raising tiles, "
            f"{n_dn} among the 10% most risk-lowering).")


def _grid_map(g, v, shape, fill=np.nan):
    out = np.full(shape, fill, np.float32)
    out[g[:, 1], g[:, 0]] = v
    return out


def assign_clusters(atlas, f):
    """Nearest atlas cluster per tile (PCA projection, then nearest k-means center)."""
    q = (f.astype(np.float32) - atlas["mean"]) @ atlas["comps"].T
    return np.argmin(((q[:, None, :] - atlas["centers"][None]) ** 2).sum(-1), 1)


def atlas_page(pdf, atlas, cl, infl, signed, direction):
    k = len(atlas["centers"])
    stats = {int(st["cluster"]): st for st in atlas["stats"]}
    desc = atlas["desc"]
    case_share = np.bincount(cl, minlength=k) / len(cl)
    cohort = np.array([stats[j]["share_of_tiles"] for j in range(k)])
    top = np.argsort(-infl)[: max(1, len(infl) // 10)]
    top_share = np.bincount(cl[top], minlength=k) / len(top)
    order = np.argsort(-np.array([stats[j]["mean_tile_risk"] for j in range(k)]))
    fig = plt.figure(figsize=A4)
    fig.text(0.07, 0.96, "Tissue patterns (unsupervised clusters) in this slide", fontsize=13, weight="bold")
    ax = fig.add_axes([0.38, 0.64, 0.58, 0.26])
    yy = np.arange(k)
    ax.barh(yy - 0.2, cohort[order], 0.4, color="#bbbbbb", label="training cohort")
    ax.barh(yy + 0.2, case_share[order], 0.4, color="#2a78d6", label="this slide")
    first, second = ("risk-raising", "risk-lowering") if direction > 0 else ("risk-lowering", "risk-raising")
    other = np.argsort(-infl)[::-1][: max(1, len(infl) // 10)]  # the 10% most extreme tiles in the other direction
    other_share = np.bincount(cl[other], minlength=k) / len(other)
    # circles = risk-raising set, squares = risk-lowering set, as on page 1; patterns absent from a set get no marker
    for share, kind, label in ((top_share, first, f"this slide's 10% top-ranked map tiles ({first})"),
                               (other_share, second, f"this slide's 10% most {second} tiles")):
        on = share[order] > 0
        ax.scatter(share[order][on], yy[on], marker="o" if kind == "risk-raising" else "s", s=20, facecolors="none",
                   edgecolors="black", linewidths=0.9, zorder=3, label=label)
    name = {j: re.sub(r" \([a-z]\)$", "", desc[str(j)]["name"]) for j in range(k)}  # (a)-(d) suffixes clash with cN
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{name[j]} (c{j})" for j in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("share of tiles", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    fig.text(0.07, 0.594, "Patterns are sorted by the model's mean tile risk across the training cohort (top = raises risk).",
             fontsize=7.5)

    art = np.array([desc[str(j)].get("group") == "artefact" for j in range(k)])
    rows, listed = [], []
    for set_name, share in ((f"map's top 10%\n({first})", top_share), (f"10% most\n{second}", other_share)):
        for n, j in enumerate(np.argsort(-share)[:3]):
            if n and share[j] < 0.1:
                continue
            st = stats[j]
            ctr, ctc = st["c_index_of_proportion_train"], st["c_index_of_proportion_tcga"]
            rows.append([set_name if n == 0 else "", f"c{j} {name[j]}", f"{100 * share[j]:.0f}%",
                         f"{100 * case_share[j]:.1f}% ({100 * cohort[j]:.1f}%)", f"{st['mean_tile_risk']:+.2f}",
                         f"{ctr:.2f}", f"{ctc:.2f}",
                         ("base rate; " if share[j] < 1.3 * case_share[j] else "")
                         + _noise_note(ctr, ctc, st["mean_tile_risk"])])
            listed.append(j)
    fig.text(0.07, 0.572, "What the highlighted tiles are", fontsize=8.5, weight="bold")
    tax = fig.add_axes([0.04, 0.37, 0.92, 0.19])
    tax.axis("off")
    tb = tax.table(cellText=rows, colLabels=["set", "pattern", "share\nof set", "share of slide\n(cohort)",
                                             "cohort\ntile risk", "C\ntraining", "C\nTCGA", "note"],
                   colWidths=[0.11, 0.31, 0.06, 0.11, 0.07, 0.065, 0.065, 0.20], loc="upper center", cellLoc="left")
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.0)
    for (r_, c_), cell in tb.get_celld().items():
        cell.set_height(0.16 if r_ == 0 else 0.13)
        cell.set_linewidth(0.3)
    a_share = float(case_share[art].sum())
    descr = [f"c{j} {name[j]}: {desc[str(j)]['description'].rstrip('.')}." for j in dict.fromkeys(listed + [int(cl[np.argmax(signed)]), int(cl[np.argmin(signed)])])]
    y = _text(fig, 0.56 - 0.19 * (0.16 + 0.13 * len(rows)) - 0.018, "\n".join(descr) + (
        f"\nArtefact patterns (c11, c12) make up {100 * a_share:.1f}% of this slide's tissue. {_art_dir(cl, art, signed)} "
        "Their direction depends on the slide's own score (see below); discount them wherever they appear (page 5 shows "
        "where they are)." if art.any() else "")
        + "\nEach set lists its largest pattern and any other with at least 10% of its tiles (at most three); the eight "
        "tiles on page 2 are the most extreme of each set, so their patterns can differ from the set as a whole.",
        size=6.6, width=142)
    _text(fig, y - 0.008, (
        "How this page is made. The 16 patterns are clusters of tile features from all training slides, formed without "
        "outcome or annotations. Their names were written from example tiles by an AI assistant and are not verified by a "
        "pathologist: they describe appearance and are not diagnoses. Outcome statistics are descriptive, not causal. Tile "
        "risk is in the model's log-risk units; the slide risk is the attention-weighted mean of tile risks, so a tile "
        "raises or lowers this slide's score relative to the slide's own risk. C-index of a pattern's share: 0.5 = no "
        "association, above 0.5 = a larger share goes with earlier recurrence. Values within 0.06 (508 training patients) "
        "or 0.08 (298 TCGA patients) of 0.5 count as noise; 'replicated in TCGA' = beyond noise in both cohorts in the "
        "same direction, 'not replicated in TCGA' = in training only, 'TCGA only' = in TCGA only, 'base rate' = no more "
        "common in the set than in the slide. Training values are optimistic (the same patients trained the model); TCGA "
        "values are external."
        ), size=6.6, width=142)
    pdf.savefig(fig)
    plt.close(fig)


def make(path, *, wsi, model, f, g, c, signed, infl, direction, risk, years, klass, meta, curve, mask, cell):
    cl = assign_clusters(model.atlas, f) if getattr(model, "atlas", None) else None
    gh, gw = meta["grid"]
    q = np.array(model.cal["train_risk_quantiles"])
    pct = int(np.clip(np.searchsorted(q, risk), 0, 100))
    fracs, _, rand = curve  # its top-ranked curve is redrawn by _removal_curve, the computation that chose the direction
    pcts, ys_map, auc_map = _removal_curve(model, f, infl)
    _, ys_other, auc_other = _removal_curve(model, f, -infl)
    m16, o16 = _mean_upto(pcts, ys_map, 16), _mean_upto(pcts, ys_other, 16)
    xr, yr = np.r_[0.0, fracs], np.r_[0.0, rand]
    auc_rand = float(np.sum(np.diff(xr) * (yr[1:] + yr[:-1]) / 2) / (xr[-1] - xr[0]))
    first, second = ("risk-raising", "risk-lowering") if direction > 0 else ("risk-lowering", "risk-raising")

    with tifffile.TiffFile(wsi) as tf:
        thumb = owl_wsi.thumbnail(tf)
        tile_arr = zarr.open(tf.aszarr(level=meta["level"]), mode="r")
        T = meta.get("src", owl_wsi.TILE)  # source px per tile at meta["level"]

        def tile(i):
            gx, gy = g[i]
            return np.asarray(tile_arr[gy * T:(gy + 1) * T, gx * T:(gx + 1) * T])[..., :3]

        hi_idx = np.argsort(-signed)[:8]
        lo_idx = np.argsort(signed)[:8]
        hi_tiles, lo_tiles = [tile(i) for i in hi_idx], [tile(i) for i in lo_idx]

        def context(i, r=2):
            """~(2r+1) tiles square around tile i, downsampled to <= 600 px; the tile's box and px per um."""
            gx, gy = g[i]
            ya, yb = max(0, (gy - r) * T), min(tile_arr.shape[0], (gy + r + 1) * T)
            xa, xb = max(0, (gx - r) * T), min(tile_arr.shape[1], (gx + r + 1) * T)
            step = max(1, max(yb - ya, xb - xa) // 600)
            crop = np.asarray(tile_arr[ya:yb:step, xa:xb:step])[..., :3]
            full = len(range(0, (2 * r + 1) * T, step))  # pad edge crops with white so every panel has one size
            canvas = np.full((full, full, 3), 255, np.uint8)
            oy, ox = (ya - (gy - r) * T) // step, (xa - (gx - r) * T) // step
            canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop[:full - oy, :full - ox]
            return (canvas, (r * T / step, r * T / step, T / step), T / (owl_wsi.TILE * meta["mpp"]) / step,
                    (oy, ox, min(crop.shape[0], full - oy), min(crop.shape[1], full - ox)))

        ctx_hi, ctx_lo = [context(i) for i in hi_idx[:3]], [context(i) for i in lo_idx[:3]]

    ext = [0, gw, gh, 0]
    with PdfPages(path) as pdf:
        # page 1: prediction and maps
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.96, "OWL biochemical-recurrence report (research use only)", fontsize=14, weight="bold")
        y = _text(fig, 0.94, f"Slide: {os.path.splitext(os.path.basename(wsi))[0]}.  Model: DaintyVision v4.7, "
                  f"{len(model.mils)}-member ensemble.", size=8, width=105)
        tau = model.cal["tau"]
        t1, t2 = model.cal["time_tertiles"]
        member_years, att, trisk = _member_terms(model, f)
        votes = Counter(model.risk_class(t) for t in member_years)
        agree = f"{votes[klass]} of {len(member_years)} ensemble members give {klass.upper()}" + (
            "" if votes[klass] == len(member_years) else
            " (" + ", ".join(f"{n} {k.upper()}" for k, n in votes.items() if k != klass) + ")")
        y = _text(fig, y - 0.004, (
            f"Predicted BCR-free time: {years:.2f} years (restricted mean over {tau:.0f} years: the expected number of "
            f"years without BCR in the first {tau:.0f} after surgery, so it cannot exceed {tau:.0f}).  Risk class: "
            f"{klass.upper()}; classes are the thirds of the model's predicted times on the training cohort: high below "
            f"{t1:.2f} years, low above {t2:.2f}.  Model risk score "
            f"{risk:+.2f}, {_rank_phrase(risk, q, pct)} (higher = earlier predicted recurrence)."), size=9, width=116)
        n10 = max(1, len(signed) // 10)
        up10, dn10 = np.argsort(-signed)[:n10], np.argsort(signed)[:n10]

        def _cluster_note(idx, word):
            d = np.sqrt(((g[idx] - g[idx[0]]).astype(float) ** 2).sum(-1)) * owl_wsi.TILE * meta["mpp"] / 1000
            near = [i for i, di in zip(idx, d) if di <= 1.5]
            if len(near) < 5:
                return ""
            edge = sum(_off_tissue(mask, g[i], cell) >= 0.05 for i in near)
            count = "All eight" if len(near) == 8 else f"{['', '', '', '', '', 'Five', 'Six', 'Seven'][len(near)]} of the eight"
            sign = "+" if word == "raise" else "-"
            return (f" {count} strongest tiles that {word} the score lie within 1.5 mm of tile {sign}1"
                    + (f" ({edge} at the tissue edge)" if edge else "") + ", so that evidence comes mostly from one area.")

        def _sampling_note():
            """When one pattern dominates the risk-raising tiles and is far above its cohort share, say so."""
            if cl is None:
                return ""
            j = int(np.bincount(cl[up10]).argmax())
            cs = float(np.mean(cl == j))
            co = {int(st["cluster"]): st for st in model.atlas["stats"]}[j]["share_of_tiles"]
            head = f" Pattern c{j} makes up {100 * cs:.1f}% of this slide's tissue (training average {100 * co:.1f}%)"
            if cs >= 3 * co:
                return (head + "; since the slide risk is a weighted mean of tile risks, this share "
                        + ("largely sets" if cs >= 0.5 else "strongly influences") + " the score.")
            return head + (", more than average." if cs >= 1.5 * co else ", less than average." if cs <= co / 1.5
                           else ", about average.")

        def dom(idx):
            if cl is None:
                return "tissue"
            j = int(np.bincount(cl[idx]).argmax())
            return f"{model.atlas['desc'][str(j)]['name']} (c{j}, {100 * np.mean(cl[idx] == j):.0f}% of them)"

        art_up = 0.0 if cl is None else float(np.mean(
            [model.atlas["desc"][str(j)].get("group") == "artefact" for j in cl[up10]]))

        def _cautions(text):
            return "\nCautions:" + text if text else ""

        def without(order_):  # predicted time when the model is re-run without the first n10 tiles of order_
            return model.time_of(model.mil(f[np.sort(order_[n10:])])[0])

        t_no_up, t_no_dn = without(np.argsort(-signed, kind="stable")), without(np.argsort(signed, kind="stable"))

        def cls_word(t):
            return f"{model.risk_class(t).upper()}" if model.risk_class(t) != klass else f"still {klass.upper()}"

        if klass == "high":
            firm = f"{t1 - years:.2f} years below the HIGH/MEDIUM cut-off ({t1:.2f})"
        elif klass == "low":
            firm = f"{years - t2:.2f} years above the MEDIUM/LOW cut-off ({t2:.2f})"
        elif years - t1 < t2 - years:
            firm = f"{years - t1:.2f} years above the HIGH/MEDIUM cut-off ({t1:.2f})"
        else:
            firm = f"{t2 - years:.2f} years below the MEDIUM/LOW cut-off ({t2:.2f})"
        y = _text(fig, y - 0.006, (
            f"Why {klass.upper()}: " + (
                f"the 10% of tissue that lowers the score most is mostly {dom(dn10)}; the 10% that raises it most is "
                f"mostly {dom(up10)}, in the {_where(g[up10], g)} of the tissue." + _sampling_note()
                + f" Re-running the model without the risk-lowering 10% gives {t_no_dn:.2f} years ({cls_word(t_no_dn)}); "
                f"without the risk-raising 10%, {t_no_up:.2f} years ({cls_word(t_no_up)})."
                if klass == "low" else
                f"the 10% of tissue that raises the score most is mostly {dom(up10)}, in the {_where(g[up10], g)} of "
                f"the tissue; the 10% that lowers it most is mostly {dom(dn10)}." + _sampling_note()
                + f" Re-running the model without the risk-raising 10% gives {t_no_up:.2f} years ({cls_word(t_no_up)}); "
                f"without the risk-lowering 10%, {t_no_dn:.2f} years ({cls_word(t_no_dn)}).")
            + " Pattern names are unverified AI-written descriptions (page 3).\n"
            f"How firm: {agree} (their times: {min(member_years):.2f}-{max(member_years):.2f} years); the predicted "
            f"time is {firm}.\n"
            "Where to look: circles +1 to +8 (raise) and squares -1 to -8 (lower) on the red/blue map; page 2 shows "
            "these tiles, page 5 their surroundings."
            + "".join(_cluster_note(idx_, word) for idx_, word in ((hi_idx, "raise"), (lo_idx, "lower")))
            + _cautions(
               (" The score is outside the range of the training patients, so the predicted time is an extrapolation."
               if risk > q[-1] or risk < q[0] else "")
            + (f" This slide has far less tissue than the training slides ({len(f) * (owl_wsi.TILE * meta['mpp'] / 1000) ** 2:.0f} "
               "mm² against a median of 937), another reason for caution."
               if len(f) * (owl_wsi.TILE * meta["mpp"] / 1000) ** 2 < 477 else "")
            + (" The submitted map's direction is the weaker one at removals up to 16% (page 4)." if m16 < o16 else "")
            + (f" Artefact patterns (ink, debris, blur) make up {100 * art_up:.0f}% of the 10% most risk-raising tiles "
               "(page 5 maps them); discount that part." if art_up >= 0.1 else ""))), size=8.3, width=121)
        top = min(0.83, y - 0.075)
        ax = fig.add_axes([0.05, 0.455, 0.44, top - 0.455])
        ax.imshow(thumb)
        ax.set_title("H&E overview", fontsize=9)
        ax.axis("off")
        ax = fig.add_axes([0.52, 0.455, 0.44, top - 0.455])
        ax.imshow(thumb, extent=ext)
        rk = ax.imshow(_grid_map(g, np.argsort(np.argsort(infl)) / max(1, len(infl) - 1), (gh, gw)), extent=ext,
                       cmap="inferno", alpha=0.65, vmin=0, vmax=1)
        ax.set_title(f"Submitted attribution map: {first} tiles ranked by influence\nbright = ranked first (removal "
                     f"{'lengthens' if direction > 0 else 'shortens'} the predicted time most)\ndark = ranked last, "
                     f"including the tissue pushing hardest the other way", fontsize=8.2)
        ax.axis("off")
        q0 = float(np.mean(infl < 0))  # tiles ranked below q0 push the score against the map's direction
        ticks, labels = ([0, q0, 1], ["ranked last", "no effect", "ranked first"]) if 0.12 < q0 < 0.88 else \
            ([0, 1], ["ranked last", "ranked first"])
        cb = fig.colorbar(rk, cax=fig.add_axes([0.58, 0.443, 0.32, 0.006]), orientation="horizontal", ticks=ticks)
        cb.ax.set_xticklabels(labels, fontsize=6)
        cb.ax.axvline(q0, color="white", linewidth=1.5)
        ax = fig.add_axes([0.05, 0.068, 0.44, 0.332])
        neg, posv = signed[signed < 0], signed[signed > 0]
        lo = -(np.quantile(-neg, 0.99) if len(neg) else 1e-12) - 1e-12
        hi = (np.quantile(posv, 0.99) if len(posv) else 1e-12) + 1e-12
        ax.imshow(thumb, extent=ext)
        im = ax.imshow(_grid_map(g, signed, (gh, gw)), extent=ext, cmap="coolwarm", alpha=0.75,
                       norm=matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=lo, vmax=hi))
        ax.scatter(g[hi_idx, 0] + 0.5, g[hi_idx, 1] + 0.5, s=22, marker="o", facecolors="none", edgecolors="black",
                   linewidths=0.8)
        ax.scatter(g[lo_idx, 0] + 0.5, g[lo_idx, 1] + 0.5, s=22, marker="s", facecolors="none", edgecolors="black",
                   linewidths=0.8)
        placed = []  # numbers match the tiles on page 2 (#1 = strongest); a number that would collide is left out
        for idx_, sign_ in ((hi_idx, "+"), (lo_idx, "-")):
            for n_, i_ in enumerate(idx_):
                p_ = g[i_] + 0.5
                if p_[1] > 0.04 * gh and all(np.hypot(*(p_ - q_)) > 0.05 * max(gw, gh) for q_ in placed):
                    # labels in the top 4% could reach the panel title
                    ax.annotate(f"{sign_}{n_ + 1}", p_, xytext=(3, 2), textcoords="offset points", fontsize=6,
                                weight="bold")
                    placed.append(p_)
        ax.set_xlim(0, gw)
        ax.set_ylim(gh, 0)
        ax.set_title("Red raises this slide's risk score, blue lowers it (same in every report)\ncircle +n / square -n = "
                     "tile #n of page 2 that raises / lowers it", fontsize=8.5)
        ax.axis("off")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, ticks=[lo, 0, hi])
        cb.ax.set_xticklabels(["lowers most", "no effect", "raises most"])
        cb.set_label("tile's contribution to the slide risk score; red and blue are scaled separately\n(the strongest "
                     + (f"red is {hi / -lo:.1f}x the strongest blue)" if hi >= -lo else f"blue is {-lo / hi:.1f}x the strongest red)"),
                     fontsize=6.5)
        cb.ax.tick_params(labelsize=7)
        _text(fig, 0.418, (
            f"Tissue analysed: {len(f)} tiles of 224 x 224 px at {meta['mpp']:.2f} µm/px "
            f"({len(f) * (owl_wsi.TILE * meta['mpp'] / 1000) ** 2:.0f} mm²; training slides: median 937 mm², "
            "5th percentile 477 mm²).\n"
            "Raising and lowering are relative to this slide's own score: tissue riskier than the slide as a whole "
            "raises it.\n\n"
            "Scores across centres: 95% of 298 TCGA-PRAD patients from other centres fell in the high class. They do "
            "recur earlier (56% BCR-free at 5 years, against 86% in training), but their predicted times ran short "
            "(4.4 years on average against an observed 6.7). Compare patients within one centre by risk score."),
            size=7.1, width=58, x=0.55)
        pdf.savefig(fig)
        plt.close(fig)

        # page 2: example tiles
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.96, "Tiles with the largest contributions", fontsize=13, weight="bold")
        um = owl_wsi.TILE * meta["mpp"]
        for row, (tiles, idx, name) in enumerate(((hi_tiles, hi_idx, "raise"), (lo_tiles, lo_idx, "lower"))):
            fig.text(0.07, 0.92 - row * 0.45, f"Eight tiles that most {name} this slide's risk score (~{um:.0f} µm wide; "
                     "value: change in predicted time if that tile alone is removed)", fontsize=8.5)
            for j, (t, i) in enumerate(zip(tiles, idx)):
                ax = fig.add_axes([0.05 + (j % 4) * 0.23, 0.72 - row * 0.45 - (j // 4) * 0.20, 0.21, 0.17])
                ax.imshow(t)
                tag = ""
                if cl is not None:
                    tag = f"  c{cl[i]}" + (" artefact" if model.atlas["desc"][str(cl[i])].get("group") == "artefact" else "")
                off = _off_tissue(mask, g[i], cell)
                if off >= 0.05:
                    tag += f"  {100 * off:.0f}% off-tissue"
                dy = model.time_of(risk - signed[i]) - years  # removing tile i changes the ensemble risk by -signed[i]
                ax.set_title(f"#{j + 1}  {dy * 365.25:+.2f} days{tag}\n@({c[i][0]},{c[i][1]})", fontsize=7)
                ax.axis("off")
        span = round(owl_wsi.TILE * meta["mpp"] / meta["mpp0"])
        _text(fig, 0.064, f"Coordinates: top-left corner in level-0 pixels (x, y); each tile spans {span} x {span} level-0 "
                         f"px. Days: change in predicted time if that tile alone is removed; one tile of {len(f)} moves it "
                         f"little, but together the 10% most risk-raising tiles move it by {abs(t_no_up - years):.2f} years "
                         "(page 1). 'off-tissue': share outside the provided tissue mask, marked from 5%."
                         + (" cN = tissue pattern N (page 3)." if cl is not None else ""), size=7, width=140)
        pdf.savefig(fig)
        plt.close(fig)

        if cl is not None:
            atlas_page(pdf, model.atlas, cl, infl, signed, direction)

        # page 3: faithfulness self-check and method notes
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.96, "Does the map reflect the model? Removal test", fontsize=13, weight="bold")
        ax = fig.add_axes([0.12, 0.60, 0.78, 0.30])
        ax.plot(pcts, ys_map, "o-", label=f"remove the map's top-ranked tiles ({first}): mean {auc_map:.2f} y "
                                          f"(1-16%: {_mean_upto(pcts, ys_map, 16):.2f} y)")
        ax.plot(pcts, ys_other, "^-.", label=f"remove the top {second} tiles instead: mean {auc_other:.2f} y "
                                             f"(1-16%: {_mean_upto(pcts, ys_other, 16):.2f} y)")
        ax.plot([100 * x for x in fracs], rand, "s--", label=f"remove random tiles (mean of 10 draws): mean {auc_rand:.2f} y")
        ax.set_xscale("log", base=2)
        ax.set_xticks(pcts, [str(p) for p in pcts])
        ax.set_xlabel("tissue removed (%, log scale)")
        ax.set_ylabel("|change in predicted time| (years)")
        ax.legend(fontsize=8)
        t1, t2 = model.cal["time_tertiles"]
        encoded = (f"all {meta['n']} were encoded" if meta["n"] == meta["n_candidates"] else
                   f"{meta['n']} were encoded (a fixed, reproducible subset, to bound run time)")
        _text(fig, 0.53, (
            "The removal test re-runs the model on this case with tiles removed in the map's order and compares with "
            "removing the same number of random tiles (random removal takes tissue of both directions, so it is a low "
            "bar). The plot shows the size of the change; the legend gives each curve's mean over 1-64% (the summary "
            "the challenge's faithfulness test uses) and over 1-16%, closer to realistic removals. The map ranks "
            f"{first} tissue first because that direction has the larger 1-64% mean; it moves the prediction more "
            f"{_step_text(pcts, ys_map, ys_other)}. The {tau:.0f}-year limit makes the comparison uneven "
            f"({tau - years:.1f} years of room to lengthen, {years:.1f} to shorten), and removals of a third of the "
            "tissue or more only show the trend.\n\n"
            f"How the map is computed. In each of the {len(model.mils)} ensemble members every tile gets a risk r_i and "
            "an attention weight a_i (the weights sum to 1), and the slide risk R is the weighted sum. In each member, "
            "tile i contributes exactly a_i (r_i - R) / (1 - a_i) to R: removing that one tile changes that member's R "
            "by minus this amount (several removals are not additive, so the test above re-runs the "
            "model). The red/blue map shows this contribution averaged over the members; the attribution map ranks "
            "tiles by it in the direction whose removal moves this patient's prediction more.\n\n"
            "Method. Each tissue tile is encoded by a frozen, publicly released pathology foundation model (Kaiko "
            "Midnight, ViT-g/14, MIT licence). An attention-based multiple-instance model, trained only on patient-level "
            "time to biochemical recurrence (a standard survival loss) of the 508 Radboudumc training patients, combines "
            "the tiles. No grading or staging system, morphological annotation or biomarker detector is used at any "
            "step. Nothing in it is specific to prostate or to recurrence (general pathology encoder, patterns found "
            "without labels, exact contributions for any attention-weighted model), but retraining it for another "
            "outcome or cancer has not been tested. Validation: the training recipe "
            "reached a C-index of 0.687 in 5-fold cross-validation on the training cohort (3 repeats); this ensemble "
            "reached 0.637 (95% CI 0.552-0.718) on 298 TCGA-PRAD patients from other centres.\n\n"
            "Limits. The prediction is a research output, not a clinical decision aid. The attribution map shows which "
            "tissue moves the prediction; the pattern names on page 3 are unverified descriptions, not diagnoses. "
            f"The predicted time is the expected BCR-free time within the first {tau:.0f} years (restricted mean), so it "
            f"cannot exceed {tau:.0f}; the middle third of the training patients' predicted times lies between "
            f"{t1:.1f} and {t2:.1f} years. Training follow-up is censored for most patients, so the model knows little "
            f"about recurrence late in that window. This slide had {meta['n_candidates']} tissue tiles; {encoded}, and "
            f"{len(f)} remained after dropping near-blank tiles."), size=8.3, width=114)
        pdf.savefig(fig)
        plt.close(fig)

        # page 5: lower-power context around the strongest tiles
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.96, "Context around the strongest tiles", fontsize=13, weight="bold")
        _text(fig, 0.935, "About 1 mm of tissue around tiles #1-#3 of each row on page 2 (the yellow box is the tile; "
                          "the black bar is 200 µm). Numbers match page 2 and the markers on page 1.", size=8.5, width=110)
        for row, (crops, idx, name) in enumerate(((ctx_hi, hi_idx, "raises"), (ctx_lo, lo_idx, "lowers"))):
            for j, (crop, (bx, by, bw), px_um, (oy_, ox_, ch_, cw_)) in enumerate(crops):
                ax = fig.add_axes([0.04 + j * 0.32, 0.63 - row * 0.32, 0.29, 0.25])
                ax.imshow(crop)
                ax.add_patch(plt.Rectangle((bx, by), bw, bw, fill=False, edgecolor="yellow", linewidth=1.2))
                h_, w_ = crop.shape[:2]
                ax.plot([ox_ + 0.05 * cw_, ox_ + 0.05 * cw_ + 200 * px_um], [oy_ + 0.95 * ch_] * 2, color="black",
                        linewidth=2.5)
                i = idx[j]
                pat = f"c{cl[i]} {model.atlas['desc'][str(cl[i])]['name']}\n" if cl is not None else ""
                ax.set_title(f"#{j + 1} {name} the score\n{pat}@({c[i][0]},{c[i][1]})", fontsize=6.5)
                ax.axis("off")
        if cl is not None:
            art_ = np.array([model.atlas["desc"][str(int(j))].get("group") == "artefact" for j in cl])
            k_up = int(np.bincount(cl[up10]).argmax())
            top_k = np.zeros(len(cl), bool)
            top_k[up10] = True
            top_k &= cl == k_up
            rest_k = (cl == k_up) & ~top_k
            for col, (title, layers) in enumerate((
                    (f"Artefact patterns (black): {100 * art_.mean():.1f}% of this slide's tissue.\n"
                     + _art_dir(cl, np.array([model.atlas["desc"][str(j)].get("group") == "artefact"
                                              for j in range(len(model.atlas["centers"]))]), signed, short=True),
                     [(art_, "black")]),
                    (f"c{k_up} {model.atlas['desc'][str(k_up)]['name']}:\nyellow = among the 10% most risk-raising "
                     f"tiles, cyan = the rest of this pattern\n(mean contribution {signed[top_k].mean():+.1e} vs "
                     f"{signed[rest_k].mean() if rest_k.any() else 0:+.1e}).\n"
                     + ("Nearly all of this pattern's tiles are in the top 10%." if rest_k.sum() < 0.2 * max(1, top_k.sum())
                        else f"Yellow tiles get {att[top_k].mean() / max(att[rest_k].mean(), 1e-12):.1f}x the attention and "
                        f"a tile risk of {trisk[top_k].mean():+.2f} vs {trisk[rest_k].mean():+.2f}.\nWhat sets them apart "
                        "in the image was not examined; higher tile risk\ndoes not necessarily mean less well-formed glands."),
                     [(rest_k, "cyan"), (top_k, "yellow")]))):
                ax = fig.add_axes([0.05 + col * 0.47, 0.03, 0.43, 0.23])
                ax.imshow(thumb, extent=ext)
                for sel_, colr in layers:
                    if sel_.any():
                        size = (0.43 * A4[0] * 72 / max(gw, gh * 0.43 * A4[0] / (0.23 * A4[1]))) ** 2  # ~one tile
                        ax.scatter(g[sel_, 0] + 0.5, g[sel_, 1] + 0.5, s=max(1.5, size), c=colr, marker="s",
                                   linewidths=0)
                ax.set_xlim(0, gw)
                ax.set_ylim(gh, 0)
                ax.set_title(title, fontsize=6.5)
                ax.axis("off")
        pdf.savefig(fig)
        plt.close(fig)
