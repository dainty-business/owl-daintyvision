"""Additive attention MIL for survival: risk = sum_i a_i * r_i, so per-tile contributions a_i * r_i sum exactly to the
slide risk. Shared by training (tools/train_mil.py) and the container."""
import numpy as np
import torch
import torch.nn as nn


class AdditiveABMIL(nn.Module):
    """n_out = 1: Cox risk. n_out = K: discrete-time hazard logits per bin (risk_of turns them into a scalar)."""

    def __init__(self, d_in=3072, d=256, d_att=128, dropout=0.25, n_out=1):
        super().__init__()
        self.embed = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout))
        self.att_v, self.att_u, self.att_w = nn.Linear(d, d_att), nn.Linear(d, d_att), nn.Linear(d_att, 1)
        self.head = nn.Linear(d, n_out)

    def tiles(self, x):
        """Per-tile attention logits and risks for one bag x (N, d_in)."""
        h = self.embed(x)
        logit = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        return logit, self.head(h).squeeze(-1)

    def forward(self, x):
        logit, r = self.tiles(x)
        a = torch.softmax(logit, 0)
        return ((a * r).sum() if r.ndim == 1 else (a[:, None] * r).sum(0)), a, r



class MultiAdditiveABMIL(AdditiveABMIL):
    """K attention branches, each an additive MIL over the same tiles; risk = sum_k sum_i a_ik r_ik, so removing a tile
    still changes the risk by an exact sum of per-branch renormalizations."""

    def __init__(self, d_in=3072, d=256, d_att=128, dropout=0.25, branches=2):
        super().__init__(d_in, d, d_att, dropout, n_out=branches)
        self.att_w = nn.Linear(d_att, branches)

    def tiles(self, x):
        h = self.embed(x)
        return self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))), self.head(h)

    def forward(self, x):
        logit, r = self.tiles(x)
        a = torch.softmax(logit, 0)
        return (a * r).sum(), a, r

def risk_of(out):
    """Scalar risk from a slide output: the Cox log-risk itself, or minus the summed survival over discrete bins."""
    if out.ndim == 0 or out.shape[-1:] == (1,) and out.ndim == 1:
        return out.reshape(())
    return -torch.cumprod(1 - torch.sigmoid(out), -1).sum(-1)


def nll_loss(logits, y_bin, censored, alpha=0.4):
    """Discrete-time survival NLL (Zadeh & Schmid 2020 form, as in CLAM/PORPOISE); logits (B, K), y_bin in [0, K)."""
    h = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    S = torch.cat([torch.ones_like(h[:, :1]), torch.cumprod(1 - h, 1)], 1)
    y = y_bin[:, None]
    unc = -(1 - censored) * (torch.log(S.gather(1, y)) + torch.log(h.gather(1, y))).squeeze(1)
    cen = -censored * torch.log(S.gather(1, y + 1)).squeeze(1)
    return ((1 - alpha) * (unc + cen) + alpha * unc).mean()


def cox_loss(risk, time, event):
    """Negative Cox partial log-likelihood (Breslow ties), mean over events in the batch."""
    order = torch.argsort(time, descending=True)
    risk, event = risk[order], event[order]
    return -((risk - torch.logcumsumexp(risk, 0)) * event).sum() / event.sum().clamp(min=1)


def breslow(risk, time, event):
    """Baseline cumulative hazard at the unique event times (numpy)."""
    risk, time, event = map(np.asarray, (risk, time, event))
    ts = np.unique(time[event == 1])
    h0 = np.array([((time == t) & (event == 1)).sum() / np.exp(risk[time >= t]).sum() for t in ts])
    return ts, np.cumsum(h0)


def rmst(risk, ts, H0, tau):
    """Restricted mean BCR-free time up to tau years under S(t) = exp(-H0(t) exp(risk)); monotone decreasing in risk."""
    risk = np.atleast_1d(np.asarray(risk, np.float64))
    knots = np.concatenate([[0.0], ts[ts < tau], [tau]])
    Hk = np.concatenate([[0.0], H0[ts < tau]])  # hazard on [knots[j], knots[j+1])
    widths = np.diff(knots)
    S = np.exp(-np.outer(np.exp(risk), Hk))
    return (S * widths).sum(1)
