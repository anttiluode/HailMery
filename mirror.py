"""mirror.py: a babbling child in front of a mirror, as a linear toy.

The child sends motor commands u(t). Three senses return mixtures:

  vision  = own mouth/hand (seen in the mirror) + another person + noise
  sound   = own voice                           + another person + noise
  touch   = own body (proprioception)                            + noise

The other person is ALSO seen and heard together: they are a cross-modal
common cause too. Each sense reaches the child through its own unknown,
delayed, smeared body path.

Three questions, each answered by measurement:

  Q1  Does "what my senses agree on" find the self?
      Canonical correlation between vision and sound (no motor information).
  Q2  Does "what my own commands predict" find the self?
      Regress every sense on a window of the child's own recent commands.
  Q3  Where is the self/other boundary?
      (a) the mirror image arrives late (delayed video);
      (b) another person imitates the child with a delay.
      Both swept over delay, for smooth and for jittery babbling.

This is a toy built to make definitions exact. It does not touch why any of
this would feel like anything.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

T_TRAIN, T_TEST = 20_000, 8_000
N_U, N_W = 3, 3            # motor command dims, other-person source dims
DIMS = {"vision": 8, "sound": 8, "touch": 4}
DELAY = {"vision": 3, "sound": 2, "touch": 1}   # body path delays (steps)
WINDOW = 12                # how far back the child relates commands to senses
NOISE = 0.3


def smooth_process(rng, T, dims, rho):
    """AR(1) process with unit variance; rho near 1 = smooth, near 0 = jittery."""
    x = np.zeros((T, dims))
    e = rng.normal(size=(T, dims)) * np.sqrt(1 - rho**2)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x


def shift(x, d):
    """Delay a signal by d steps (zeros at the start)."""
    if d == 0:
        return x.copy()
    out = np.zeros_like(x)
    out[d:] = x[:-d]
    return out


def body_path(rng, x, out_dims, delay, taps=4):
    """Unknown smeared, delayed linear path from a source to a sense."""
    y = np.zeros((len(x), out_dims))
    for k in range(taps):
        M = rng.normal(size=(x.shape[1], out_dims)) / np.sqrt(x.shape[1] * taps)
        y += shift(x, delay + k) @ M
    return y


def world(seed, T, babble_rho=0.9, mirror_delay=0, imitator_delay=None):
    """Return commands, senses, and the ground-truth self/other parts of each sense."""
    rng = np.random.default_rng(seed)
    u = smooth_process(rng, T, N_U, babble_rho)
    if imitator_delay is None:
        w = smooth_process(rng, T, N_W, 0.9)        # an independent other person
    else:
        P = rng.normal(size=(N_U, N_W)) / np.sqrt(N_U)
        w = shift(u, imitator_delay) @ P               # someone copying the child, late
        w /= w.std(0) + 1e-9
    senses, self_part, other_part = {"_w": w}, {}, {}
    for name, dims in DIMS.items():
        extra = mirror_delay if name == "vision" else 0
        s = body_path(rng, u, dims, DELAY[name] + extra)
        o = body_path(rng, w, dims, 2) if name != "touch" else np.zeros((T, dims))
        senses[name] = s + o + NOISE * rng.normal(size=(T, dims))
        self_part[name], other_part[name] = s, o
    return u, senses, self_part, other_part


def embed(u, L):
    """Window of the child's own recent commands: [u(t), u(t-1), ..., u(t-L+1)]."""
    return np.hstack([shift(u, k) for k in range(L)])


def r2(pred, true):
    true_c = true - true.mean(0)
    return float(1 - np.sum((pred - true) ** 2) / max(np.sum(true_c**2), 1e-12))


def contingency_learner(u_tr, y_tr, L=WINDOW, lam=1e-2):
    """Least squares from the command window to a sense: the part I can predict from my own acts."""
    X = embed(u_tr, L)
    X1 = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(X1.T @ X1 + lam * np.eye(X1.shape[1]), X1.T @ y_tr)


def contingency_predict(B, u, L=WINDOW):
    X = embed(u, L)
    return np.hstack([X, np.ones((len(X), 1))]) @ B


def cca(a, b, k):
    """Top-k canonical directions between two multichannel signals."""
    a = a - a.mean(0)
    b = b - b.mean(0)
    def whiten(x):
        C = x.T @ x / len(x) + 1e-6 * np.eye(x.shape[1])
        e, V = np.linalg.eigh(C)
        return V @ np.diag(e ** -0.5) @ V.T
    Wa, Wb = whiten(a), whiten(b)
    M = Wa @ (a.T @ b / len(a)) @ Wb
    U, s, Vt = np.linalg.svd(M)
    return (a @ Wa @ U[:, :k]), s[:k]


def explained(target, source):
    """Fraction of target's variance linearly explained by source (in-sample)."""
    X = np.hstack([source, np.ones((len(source), 1))])
    beta = np.linalg.lstsq(X, target, rcond=None)[0]
    return max(0.0, r2(X @ beta, target))


def q1_q2(seed=0):
    u, Y, S, O = world(seed, T_TRAIN + T_TEST)
    w = Y.pop("_w")
    tr, te = slice(0, T_TRAIN), slice(T_TRAIN, None)

    # Q1: what vision and sound agree on, with no motor information
    k = N_U + N_W
    comps, corr = cca(Y["vision"][tr], Y["sound"][tr], k)
    my_acts = embed(u[tr], WINDOW)
    their_acts = embed(w[tr], WINDOW)
    rows = [{"canonical_corr": float(corr[j]),
             "explained_by_self": explained(comps[:, j:j + 1], my_acts),
             "explained_by_other": explained(comps[:, j:j + 1], their_acts)} for j in range(k)]

    # Q2: what my own commands predict, on held-out time
    q2 = {}
    for name in DIMS:
        B = contingency_learner(u[tr], Y[name][tr])
        pred = contingency_predict(B, u[te])
        q2[name] = {"self_recovered": r2(pred, S[name][te]),
                    "other_leak": explained(pred, O[name][te]) if name != "touch" else 0.0}
    return {"Q1_multisensory_agreement": rows, "Q2_own_command_prediction": q2}


def attributed(u, y_part, tr, te):
    """Fraction of a component's variance the child would attribute to itself."""
    B = contingency_learner(u[tr], y_part[tr])
    return max(0.0, r2(contingency_predict(B, u[te]), y_part[te]))


def q3(seed=0, delays=(0, 2, 4, 8, 12, 16, 24, 32, 48)):
    out = {}
    for label, rho in (("smooth_babble", 0.95), ("jittery_babble", 0.3)):
        mirror, imitator = [], []
        for d in delays:
            u, Y, S, O = world(seed, T_TRAIN + T_TEST, babble_rho=rho, mirror_delay=d)
            tr, te = slice(0, T_TRAIN), slice(T_TRAIN, None)
            mirror.append(attributed(u, S["vision"], tr, te))
            u, Y, S, O = world(seed, T_TRAIN + T_TEST, babble_rho=rho, imitator_delay=d)
            imitator.append(attributed(u, O["vision"], tr, te))
        out[label] = {"delays": list(delays), "own_image_attributed_to_self": mirror,
                      "imitator_attributed_to_self": imitator, "babble_rho": rho}
    return out


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    os.makedirs("results", exist_ok=True)
    res = {"window": WINDOW, "body_delays": DELAY, **q1_q2(), "Q3_boundary": q3()}
    with open("results/mirror.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("Q1  what vision and sound agree on (no motor info):")
    for j, r in enumerate(res["Q1_multisensory_agreement"]):
        print(f"   component {j}: corr {r['canonical_corr']:.2f}   "
              f"explained by self {r['explained_by_self']:.2f}   by other {r['explained_by_other']:.2f}")
    print("Q2  what my own commands predict (test data):")
    for k, v in res["Q2_own_command_prediction"].items():
        print(f"   {k:6s} self part recovered R2 {v['self_recovered']:.3f}   other leaked {v['other_leak']:.3f}")
    print(f"Q3  self/other boundary (window = {WINDOW} steps):")
    for label, v in res["Q3_boundary"].items():
        print(f"   {label} (rho {v['babble_rho']}):")
        print("     delay           " + " ".join(f"{d:5d}" for d in v["delays"]))
        print("     own image       " + " ".join(f"{x:5.2f}" for x in v["own_image_attributed_to_self"]))
        print("     imitator        " + " ".join(f"{x:5.2f}" for x in v["imitator_attributed_to_self"]))
    plot(res)


def plot(res):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (label, v) in zip(axes, res["Q3_boundary"].items()):
        ax.plot(v["delays"], v["own_image_attributed_to_self"], "-o", label="own mirror image")
        ax.plot(v["delays"], v["imitator_attributed_to_self"], "-s", label="someone imitating me")
        ax.axvline(res["window"], color="#999", ls="--", lw=1)
        ax.text(res["window"] + 0.5, 0.95, "my window", color="#777", fontsize=9, va="top")
        ax.set_title(label.replace("_", " ") + f"  (rho = {v['babble_rho']})")
        ax.set_xlabel("delay (steps)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fraction attributed to self")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig("results/boundary.png", dpi=130)


if __name__ == "__main__":
    main()
