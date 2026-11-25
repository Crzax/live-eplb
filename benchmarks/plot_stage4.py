"""Plot results/stage4_timeline.json -> results/synthetic_tpot_timeline.png"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = "results/stage4_timeline.json"
OUT = "results/synthetic_tpot_timeline.png"
STYLE = {
    "static":                ("#7f7f7f", "Static EP (no balancing)", "-"),
    "stw":                   ("#d62728", "STW, our planner, re-capture", "-"),
    "stw_inplace":           ("#ff7f0e", "STW, ours, in-place commit", "-"),
    "stw_official":          ("#9467bd", "STW, official EPLB + recapture", "-"),
    "stw_official_inplace":  ("#2ca02c", "STW, official EPLB + in-place", "-"),
    "live":                  ("#1f77b4", "LiveEPLB", "-"),
}


def rolling(x, n=15):
    x = np.asarray(x)
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[n:] - c[:-n]) / n


def main():
    shift_at = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    src = sys.argv[2] if len(sys.argv) > 2 else SRC
    out = sys.argv[3] if len(sys.argv) > 3 else OUT
    d = json.load(open(src))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    base = np.median(np.array(d["static"]["tpot"]))
    for p, (c, label, _) in STYLE.items():
        if p not in d:
            continue
        t = np.array(d[p]["tpot"])
        y = np.clip(t - base, 5e-3, None)
        rm = rolling(y)
        # scatter: per-step dots so zigzag 不连线; rolling median 15-step 鉴趋势
        xs = np.arange(len(y))
        ax1.scatter(xs, y, color=c, s=3, alpha=0.25, lw=0)
        ax1.plot(np.arange(len(rm)) + 7, rm, color=c, lw=1.4,
                 label=f"{label}  (worst +{t.max() - base:.1f} ms)")
    ax1.axhline(0.0, color="k", lw=0.6, ls="--")
    ax1.text(2, 0.03, "baseline = static p50", fontsize=8)
    ax1.axvline(shift_at, color="k", ls="--", lw=0.8)
    ax1.text(shift_at + 3, ax1.get_ylim()[1] * 0.85, "traffic shift", fontsize=9)
    ax1.set_yscale("log")
    ax1.set_ylabel("TPOT - static p50 (ms, log)")
    ax1.legend(loc="upper right", fontsize=7)
    ax1.grid(alpha=0.3)
    ax1.set_title(f"Expert rebalancing under a traffic shift, 8x H20, "
                  f"{len(d['static']['tpot'])} decode steps")

    for p, (c, label, _) in STYLE.items():
        if p not in d:
            continue
        ax2.plot(d[p]["imbalance"], color=c, lw=1.1, label=label)
    ax2.axvline(shift_at, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("decode step")
    ax2.set_ylabel("Max/Mean rank load")
    ax2.grid(alpha=0.3)

    for p, mark, label in (("live", "commit", "LiveEPLB in-place commit (async)"),
                           ("stw_inplace", "commit", "STW in-place commit (sync)"),
                           ("stw_official_inplace", "commit",
                            "STW official in-place commit"),
                           ("stw", "recapture", "STW re-capture"),
                           ("stw_official", "recapture", "STW official re-capture")):
        if p not in d:
            continue
        at = [i for i, a in enumerate(d[p]["actions"]) if a == mark]
        c = STYLE[p][0]
        ax2.plot(at, [d[p]["imbalance"][i] for i in at], "v", color=c, ms=7)
        ax2.plot([], [], "v", color=c, ms=7, label=label)
    ax2.legend(loc="upper right", fontsize=6, ncol=2)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
