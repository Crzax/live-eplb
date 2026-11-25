"""Plot results/stage4_timeline.json -> results/synthetic_tpot_timeline.png"""
import json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = "results/stage4_timeline.json"
OUT = "results/synthetic_tpot_timeline.png"
STYLE = {"static": ("#888888", "Static EP (no balancing)"),
         "stw": ("#d62728", "Stop-The-World EPLB (re-capture)"),
         "stw_inplace": ("#e4572e", "Stop-The-World, in-place commit"),
         "live": ("#1f77b4", "LiveEPLB")}


def main():
    shift_at = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    src = sys.argv[2] if len(sys.argv) > 2 else SRC
    out = sys.argv[3] if len(sys.argv) > 3 else OUT
    d = json.load(open(src))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    for p, (c, label) in STYLE.items():
        t = np.array(d[p]["tpot"])
        ax1.plot(t, color=c, lw=1.1, label=f"{label}  (p99 {np.percentile(t, 99):.1f} ms, "
                                          f"worst {t.max():.1f} ms)")
    ax1.axvline(shift_at, color="k", ls="--", lw=0.8)
    ax1.text(shift_at + 3, ax1.get_ylim()[1] * 0.9, "traffic shift", fontsize=9)
    ax1.set_ylabel("TPOT per decode step (ms)")
    ax1.set_yscale("log")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    for p, (c, label) in STYLE.items():
        ax2.plot(d[p]["imbalance"], color=c, lw=1.1, label=label)
    ax2.axvline(shift_at, color="k", ls="--", lw=0.8)
    ax2.set_xlabel("decode step")
    ax2.set_ylabel("Max/Mean rank load")
    ax2.grid(alpha=0.3)

    for p, mark, label in (("live", "commit", "LiveEPLB in-place commit (async)"),
                           ("stw_inplace", "commit", "STW in-place commit (sync)"),
                           ("stw", "recapture", "STW re-capture")):
        at = [i for i, a in enumerate(d[p]["actions"]) if a == mark]
        c = STYLE[p][0]
        ax2.plot(at, [d[p]["imbalance"][i] for i in at], "v", color=c, ms=7)
        ax2.plot([], [], "v", color=c, ms=7, label=label)
    steps = len(d["static"]["tpot"])
    ax1.set_title(f"Expert rebalancing under a traffic shift, 40 MoE layers, 8x H20, {steps} decode steps")
    ax2.legend(loc="upper right", fontsize=8, ncol=2)

    os.makedirs("results", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
