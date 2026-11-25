"""Stage 1 replay on real traces: incremental plan vs vendored official EPLB,
weights = phase counts normalized to TOKENS routed tokens per layer.

Writes results/stage5_plan.json and results/qwen3_6_layer_skew_heatmap.png
(per-layer Max/Mean: static on A, static on B, LiveEPLB-replanned on B).
"""
import json, os, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live_eplb.planner import (plan_incremental_redundant, rank_load,
                               static_phy2log)
from live_eplb.eplb_official import rebalance_experts

L, R, EPR, RED, E = 40, 8, 32, 4, 256
TOKENS = 64
SLOT_MB = 6.29


def maxmean(p2l, w):
    load = rank_load(p2l, w, R)
    return load.max(-1).values / load.mean(-1)


def weights_of(counts):
    # normalize each layer to TOKENS routed assignments per step
    return counts / counts.sum(-1, keepdim=True) * TOKENS


def main():
    d = torch.load("results/router_traces.pt", weights_only=False)
    wA = weights_of(d["phases"]["phaseA_code"]["counts"])
    wB = weights_of(d["phases"]["phaseB_math_wiki"]["counts"])

    rows = {}
    p2l0 = static_phy2log(L, R, EPR, RED)

    for tag, (w0, w1) in (("A->B (live)", (wA, wB)),):
        # official EPLB planned from scratch on phase B, full reshuffle off A
        phy_off, l2p_off, _ = rebalance_experts(w1, R * (EPR + RED), 1, 1, R)
        # LiveEPLB: plan on A, that is what runs during phase A;
        # then incremental re-plan when traffic shifts to B
        phy_a, log2phy_a, _, tasks_a = plan_incremental_redundant(
            w0, R, EPR, RED, prev_phy2log=p2l0)
        phy_live, log2phy_live, _, tasks_live = plan_incremental_redundant(
            w1, R, EPR, RED, prev_phy2log=phy_a)

        def slots_move(tasks):
            return len(tasks), round(len(tasks) * SLOT_MB / 1024, 2)

        rows["static on A"] = dict(maxmean=maxmean(p2l0, wA).mean().item(),
                                   moved=0, gb=0.0)
        rows["static on B"] = dict(maxmean=maxmean(p2l0, wB).mean().item(),
                                   moved=0, gb=0.0)
        rows["live plan@A"] = dict(maxmean=maxmean(phy_a, wA).mean().item(),
                                   moved=slots_move(tasks_a)[0],
                                   gb=slots_move(tasks_a)[1])
        rows["live incr A->B"] = dict(
            maxmean=maxmean(phy_live, wB).mean().item(),
            worst_layer=maxmean(phy_live, wB).max().item(),
            moved=slots_move(tasks_live)[0], gb=slots_move(tasks_live)[1])
        rows["official reshuffle @B"] = dict(
            maxmean=maxmean(phy_off, wB).mean().item(),
            moved=int((phy_off != p2l0).sum().item()),
            gb=round((phy_off != p2l0).sum().item() * SLOT_MB / 1024, 2))

    print(f"{'policy':<24}{'Max/Mean':>10}{'worst layer':>13}{'slots moved':>13}"
          f"{'GB':>8}")
    for k, v in rows.items():
        print(f"{k:<24}{v['maxmean']:>10.3f}{v.get('worst_layer', 0):>13.3f}"
              f"{v['moved']:>13}{v['gb']:>8}")

    mm = {
        "A_static": maxmean(p2l0, wA).tolist(),
        "B_static": maxmean(p2l0, wB).tolist(),
        "B_liveeplb": maxmean(phy_live, wB).tolist(),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/stage5_plan.json", "w") as f:
        json.dump(dict(rows=rows, heatmap=mm), f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    M = np.array([mm["A_static"], mm["B_static"], mm["B_liveeplb"]]).T
    fig, ax = plt.subplots(figsize=(5.4, 8))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=1.0, vmax=2.8)
    ax.set_yticks(range(0, 40, 5))
    ax.set_ylabel("MoE layer")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Phase A\n(code), static", "Phase B\n(math/wiki), static",
                        "Phase B, after\nLiveEPLB re-plan"], fontsize=8)
    for i in range(3):
        for j in range(40):
            ax.text(i, j, f"{M[j, i]:.2f}", ha="center", va="center",
                    fontsize=4.5, color="w")
    ax.set_title("Per-layer Max/Mean rank load, real router traces\n"
                 "(1.0 = perfect balance)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("results/qwen3_6_layer_skew_heatmap.png", dpi=150)
    print("wrote results/qwen3_6_layer_skew_heatmap.png")


if __name__ == "__main__":
    main()
