"""Stage 1: official EPLB full reshuffle vs Pinned-Base incremental planner.

Qwen3.6-35B-A3B shape: 40 MoE layers, 256 routed experts, 8-way EP.
Each expert slot is 3 * 2048 * 512 params = 6.3 MB in BF16.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from live_eplb.eplb_official import rebalance_experts
from live_eplb.planner import (
    plan_incremental_redundant,
    rank_load,
    static_phy2log,
    log2phy_from_phy2log,
)

NUM_LAYERS, NUM_EXPERTS = 40, 256
RANKS, EPR, RED = 8, 32, 4
SLOTS = EPR + RED
EXPERT_MB = 3 * 2048 * 512 * 2 / 1e6   # 6.29 MB per expert, BF16


def zipf_load(num_layers, num_experts, alpha, seed, phase_shift=0):
    """Zipf-skewed per-expert token counts; `phase_shift` rotates the hot set."""
    g = torch.Generator().manual_seed(seed)
    base = 1.0 / torch.arange(1, num_experts + 1).float() ** alpha
    w = torch.empty(num_layers, num_experts)
    for l in range(num_layers):
        perm = torch.randperm(num_experts, generator=g)
        perm = (perm + phase_shift) % num_experts
        w[l] = base[perm] * (0.6 + 0.8 * torch.rand(num_experts, generator=g))
    return w * 1e4


def official(weight):
    """DeepSeek official EPLB, global policy (num_groups=1, num_nodes=1) = our EP=8 layout."""
    return rebalance_experts(weight, RANKS * SLOTS, num_groups=1, num_nodes=1, num_gpus=RANKS)


def row(name, phy2log, weight, plan_s, prev=None, base_frozen=False):
    load = rank_load(phy2log, weight, RANKS)
    ratio = (load.max(-1).values / load.mean(-1)).mean().item()
    changed, base_changed = 0, 0
    if prev is not None:
        diff = (phy2log != prev)
        changed = int(diff.sum().item())
        if base_frozen:
            for r in range(RANKS):
                base_changed += int(diff[:, r * SLOTS:r * SLOTS + EPR].sum().item())
    return (name, ratio, load.max(-1).values.mean().item(), plan_s * 1e3,
            changed, base_changed, changed * EXPERT_MB / 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=1.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drift", type=int, default=64, help="how far the hot set rotates")
    args = ap.parse_args()

    w0 = zipf_load(NUM_LAYERS, NUM_EXPERTS, args.alpha, args.seed)
    w1 = zipf_load(NUM_LAYERS, NUM_EXPERTS, args.alpha, args.seed, phase_shift=args.drift)

    print(f"Qwen3.6-35B-A3B: {NUM_LAYERS} MoE layers x {NUM_EXPERTS} experts, EP={RANKS}, "
          f"{EPR} base + {RED} redundant slots/rank = {RANKS * SLOTS} physical slots")
    print(f"expert slot = {EXPERT_MB:.2f} MB BF16; traffic Zipf(alpha={args.alpha}), "
          f"hot set rotates by {args.drift} experts between snapshots\n")

    base = static_phy2log(NUM_LAYERS, RANKS, EPR, RED)

    t = time.perf_counter(); off0, _, _ = official(w0); t_off0 = time.perf_counter() - t
    t = time.perf_counter(); off1, _, _ = official(w1); t_off1 = time.perf_counter() - t
    t = time.perf_counter(); inc0, _, _, _ = plan_incremental_redundant(w0, RANKS, EPR, RED)
    t_inc0 = time.perf_counter() - t
    t = time.perf_counter()
    inc1, _, _, tasks = plan_incremental_redundant(w1, RANKS, EPR, RED, prev_phy2log=inc0)
    t_inc1 = time.perf_counter() - t

    rows = [
        row("Static EP (no balancing)", base, w1, 0.0),
        row("Official EPLB  plan@t0", off0, w0, t_off0),
        row("Official EPLB  reshuffle", off1, w1, t_off1, prev=off0),
        row("LiveEPLB       plan@t0", inc0, w0, t_inc0),
        row("LiveEPLB       incremental", inc1, w1, t_inc1, prev=inc0, base_frozen=True),
    ]

    hdr = (f"{'policy':<28}{'Max/Mean':>10}{'MaxLoad':>11}{'plan ms':>10}"
           f"{'slots moved':>13}{'base moved':>12}{'GB copied':>11}")
    print(hdr)
    print("-" * len(hdr))
    for name, ratio, maxload, ms, moved, bmoved, gb in rows:
        print(f"{name:<28}{ratio:>10.3f}{maxload:>11.0f}{ms:>10.1f}{moved:>13}"
              f"{bmoved:>12}{gb:>11.2f}")

    print("\n--- copy plan for the traffic shift (snapshot t0 -> t1) ---")
    print(f"LiveEPLB : {len(tasks):>5} tasks, {len(tasks) * EXPERT_MB / 1024:.2f} GB"
          f"  ({len(tasks) / (NUM_LAYERS * RANKS * RED) * 100:.0f}% of redundant slots)")
    print(f"Official : {int((off1 != off0).sum()):>5} tasks, "
          f"{int((off1 != off0).sum()) * EXPERT_MB / 1024:.2f} GB  (full reshuffle)")
    if tasks:
        print("first 3 (layer, src_rank, src_slot, dst_rank, dst_slot, expert):")
        for tsk in tasks[:3]:
            print("   ", tsk)

    for tag, p2l in (("official", off1), ("incremental", inc1)):
        _, _, lc = log2phy_from_phy2log(p2l, NUM_EXPERTS)
        assert (lc >= 1).all(), f"{tag}: unreachable expert"
        assert torch.allclose(rank_load(p2l, w1, RANKS).sum(-1), w1.sum(-1), rtol=1e-4)
    print("\ncorrectness: every expert reachable, per-layer token count conserved [OK]")

    os.makedirs("results", exist_ok=True)
    torch.save(dict(base=base, official_0=off0, official_1=off1,
                    incremental_0=inc0, incremental_1=inc1,
                    weight_0=w0, weight_1=w1, tasks=tasks),
               "results/stage1_plans.pt")
    print("saved results/stage1_plans.pt")


if __name__ == "__main__":
    main()
