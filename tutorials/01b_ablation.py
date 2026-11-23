"""Stage 1 ablation: is the advantage stable, and where does the balance gap come from?

Two questions:
  A) Does the migration-volume advantage hold across traffic skew levels (alpha sweep)?
  B) Is our greedy worse than the official packing, or is the gap purely the
     Pinned-Base constraint? Run the official policy under the SAME budget and compare.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from live_eplb.eplb_official import rebalance_experts
from live_eplb.planner import plan_incremental_redundant, rank_load, static_phy2log

NUM_LAYERS, NUM_EXPERTS = 40, 256
RANKS, EPR, RED = 8, 32, 4
SLOTS = EPR + RED
EXPERT_MB = 3 * 2048 * 512 * 2 / 1e6


def zipf_load(num_layers, num_experts, alpha, seed, phase_shift=0):
    g = torch.Generator().manual_seed(seed)
    base = 1.0 / torch.arange(1, num_experts + 1).float() ** alpha
    w = torch.empty(num_layers, num_experts)
    for l in range(num_layers):
        perm = torch.randperm(num_experts, generator=g)
        perm = (perm + phase_shift) % num_experts
        w[l] = base[perm] * (0.6 + 0.8 * torch.rand(num_experts, generator=g))
    return w * 1e4


def official(weight):
    return rebalance_experts(weight, RANKS * SLOTS, num_groups=1, num_nodes=1, num_gpus=RANKS)[0]


def imbalance(phy2log, weight):
    load = rank_load(phy2log, weight, RANKS)
    return (load.max(-1).values / load.mean(-1)).mean().item()


def changed_slots(new, old):
    return int((new != old).sum().item())


def redundant_only_official(weight):
    """Official replica *selection* (argmax w/c) but forced into the Pinned-Base layout.

    This isolates the constraint from the algorithm: same budget (4 redundant slots
    per rank, base frozen), official greedy choice of which expert to replicate.
    """
    num_layers, num_experts = weight.shape
    w = weight.float()
    phy2log = static_phy2log(num_layers, RANKS, EPR, RED)
    logcnt = torch.ones(num_layers, num_experts)
    for l in range(num_layers):
        free = {r: list(range(EPR, SLOTS)) for r in range(RANKS)}
        holds = {r: set(range(r * EPR, (r + 1) * EPR)) for r in range(RANKS)}
        for _ in range(RANKS * RED):
            # official rule: hottest per-replica expert, placed on least-loaded rank
            per_replica = w[l] / logcnt[l]
            order = torch.argsort(per_replica, descending=True)
            placed = False
            for e in order.tolist():
                cand = [r for r in range(RANKS) if free[r] and e not in holds[r]]
                if not cand:
                    continue
                load = {r: sum(float(w[l, x] / logcnt[l, x]) for x in holds[r])
                        for r in range(RANKS)}
                r = min(cand, key=lambda x: load[x])
                slot = free[r].pop(0)
                phy2log[l, r * SLOTS + slot] = e
                holds[r].add(e)
                logcnt[l, e] += 1
                placed = True
                break
            if not placed:
                break
    return phy2log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drift", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.5, 0.8, 1.1, 1.5, 2.0])
    args = ap.parse_args()

    print("=" * 78)
    print("A) alpha sweep: does the migration advantage hold across skew levels?")
    print("=" * 78)
    print(f"{'alpha':>6}{'static':>9}{'official':>10}{'ours':>8}"
          f"{'off GB':>9}{'our GB':>9}{'reduction':>11}{'our cap':>9}")
    print("-" * 78)

    sweep = []
    for alpha in args.alphas:
        w0 = zipf_load(NUM_LAYERS, NUM_EXPERTS, alpha, args.seed)
        w1 = zipf_load(NUM_LAYERS, NUM_EXPERTS, alpha, args.seed, phase_shift=args.drift)

        off0, off1 = official(w0), official(w1)
        inc0, _, _, _ = plan_incremental_redundant(w0, RANKS, EPR, RED)
        inc1, _, _, tasks = plan_incremental_redundant(w1, RANKS, EPR, RED, prev_phy2log=inc0)

        off_gb = changed_slots(off1, off0) * EXPERT_MB / 1024
        our_gb = len(tasks) * EXPERT_MB / 1024
        cap_gb = NUM_LAYERS * RANKS * RED * EXPERT_MB / 1024   # hard upper bound

        sweep.append((alpha, imbalance(static_phy2log(NUM_LAYERS), w1),
                      imbalance(off1, w1), imbalance(inc1, w1), off_gb, our_gb, cap_gb))
        print(f"{alpha:>6.1f}{sweep[-1][1]:>9.3f}{sweep[-1][2]:>10.3f}{sweep[-1][3]:>8.3f}"
              f"{off_gb:>9.2f}{our_gb:>9.2f}{off_gb / max(our_gb, 1e-9):>10.1f}x{cap_gb:>9.2f}")

    print(f"\nour migration volume never exceeds the hard cap of {sweep[0][6]:.2f} GB "
          f"({NUM_LAYERS * RANKS * RED} slots), regardless of alpha.")
    off_gbs = [s[4] for s in sweep]
    print(f"official ranges {min(off_gbs):.2f} - {max(off_gbs):.2f} GB "
          f"({max(off_gbs) / sweep[0][6]:.1f}x our cap at worst)")

    print()
    print("=" * 78)
    print("B) same budget: is our greedy worse than the official rule?")
    print("   (both restricted to Pinned-Base: 32 base frozen + 4 redundant per rank)")
    print("=" * 78)
    alpha = 1.1
    w1 = zipf_load(NUM_LAYERS, NUM_EXPERTS, alpha, args.seed, phase_shift=args.drift)
    base = static_phy2log(NUM_LAYERS, RANKS, EPR, RED)

    t = time.perf_counter(); off_free = official(w1); t_free = time.perf_counter() - t
    t = time.perf_counter(); off_bound = redundant_only_official(w1); t_bound = time.perf_counter() - t
    t = time.perf_counter()
    ours, _, _, _ = plan_incremental_redundant(w1, RANKS, EPR, RED)
    t_ours = time.perf_counter() - t

    print(f"{'policy':<42}{'Max/Mean':>10}{'budget':>18}{'plan ms':>10}")
    print("-" * 80)
    for name, p2l, ms, budget in [
            ("Static EP", base, 0.0, "no replicas"),
            ("Official EPLB (unconstrained)", off_free, t_free * 1e3, "all 288 slots"),
            ("Official rule under Pinned-Base", off_bound, t_bound * 1e3, "32 redundant only"),
            ("Ours (peak-driven) under Pinned-Base", ours, t_ours * 1e3, "32 redundant only")]:
        print(f"{name:<42}{imbalance(p2l, w1):>10.3f}{budget:>18}{ms:>10.1f}")

    print("\nReading: row 2 vs row 3 is the price of the constraint;")
    print("         row 3 vs row 4 is our greedy vs the official rule at equal budget.")


if __name__ == "__main__":
    main()
