"""Stage 4: TPOT under a traffic shift, three policies, a full stack of layers.

torchrun --nproc_per_node=8 benchmarks/run_synthetic_8gpu.py

  static  - no balancing at all
  stw     - plan, copy synchronously, re-capture the graph (what EPLB costs today)
  live    - plan, stage in background, commit the table in place
"""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from live_eplb.controller import LiveEPLBController, rank_local_tables
from live_eplb.moe_layer import MoEStack
from live_eplb.p2p import P2PSwapper
from live_eplb.planner import plan_incremental_redundant, rank_load, static_phy2log

EXPERTS, RANKS, EPR, RED = 256, 8, 32, 4
SLOTS = EPR + RED
HIDDEN, MOE_INTER = 2048, 512


def zipf_probs(experts, alpha, shift):
    """-> probs [experts], Zipf(alpha) over a hot set rotated by `shift`.

    The permutation is seeded, not drawn per step: a hot set that moves every
    step makes every expert equally hot on average, and then no placement can
    beat the static one.
    """
    p = 1.0 / torch.arange(1, experts + 1, dtype=torch.float32) ** alpha
    perm = torch.randperm(experts, generator=torch.Generator().manual_seed(0))
    probs = torch.zeros(experts)
    probs[(perm + shift) % experts] = p / p.sum()
    return probs


def phase_probs(layers, alpha, shift):
    """-> probs [L, E]; layer l's hot set is rotated a further 8 experts.

    Layers that shared a hot set would all get the same plan, and then 40 layers
    would only be the single-layer experiment repeated.
    """
    return torch.stack([zipf_probs(EXPERTS, alpha, shift + 8 * l)
                        for l in range(layers)])


def build(rank, layers):
    """-> a MoEStack whose every layer holds experts [rank*EPR, (rank+1)*EPR)
    in its base slots, each slot tagged with its (layer, expert).

    Weights are drawn on the device: 40 layers of base experts is ~8 GB, which
    a host generator would spend minutes on.
    """
    stack = MoEStack(layers, EXPERTS, SLOTS, HIDDEN, MOE_INTER, max_replicas=8)
    l2p = torch.full((layers, EXPERTS, 8), -1, dtype=torch.int32)
    cnt = torch.zeros(layers, EXPERTS, dtype=torch.int32)
    mine = torch.arange(rank * EPR, (rank + 1) * EPR)
    l2p[:, mine, 0] = torch.arange(EPR, dtype=torch.int32)
    cnt[:, mine] = 1
    stack.load_routing(l2p.cuda(), cnt.cuda())

    g = torch.Generator(device="cuda").manual_seed(rank)
    for l in range(layers):
        lo = l * SLOTS
        stack.w_pool[lo:lo + EPR].copy_(
            torch.randn(EPR, 3, MOE_INTER, HIDDEN, generator=g, device="cuda",
                        dtype=torch.bfloat16) * 0.02)
        # (layer, expert) tag, read back by verify_replicas; both fit exactly in
        # bf16, which is integer-exact up to 256
        stack.w_pool[lo:lo + EPR, 0, 0, 0] = l
        stack.w_pool[lo:lo + EPR, 0, 0, 1] = mine.to("cuda", torch.bfloat16)
    return stack


def verify_replicas(stack, layers):
    """-> list of replica slots whose weights are not their own expert's.

    Checks the tag build() planted, so a pull that landed in the right slot of
    the wrong layer -- the way multi-layer breaks if the layer is dropped from a
    copy task -- shows up here rather than as slightly worse output.
    """
    l2p, cnt = stack.log2phy.cpu(), stack.logcnt.cpu()
    tag = stack.w_pool[:, 0, 0, :2].float().cpu()
    bad = []
    for l in range(layers):
        for e in range(EXPERTS):
            for s in l2p[l, e, :cnt[l, e]].tolist():
                got = (int(tag[l * SLOTS + s, 0]), int(tag[l * SLOTS + s, 1]))
                if got != (l, e):
                    bad.append((l, e, s, got))
    return bad


def local_tokens(ids, logcnt, size):
    """ids [L, T] -> [L, size] of experts reachable in each layer's own table.

    Fixed length so the captured graph keeps a static input shape; padded by
    repeating, which is what a real engine's bucketed batch does.
    """
    cnt = logcnt.cpu()
    out = torch.empty(ids.shape[0], size, dtype=ids.dtype)
    for l in range(ids.shape[0]):
        reach = ids[l][cnt[l, ids[l]] > 0]
        if reach.numel() == 0:
            reach = (cnt[l] > 0).nonzero()[:1, 0]
        out[l] = reach[torch.arange(size) % reach.numel()]
    return out


def run(policy, args, rank, world, stack, sw):
    """Drive `stack` for args.steps decode steps under `policy`.

    -> dict of per-step tpot [steps] in ms, ctrl_ms [steps] (the policy's own
    CPU time), imbalance [steps], actions [steps], commit counts, and for stw
    the plan / copy / recapture split of each stall.
    """
    torch.cuda.set_device(rank)
    L = args.layers
    gen = torch.Generator().manual_seed(1234)
    ctrl = None
    if policy == "live":
        ctrl = LiveEPLBController(sw, stack, num_layers=L, num_ranks=RANKS,
                                  experts_per_rank=EPR, redundant_per_rank=RED,
                                  threshold=args.threshold,
                                  warmup_steps=args.warmup,
                                  layers_per_step=args.layers_per_step,
                                  cooldown_steps=args.cooldown,
                                  check_every=args.check_every)

    base_phy2log = static_phy2log(L, RANKS, EPR, RED)
    ema = torch.zeros(L, EXPERTS)
    tpot, actions, imbalance, ctrl_ms, split = [], [], [], [], []
    stw_last, stw_commits = -args.cooldown, 0
    host_l = torch.full(tuple(stack.log2phy.shape), -1, dtype=torch.int32)
    host_c = torch.zeros(tuple(stack.logcnt.shape), dtype=torch.int32)

    if args.trace:
        td = torch.load(args.trace, weights_only=False)
        phase = [td["phases"][k]["counts"][:L].float() + 1e-6
                 for k in ("phaseA_code", "phaseB_math_wiki")]
    else:
        phase = [phase_probs(L, args.alpha, 0),
                 phase_probs(L, args.alpha, args.drift)]
    # counts are generated on the host here; stage them pinned, as a .cuda() on
    # pageable memory would block until the stream drains and charge the whole
    # decode step to the controller
    pin = torch.empty(L, EXPERTS).pin_memory()
    dev_counts = torch.empty(L, EXPERTS, device="cuda")
    x = torch.randn(args.tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids0 = torch.multinomial(phase[0], args.tokens, replacement=True, generator=gen)
    stack.capture(x, local_tokens(ids0, stack.logcnt, args.tokens).cuda().to(torch.int32))
    # exercise the planner once off the clock: first call pays numpy/BLAS import
    # and allocator warmup, which would otherwise land on whichever step plans
    plan_incremental_redundant(torch.ones(L, EXPERTS), RANKS, EPR, RED)
    for _ in range(3):
        stack.replay()
    torch.cuda.synchronize()

    for step in range(args.steps):
        ids = torch.multinomial(phase[0 if step < args.shift_at else 1],
                                args.tokens, replacement=True, generator=gen)
        counts = torch.zeros(L, EXPERTS).scatter_add_(
            1, ids, torch.ones(L, args.tokens))
        ema.mul_(0.9).add_(counts, alpha=0.1)
        stack.static_expert.copy_(
            local_tokens(ids, stack.logcnt, args.tokens).to(torch.int32))
        pin.copy_(counts)
        dev_counts.copy_(pin, non_blocking=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        stack.replay()

        act, t1 = "-", time.perf_counter()
        if policy == "live":
            act = ctrl.on_step(dev_counts)
        elif policy in ("stw", "stw_inplace") and step >= args.warmup \
                and step - stw_last >= args.cooldown:
            # same trigger and same plan as live; only the way it is applied
            # differs, and all of it sits on the critical path. stw re-captures
            # the graph as if the layout were baked into pointers; stw_inplace
            # owns the same routing-table indirection but applies it
            # synchronously, so its stall is plan + copy (+HBM contention) with
            # no re-capture and no retraction
            stats = ema * world
            load = rank_load(base_phy2log, stats, RANKS)
            per_layer = load.max(-1).values / load.mean(-1)
            hot = (per_layer > args.threshold).nonzero().flatten().tolist()
            if hot:
                a = time.perf_counter()
                p2l, l2p, _, tasks = plan_incremental_redundant(
                    stats, RANKS, EPR, RED, prev_phy2log=base_phy2log, layers=hot)
                mine = sw.local_tasks(tasks, SLOTS)
                b = time.perf_counter()
                if mine:
                    # drain the replay first: pull_batch waits on it anyway, and
                    # charging that wait to the copy made planning look free.
                    # for stw_inplace this drain also proves the copy cannot
                    # race a read, so no table has to be retracted first
                    torch.cuda.synchronize()
                    c = time.perf_counter()
                    sw.pull_batch(mine).synchronize()
                    d = time.perf_counter()
                    l, cn = rank_local_tables(l2p, rank, SLOTS, host_l, host_c)
                    if policy == "stw":
                        stack.load_routing(l.cuda(), cn.cuda())
                        stack.capture(stack.static_x, stack.static_expert, warmup=0)
                    else:
                        sw.commit(stack.log2phy, l.cuda(), stack.logcnt, cn.cuda())
                    torch.cuda.synchronize()
                    split.append((step, len(mine), (b - a) * 1e3, (c - b) * 1e3,
                                  (d - c) * 1e3, (time.perf_counter() - d) * 1e3))
                    stw_commits += 1
                    act = "recapture" if policy == "stw" else "commit"
                base_phy2log, stw_last = p2l, step

        t2 = time.perf_counter()
        torch.cuda.synchronize()
        tpot.append((time.perf_counter() - t0) * 1e3)
        ctrl_ms.append((t2 - t1) * 1e3)
        actions.append(act)

        # every rank sees the same traffic mix here, so the local EMA stands in
        # for the global load; only the layout in force differs per policy
        layout = ctrl.phy2log if policy == "live" else base_phy2log
        load = rank_load(layout, ema * world, RANKS)
        imbalance.append((load.max(-1).values / load.mean(-1).clamp(min=1e-9)).mean().item())

    return dict(policy=policy, tpot=tpot, actions=actions, imbalance=imbalance,
                ctrl_ms=ctrl_ms, commits=ctrl.commits if ctrl else stw_commits,
                plans=0 if ctrl is None else ctrl.plans, split=split,
                bad=verify_replicas(stack, L))


def stats(v):
    """v [steps] of per-step times -> dict of p50 / p99 / worst / mean."""
    t = torch.tensor(v)
    return dict(p50=t.median().item(), p99=t.quantile(0.99).item(),
                worst=t.max().item(), mean=t.mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--shift-at", type=int, default=80)
    # the forward gathers [tokens, 3, moe_inter, hidden] per layer, so tokens
    # sets both the step cost and the graph's peak memory
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.1)
    ap.add_argument("--drift", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=1.35)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--cooldown", type=int, default=20)
    ap.add_argument("--check-every", type=int, default=8)
    ap.add_argument("--layers-per-step", type=int, default=4)
    ap.add_argument("--policies", default="static,stw,stw_inplace,live")
    # real Qwen3.6 router decisions from collect_router_trace.py, replacing the
    # synthetic Zipf phases with phaseA_code / phaseB_math_wiki
    ap.add_argument("--trace")
    ap.add_argument("--out", default="results/stage4_timeline.json")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    policies = args.policies.split(",")

    out = {}
    for policy in policies:
        stack = build(rank, args.layers)
        sw = P2PSwapper(stack.w_pool, rank, world)
        sw.exchange_handles()
        dist.barrier()
        out[policy] = run(policy, args, rank, world, stack, sw)
        sw.close()
        del stack, sw
        torch.cuda.empty_cache()
        dist.barrier()

    if rank == 0:
        print(f"{args.layers} layers, {args.tokens} tokens/step, {args.steps} steps, "
              f"shift at {args.shift_at}, Zipf(alpha={args.alpha}) rotating "
              f"{args.drift} experts, peak "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB\n")
        hdr = (f"{'policy':<10}{'TPOT p50':>10}{'p99':>9}{'worst':>9}"
               f"{'Max/Mean last100':>18}{'commits':>9}")
        print(hdr)
        print("-" * len(hdr))
        for p in policies:
            r = out[p]
            s = stats(r["tpot"])
            # a single step's Max/Mean is mostly sampling noise; average the
            # steady state instead
            settled = torch.tensor(r["imbalance"][-100:]).mean().item()
            print(f"{p:<10}{s['p50']:>10.3f}{s['p99']:>9.3f}{s['worst']:>9.3f}"
                  f"{settled:>18.3f}{r['commits']:>9}")

        for p in policies:
            if out[p]["bad"]:
                print(f"\n!! {p}: {len(out[p]['bad'])} replica slots hold the "
                      f"wrong weights, first {out[p]['bad'][:3]}")
            if out[p]["split"]:
                base = torch.tensor(out["static"]["tpot"]).median().item() \
                    if "static" in out else 0.0
                print(f"\n{p} stalls (ms). drain = waiting out the replay this "
                      f"step launched, which every step pays; plan runs on the "
                      f"CPU beside it and is free whenever drain > 0.")
                print(f"  {'step':>5}{'slots':>7}{'plan':>8}{'drain':>8}"
                      f"{'copy':>8}{'recap':>8}{'TPOT':>8}{'vs p50':>8}")
                for st, n, pl, dr, cp, rc in out[p]["split"]:
                    t = out[p]["tpot"][st]
                    print(f"  {st:>5}{n:>7}{pl:>8.1f}{dr:>8.1f}{cp:>8.1f}"
                          f"{rc:>8.1f}{t:>8.1f}{t - base:>+8.1f}")

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({p: dict(tpot=out[p]["tpot"], imbalance=out[p]["imbalance"],
                               actions=out[p]["actions"], ctrl_ms=out[p]["ctrl_ms"],
                               split=out[p]["split"]) for p in out}, f)
        print(f"\nsaved {args.out}")

        if "stw" in out and "live" in out:
            worst_stw = stats(out["stw"]["tpot"])["worst"]
            worst_live = stats(out["live"]["tpot"])["worst"]
            print(f"worst single step: stw {worst_stw:.2f} ms vs live "
                  f"{worst_live:.2f} ms ({worst_stw / worst_live:.1f}x)")

        if "live" in out:
            a = out["live"]["actions"]
            print(f"live: {a.count('copying')} steps spent copying over "
                  f"{out['live']['commits']} commits, "
                  f"{a.count('planning')} spent planning")
            t = torch.tensor(out["live"]["tpot"])
            c = torch.tensor(out["live"]["ctrl_ms"])
            for i in (t - c).topk(3).indices.tolist() + t.topk(3).indices.tolist():
                print(f"  live step {i:>3} {out['live']['actions'][i]:<9} "
                      f"TPOT {t[i]:6.2f} ms, of which controller CPU {c[i]:6.2f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
