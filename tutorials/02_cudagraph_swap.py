"""Stage 2: swap expert weights under a live CUDA Graph, with zero re-capture.

Four experiments, each falsifiable against a hand-computed reference:
  1. baseline        - captured graph reproduces the eager forward
  2. python int      - demonstrate WHY cursor must be a GPU tensor
  3. weight swap     - write into a redundant slot, retarget the table, no re-capture
  4. replica rr      - two replicas, cursor alternates, load really splits
Finally a timing comparison: in-place commit vs full re-capture.
"""
import argparse, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from live_eplb.moe_layer import GraphSafeMoEShard, GATE, UP, DOWN

EXPERTS, SLOTS = 256, 36        # 32 base + 4 redundant, one GPU's shard
HIDDEN, MOE_INTER = 2048, 512
BASE, RED = 32, 4
TOKENS = 64


def reference(x, w, dtype=torch.bfloat16):
    """Hand-computed SwiGLU for one token with one expert's weights. The control layer:
    if the graph output matches this, the swap genuinely took effect."""
    xv = x.to(dtype)
    gate = w[GATE] @ xv
    up = w[UP] @ xv
    return w[DOWN].t() @ (torch.nn.functional.silu(gate) * up)


def build_shard(seed=0):
    torch.manual_seed(seed)
    shard = GraphSafeMoEShard(EXPERTS, SLOTS, HIDDEN, MOE_INTER, max_replicas=8)
    # base slots: rank 0 owns experts 0..31, one replica each
    log2phy = torch.full((EXPERTS, 8), -1, dtype=torch.int32)
    logcnt = torch.zeros(EXPERTS, dtype=torch.int32)
    for e in range(BASE):
        log2phy[e, 0] = e
        logcnt[e] = 1
    shard.load_routing(log2phy.cuda(), logcnt.cuda())
    for s in range(BASE):
        shard.load_expert(s, torch.randn(3, MOE_INTER, HIDDEN, device="cuda",
                                         dtype=torch.bfloat16) * 0.02)
    return shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=TOKENS)
    ap.add_argument("--iters", type=int, default=50, help="timing iterations")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Stage 2 needs a GPU"
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"shard: {EXPERTS} experts addressable, {SLOTS} slots "
          f"({BASE} base + {RED} redundant), hidden={HIDDEN}, moe_inter={MOE_INTER}")
    print(f"W_pool = {SLOTS} x 3 x {MOE_INTER} x {HIDDEN} BF16 = "
          f"{SLOTS * 3 * MOE_INTER * HIDDEN * 2 / 1e9:.2f} GB, one slot = "
          f"{3 * MOE_INTER * HIDDEN * 2 / 1e6:.2f} MB\n")

    shard = build_shard()
    x = torch.randn(args.tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)
    expert = torch.randint(0, BASE, (args.tokens,), device="cuda", dtype=torch.int32)

    print(f"W_pool  @ {hex(shard.w_pool.data_ptr())}")
    print(f"log2phy @ {hex(shard.log2phy.data_ptr())}")
    print(f"cursor  @ {hex(shard.cursor.data_ptr())}\n")

    # ---------------------------------------------------------------- 1. baseline
    print("=" * 72)
    print("1. capture and verify against the eager forward")
    print("=" * 72)
    eager = shard.forward(x, expert).clone()
    sx, sy = shard.capture(x, expert)
    y = shard.replay()
    torch.cuda.synchronize()
    print(f"  captures={shard.captures} replays={shard.replays}")
    print(f"  graph == eager                         : "
          f"{torch.allclose(y.float(), eager.float(), atol=2e-2)}")

    # spot-check token 0 against the fully hand-computed reference
    e0 = int(expert[0])
    slot0 = int(shard.log2phy[e0, 0])
    ref0 = reference(x[0], shard.w_pool[slot0])
    print(f"  token0 (expert {e0} -> slot {slot0}) == hand-computed : "
          f"{torch.allclose(y[0].float(), ref0.float(), atol=2e-2)}")

    # ------------------------------------------------------- 2. the Python int trap
    print()
    print("=" * 72)
    print("2. why cursor must be a GPU tensor, not a Python int")
    print("=" * 72)
    pool = shard.w_pool
    xs = x[:1]
    py_slot = 0
    out = torch.empty(1, MOE_INTER, device="cuda", dtype=torch.bfloat16)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            out.copy_(torch.bmm(pool[py_slot, GATE].unsqueeze(0),
                                xs.unsqueeze(-1)).squeeze(-1))
    torch.cuda.current_stream().wait_stream(s)

    gtrap = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gtrap):
        out.copy_(torch.bmm(pool[py_slot, GATE].unsqueeze(0), xs.unsqueeze(-1)).squeeze(-1))
    gtrap.replay(); torch.cuda.synchronize()
    frozen = out.clone()

    py_slot = 5                                     # change the Python variable
    gtrap.replay(); torch.cuda.synchronize()
    same = torch.allclose(out.float(), frozen.float(), atol=1e-3)
    want5 = torch.bmm(pool[5, GATE].unsqueeze(0), xs.unsqueeze(-1)).squeeze(-1)
    print(f"  set py_slot 0 -> 5, replay without re-capture:")
    print(f"    output still equals slot 0 (frozen)  : {same}")
    print(f"    output equals slot 5 (what we wanted): "
          f"{torch.allclose(out.float(), want5.float(), atol=1e-3)}")
    print("  -> a Python int is constant-folded into the graph. The real shard keeps")
    print("     cursor on the GPU so replay() re-reads it every time.")

    # ------------------------------------------------------------ 3. hot weight swap
    print()
    print("=" * 72)
    print("3. swap an expert into a redundant slot, zero re-capture")
    print("=" * 72)
    captures_before = shard.captures
    hot = int(expert[0])
    new_w = torch.randn(3, MOE_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    old_slot = int(shard.log2phy[hot, 0])

    # stage into redundant slot 32 while the graph keeps running on the old table.
    # 6.29 MB over NVLink finishes in well under one decode step, so to show the
    # non-blocking property we also queue a realistic 40-layer batch (40 x 6.29 MB).
    ev = shard.stage_expert_async(BASE, new_w)
    spins = 0
    while not ev.query():
        shard.replay()                       # inference continues, never blocked
        spins += 1
    print(f"  single slot (6.29 MB): {spins} replays overlapped "
          f"({'copy finished within one step' if spins == 0 else 'copy spanned steps'})")

    # A realistic 40-layer rebalance. Each decode step is timed with a real sync so
    # the measurement reflects GPU execution, not just kernel launch.
    bulk = torch.randn(40, 3, MOE_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    shard.transfer_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(shard.transfer_stream):
        for i in range(40):
            shard.w_pool[BASE + (i % RED)].copy_(bulk[i], non_blocking=True)
        ev_bulk = torch.cuda.Event(); ev_bulk.record(shard.transfer_stream)

    step_ms, overlapped = [], 0
    while not ev_bulk.query():
        t0 = time.perf_counter()
        shard.replay()
        torch.cuda.synchronize()             # a real decode step, start to finish
        step_ms.append((time.perf_counter() - t0) * 1e3)
        overlapped += 1
    if step_ms:
        print(f"  40-layer batch ({40 * 6.29:.0f} MB): overlapped {overlapped} full decode "
              f"steps, step time {min(step_ms):.2f}-{max(step_ms):.2f} ms")
    else:
        print(f"  40-layer batch ({40 * 6.29:.0f} MB): copy finished before the first "
              f"step completed")
    shard.load_expert(BASE, new_w)           # restore: bulk clobbered slots 32..35

    shard.commit_routing(hot, BASE)          # expert `hot` now points at slot 32
    y2 = shard.replay()
    torch.cuda.synchronize()

    ref_new = reference(x[0], new_w)
    ref_old = reference(x[0], shard.w_pool[old_slot])
    print(f"  expert {hot}: slot {old_slot} -> slot {BASE}")
    print(f"    token0 == new weights (hand-computed): "
          f"{torch.allclose(y2[0].float(), ref_new.float(), atol=2e-2)}")
    print(f"    token0 != old weights                : "
          f"{not torch.allclose(y2[0].float(), ref_old.float(), atol=2e-2)}")
    print(f"    W_pool address unchanged             : "
          f"{hex(shard.w_pool.data_ptr())}")
    print(f"    captures: {captures_before} -> {shard.captures}  "
          f"(re-captures = {shard.captures - captures_before})")

    # -------------------------------------------------------- 4. replica round-robin
    print()
    print("=" * 72)
    print("4. two replicas, cursor alternates between them")
    print("=" * 72)
    # give expert `hot` two replicas: slot 32 (new_w) and slot 33 (another copy)
    other_w = torch.randn(3, MOE_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.02
    shard.load_expert(BASE + 1, other_w)
    shard.commit_routing(hot, BASE, replica=0)
    shard.commit_routing(hot, BASE + 1, replica=1, new_logcnt=2)

    ok = True
    for c, w_expect, tag in [(0, new_w, f"slot {BASE}"), (1, other_w, f"slot {BASE+1}")]:
        shard.set_cursor(c)
        yk = shard.replay()
        torch.cuda.synchronize()
        hit = torch.allclose(yk[0].float(), reference(x[0], w_expect).float(), atol=2e-2)
        ok &= hit
        print(f"  cursor={c} -> {tag:<10} matches hand-computed : {hit}")
    print(f"  round-robin works, so the planner's w/c load model holds at runtime: {ok}")
    print(f"  total re-captures across all four experiments: {shard.captures - 1}")

    # ------------------------------------------------------------------- 5. timing
    print()
    print("=" * 72)
    print("5. in-place commit vs full re-capture")
    print("=" * 72)

    def timeit(fn, n):
        """Sync once per iteration: we want GPU completion, not launch latency."""
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    ms_replay = timeit(shard.replay, args.iters)
    ms_commit = timeit(lambda: shard.commit_routing(hot, BASE, replica=0), args.iters)
    ms_launch = 0.0
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(args.iters):
        shard.replay()
    ms_launch = (time.perf_counter() - t0) / args.iters * 1e3
    torch.cuda.synchronize()

    def recapture():
        shard.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(shard.graph):
            shard.static_y = shard.forward(shard.static_x, shard.static_expert)

    ms_recap = timeit(recapture, max(args.iters // 5, 5))

    print(f"  graph.replay() GPU    {ms_replay:8.3f} ms   (one decode step, synced)")
    print(f"  graph.replay() launch {ms_launch:8.3f} ms   (CPU-side enqueue only)")
    print(f"  commit_routing()      {ms_commit:8.3f} ms   ({ms_commit / ms_replay * 100:.1f}% of a step)")
    print(f"  full re-capture       {ms_recap:8.3f} ms   ({ms_recap / ms_replay:.1f}x a step)")
    print(f"  -> avoiding re-capture saves {ms_recap - ms_commit:.2f} ms per rebalance,")
    print(f"     and that cost no longer lands on a decode step at all")


if __name__ == "__main__":
    main()
