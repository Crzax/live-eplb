# Roadmap

Target: `Qwen/Qwen3.6-35B-A3B` on 8x H20 (NVLink all-to-all).
Verified config (`config.json`): `Qwen3_5MoeForConditionalGeneration`, MoE params nested under
`text_config` — 40 layers, `hidden_size=2048`, `moe_intermediate_size=512`, `num_experts=256`,
`num_experts_per_tok=8`, plus 1 shared expert. One routed expert = 6.29 MB BF16.

Constraint: core code stays under ~800 lines. Every stage must be runnable and measured — no
pseudo-code, no placeholder results.

As of Stage 4: **872 lines**, over the ~800 budget, across `controller.py` (371),
`moe_layer.py` (207), `planner.py` (170) and `p2p.py` (124), plus 136 lines of CUDA.
`eplb_official.py` is a vendored reference implementation used for comparison and is not counted.
The overrun came from the per-layer re-plan and the constructor warmups; it is not compressible
without dropping a correctness guard, so the budget is recorded as exceeded rather than met.

---

## Stage 1 — Bounded incremental planner — DONE

`live_eplb/planner.py`, `tutorials/01_eplb_diff.py`, `tutorials/01b_ablation.py`,
`tests/test_planner.py` (17/17, three of them covering the per-layer re-plan added in Stage 4).

Results and reasoning: see `README.md` and `docs/stage1_debug_notes.md`.

Key findings:
- The official EPLB's `arange` initialisation is an intermediate value; `balanced_packing` re-packs
  by load in step 3, so the final layout is unrelated to "the first N experts stay put".
- The root cause of full reshuffling is the missing "distance from previous layout" term in the
  objective, not a weak algorithm.
- The greedy step must be judged by *whether the global peak rank load drops*, not by
  `argmax(w_e/c_e)` + least-loaded rank. The latter has a positive feedback loop and makes uniform
  load worse than doing nothing (measured 1.062 vs 1.000).
- At most one replica of an expert per rank; more replicas on the same rank carry the same total
  load and only waste slots.
- The greedy must also respect each rank's slot capacity. Tracking "does rank r already hold e"
  is not enough: without a free-slot check the search assigned 7 replicas to a rank with 4 slots
  and the assembly step silently truncated them, so the planner optimised a layout it never
  emitted. All 13 tests passed; only the equal-budget ablation exposed it.
- Equal-budget ablation: restricted to the same 32 redundant slots, the official replica rule
  reaches 1.166 while the peak-driven greedy reaches 1.108. The remaining gap to the
  unconstrained 1.000 is the price of the constraint, not a weaker algorithm.

## Stage 2 — CUDA Graph in-place weight swap (single GPU) — DONE

`live_eplb/moe_layer.py`, `tutorials/02_cudagraph_swap.py`, `tests/test_moe_layer.py` (17/17,
three of them covering the `MoEStack` added in Stage 4).

Results: see `docs/stage2_experiments.md`. Headline numbers — `commit_routing()` 0.033 ms vs full
re-capture 25.164 ms (11.1x a decode step), zero re-captures across all four experiments, and a
252 MB staging batch overlapped a complete decode step that still took its normal 3.97 ms.

Key findings:
- A Python int index really is constant-folded at capture: setting `py_slot = 5` after capture
  leaves the output equal to slot 0 forever, with no error. A GPU-resident index is re-read on
  every replay. Verified both directions before writing the implementation.
- Overlapping the copy with inference is only safe because of the Pinned-Base invariant: an unused
  redundant slot holds -1, so no routing entry points at it and no token can reach it. xLLM needs
  staging buffers and a two-phase prepare/activate protocol precisely because it cannot assume
  the target slot is dead.
- Benchmark trap: `graph.replay()` without a per-iteration `synchronize()` measures enqueue
  (0.009 ms), not execution (2.265 ms). The first draft reported "53 replays in 0.5 ms", which was
  53 launches into a queue the GPU had barely begun.
- `assert_addresses_pinned()` needs its own test. It is the only thing between a careless
  `self.log2phy = ...` and a graph reading freed memory.
- Round-robin needs a per-expert cursor plus an in-batch rank. A single shared cursor sends the
  whole batch to one replica, so the w/c model only holds when averaged over steps — useless when
  each step is gated by its slowest rank. The sequential "read cursor, increment" form cannot be
  parallelised, so rank is computed from the batch instead of accumulated.

### Original spec, kept for reference

The core conflict: a CUDA Graph freezes every kernel argument — including weight pointers — at
capture time. If swapping an expert means pointing at different memory, the graph must be
re-captured (tens to hundreds of ms). That is exactly where stop-the-world comes from.

Two levels of indirection break it. The graph freezes *base addresses*, and those never change:

```
W_pool  [36, 3, 512, 2048] BF16   address fixed, contents mutable
log2phy [256, 4] int32            address fixed, contents mutable
logcnt  [256]    int32
captured: slot = log2phy[logical_id, cursor % logcnt[logical_id]]; matmul with W_pool[slot]
```

Tasks:
1. Build `W_pool` (36 = 32 base + 4 redundant) and the routing tables matching a single-GPU shard.
2. Capture a forward containing the in-graph indirect lookup plus the expert matmul.
3. Write a new expert into `W_pool[32]` on a separate `transfer_stream`, sync via event, update
   `log2phy` / `logcnt` in place, and show that the next `replay()` uses the new expert **with zero
   re-capture**.

Pitfalls to avoid:
- `cursor` must be a resident GPU tensor updated via `copy_` outside the graph. A Python int gets
  constant-folded at capture time.
- Capture on a side stream, and warm up a few iterations first, or cuBLAS lazy init gets recorded
  into the graph.
- Verification must be falsifiable: compare against a hand-computed expectation using the new
  weights, not merely "it did not crash".

Acceptance: output matches the hand-computed expectation; print replay count and re-capture count,
the latter must be 0.

## Stage 3 — NVLink P2P engine — DONE

`csrc/p2p_swapper.cu`, `setup.py`, `live_eplb/p2p.py`, `tutorials/03_p2p_bandwidth.py`,
`tests/test_p2p.py` (11/11 under torchrun).

Results: see `docs/stage3_experiments.md`. 8/8 ranks bit-identical pulls, commit latency 12.6 us,
40 scattered slots (252 MB) in 1.5 ms.

Key findings:
- **"Copy engine does not use SMs, therefore no interference" is wrong.** A 252 MB peer pull slows
  a concurrent GEMM by 2.4x; a 3 GB pull by 28x. The copy engine does not take SMs but it does
  write into local HBM, which the GEMM is reading. Still an order of magnitude better than a
  stop-the-world re-capture (25-57 ms measured in Stage 2), and the slowdown is gradual rather
  than a stall.
- Bandwidth is launch-bound for small copies: one 6.29 MB slot gets 147 GB/s against a 376 GB/s
  single-pair ceiling. Coalescing consecutive slots into one call reaches 272 GB/s. Scattered
  40-slot batches cannot coalesce, so they sit at 167 GB/s — and that is the realistic case.
- IPC + peer copy is **one-sided**: only the destination rank acts. NCCL send/recv needs both ends
  to call in, which would couple every rank's migration timing in Stage 4.
- New design freedom from the interference finding: spread the 40-layer migration across several
  decode steps to flatten the bandwidth spike. Only possible because the volume is bounded.

### Original spec, kept for reference

PyBind11 extension `live_eplb_cuda`:
- `get_ipc_handle(pool)` — `cudaIpcGetMemHandle`, exports the allocation base handle.
- `open_ipc_handle(handle, peer_rank)` — `cudaIpcOpenMemHandle`, maps and caches the peer pointer.
- `async_copy_expert_slot(local_pool, dst_slot, src_rank, src_slot, slot_bytes, stream_ptr)` —
  `cudaMemcpyPeerAsync`, one 6.29 MB slot over the NVLink copy engine (does not occupy SMs).
- `commit_routing_table(log2phy, new_log2phy, logcnt, new_logcnt, stream_ptr)` — single-launch
  lightweight kernel. **No atomics needed**: kernels on one stream are ordered, and "replay sees a
  complete new table" comes from stream ordering plus event sync. Atomics would only cost speed.

Pitfalls:
- Must be multi-process (`torchrun --nproc_per_node=2` minimum). `cudaIpcOpenMemHandle` on your own
  memory inside the same process fails.
- Exchange handles via `torch.distributed` object broadcast / all_gather.
- The handle exports the *allocation base*, not a tensor address. Compute `slot * slot_bytes`
  yourself and assert `is_contiguous()` and `storage_offset() == 0`.

Acceptance: bit-wise identical data after the copy; report single-slot and 40-layer bandwidth and
cross-check against `nvidia-smi topo -m`.

## Stage 4 — Async state machine + 8x H20 benchmark — DONE

`live_eplb/controller.py`, `live_eplb/moe_layer.py` (`MoEStack`),
`benchmarks/run_synthetic_8gpu.py`, `benchmarks/plot_stage4.py`,
`tests/test_controller.py` (11/11). Runs the full 40 layers of the target model.

Results: see `docs/stage4_experiments.md`. Headline — identical triggers and plans for all three
policies; **LiveEPLB's worst step is within jitter of not rebalancing at all** (66.143 vs 65.944 ms,
and that step was a cooldown step where the controller spent 0.07 ms) while stop-the-world's is
110% above (138.7 ms). Max/Mean over the last 100 steps 2.204 (static) → 1.166 (live) →
1.174 (stop-the-world) — but live stays worse than stop-the-world for ~45 steps after the shift.

Key findings:
- **Three slot id spaces, and only one of them carries the layer.** The planner works in global
  slots, a shard's routing table in layer-local ones, and a copy task in pool ids
  (`layer * spr + local`). Keeping the table layer-local is what lets the commit kernel, `_route`
  and `P2PSwapper` stay untouched from the single-layer version.
- **The stop-the-world cost is graph re-capture, nothing else.** Over 15 stalls,
  `sum(copy + re-capture)` is 811.6 ms against `sum(TPOT - p50)` of 812.0 ms, and re-capture is
  98.6% of it. Planning is free because it overlaps the replay the step launched anyway. One stall
  migrated a **single** expert and still paid 49.3 ms. So the thing to attack is not the volume of
  the copy but the existence of the re-capture.
- **A flat tail is paid for in recovery speed.** Steady-state balance matches stop-the-world, but
  for ~45 steps after a traffic shift LiveEPLB is worse (2.089 vs 1.283 at the peak): its first
  correction lands 11 steps later, because its cooldown clock only restarts at commit, and a
  per-layer re-plan needs several rounds. This is a trade, not a free win.
- **"N times better" is a function of layer count.** 38.7x at 1 layer, 2.1x at 40 — re-capture
  barely grows with depth while a decode step grows 39x. The scale-independent claim is the
  worst-step one.
- **A per-layer judgement must not drive a whole-stack action.** Redundant slots are per layer, so
  40 layers are 40 independent subproblems. The first version gated on the stack-average
  imbalance, which hides a badly skewed layer behind the rest: 11 skewed layers out of 40 cost 32%
  of step time while the average reads 1.32, under the 1.35 threshold. Switching the gate to `max`
  was worse — it never converges and re-triggers every cooldown. The fix is to re-plan only the
  layers above threshold, plus a guard that a re-plan may never raise a layer's peak rank load.
- **Only measuring TPOT cannot find a CPU-side stall.** Three of them (`self.ema.cpu()`, the first
  `commit` launch, the first `record()`) each blocked the caller for ~65 ms without moving TPOT by
  a microsecond, because each overlapped the replay it was waiting on. A separate CPU timer per
  step is what exposed them; pushing the scale from 1 to 40 layers is what made the first one
  large enough to notice.
- **"Off the critical path" is not one switch but four.** The same spike had to be chased four
  times: planning inline (33.7 ms), a GIL-holding Python loop inside the worker thread (113.9 ms
  against 4.9 ms of real planning), an `async_op=True` NCCL collective whose kernel spins on the
  GPU and exposes rank skew (up to 50 ms), and finally `threading.Thread.start()` itself on a
  loaded node (47 ms). Two more Python loops were later found inside the planner itself
  (66.9 ms and 9.6 ms), which is the same bug class recurring a month later.
- **A 1 KB all-reduce is not cheap because it is small.** The cost is synchronisation, not
  bandwidth: it stalls everyone behind the slowest rank. Moving it to a gloo group in a background
  thread, on an unconditional fixed cadence, removes both the GPU interference and the
  state-dependent deadlock risk.
- **Stage 2's "an unused slot receives no tokens" expires after the first migration.** The second
  plan overwrites slots that still serve the first plan's replicas, so the table must retract them
  first. This is exactly what xLLM's prepare/activate protocol is for; Pinned-Base removes it only
  for base slots.
- **A "has the statistic moved enough" gate can lock in a broken layout.** The first plan after a
  shift sees half-shifted statistics, fails to lower the imbalance, and the gate then reads
  "nothing changed, no need to re-plan".
- **The baseline must share the trigger.** Letting stop-the-world rebalance once is a strawman:
  its worse balance then comes from being called less often, not from the mechanism.
- Benchmark trap: re-drawing the hot set every step makes every expert equally hot on average, so
  no placement can win and the measured Max/Mean is pure sampling noise (1.28 measured vs 2.23
  structural). Offline recomputation of the same quantity is what caught it. Two more traps in the
  same file, both flattering this project: timing `pull_batch` without draining first charges the
  in-flight replay to the copy (which made an earlier "320 GB/s" figure accidentally right), and
  reading the controller's `phy2log` credits a new placement's balance from the start of COPYING
  rather than from the commit.

### Original spec, kept for reference

`LiveEPLBController`:
- EMA load tracking: `ema = 0.9 * ema + 0.1 * step_counts`
- Three states:
  - **IDLE** — when Max/Mean > 1.35, plan, issue async copies on `transfer_stream`, `event.record()`
  - **COPYING** — poll `event.query()` **without blocking** each decode step; if not done, keep
    replaying with the old table. Never wait a single step. This is where "zero downtime" lands.
  - **COMMIT** — once `query()` is true, call `commit_routing_table`
- The planner is deterministic, so every rank computes an identical plan independently —
  **no need to broadcast the placement**. Only the EMA statistics need an all-reduce (or all-gather,
  which moves less data).
 - Stage 3 measured that a bulk pull slows a concurrent step by ~2x, so consider pacing: migrate a
   couple of layers per step rather than all 40 at once. Bounded volume makes that schedulable.

300 decode steps, hot-expert shift injected at step 80, three policies, output
`results/synthetic_tpot_timeline.png`:
- Static EP (no balancing)
- Stop-The-World EPLB (synchronous copy + real graph re-capture — must actually re-capture, or the
  spike will not show)
- LiveEPLB (background migration + in-place table commit)

This plot is the central piece of evidence. Report TPOT P50 / P99 / worst single step.

## Stage 5 — Real router traces

Scope deliberately narrowed: loading the full 35B weights and monkey-patching an end-to-end
`generate()` is dropped. It is a large amount of work, easy to get stuck on HF version and
multimodal architecture details, and adds little over Stages 2-4.

Instead, do one thing: **replace the synthetic Zipf load with real router traces**, to show the
gains are not an artifact of the synthetic distribution.

1. Load the model (`trust_remote_code`), collect real `topk_ids [tokens, 8]` from all 40 gates via
   forward hooks. Domain shift: phase A code-generation prompts, phase B math / encyclopedia
   prompts, 50-100 each. Trace collection only — release the model afterwards.
2. Re-run the Stage 1 comparison and the Stage 4 benchmark on real traces. Report per-layer
   Max/Mean before and after the shift (heatmap), migration volume, and TPOT gains.

Migration volume, now measured rather than estimated: a Stage 4 commit at 40 layers moves between
**1 and 138 slots** (6 MB to 868 MB per rank) and the copy itself takes **0.1-3.0 ms**. Three
earlier figures in this file were wrong: "< 2 ms" was optimistic, the "6.57 GB / ~19 ms" correction
assumed a full redundant-slot refresh while the incremental planner only rewrites what changed,
and the "~320 GB/s" back-calculation was derived from a copy timing that had the in-flight replay
folded into it. The metric remains **TPOT jitter**, not total migration time: copying is 1.4% of a
stop-the-world stall and planning is free (it overlaps the replay the step launched anyway), so
**98.6% of the stall is graph re-capture**.

## Deliverables

- `results/synthetic_tpot_timeline.png` (core plot)
- `results/qwen3_6_layer_skew_heatmap.png`
- `README.md` — positioning, the TPOT plot, the numbers table, and an honest comparison with
  xLLM / SwiftBalancer stating which ideas are reproduced and which trade-off is ours.
