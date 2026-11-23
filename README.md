# LiveEPLB

MoE expert live migration for 8-way expert parallelism: **bounded-migration planning** plus
(work in progress) CUDA Graph in-place weight swap without re-capture.

Target model: `Qwen/Qwen3.6-35B-A3B` — 40 MoE layers, 256 routed experts, top-8 routing,
`moe_intermediate_size=512`. One routed expert is `3 * 2048 * 512 = 3.15M` params = **6.29 MB** in BF16.

---

## The problem

DeepSeek's [EPLB](https://github.com/deepseek-ai/EPLB) answers *where experts should go*, but not
*how to get there without stopping inference*. Its objective function contains only the balance of
the current snapshot — there is no term for "difference from the previous layout". Every call is
solved as a fresh cold-start problem, so the solution is discontinuous: a small change in load
produces a completely different placement.

Measured on layer 0 / GPU 0 after the hot expert set rotates by 64 experts:

```
t0: [138,  9, 86, 105, 253, 214, 251, 80, 67, 64, ...]
t1: [ 68, 68, 31, 160, 163, 219, 153, 118, 58, 77, ...]
```

All 36 slots changed. Globally 11520 of 11520 slots move — 70.8 GB of expert weights.

That is acceptable for the original setting (offline / low-frequency reconfiguration, roughly every
10 minutes). It is not acceptable for online adjustment every few dozen decode steps.

> Note: the `torch.arange` initialisation inside the official `replicate_experts` is only an
> intermediate value. `balanced_packing` in step 3 sorts by load and re-packs everything, so the
> final layout has nothing to do with "the first N experts stay put".

## The approach: Pinned-Base

Freeze the base slots. GPU `r` permanently owns logical experts `[32r, 32r+32)`. Only the trailing
4 redundant slots per GPU are planned, so:

- **migration volume has a hard upper bound** of 32 slots per layer, independent of traffic shape;
- the source address of every copy is a constant (`src_rank = e // 32`, `src_slot = e % 32`),
  so no lookup and no circular dependency between moves;
- base weights are read-only for the whole service lifetime, and empty redundant slots receive no
  tokens — writing into them needs no mutual exclusion with the forward pass.

The cost is a lower ceiling on balance: an expert can have at most 8 replicas (one per GPU), so
perfect balance is unreachable.

## Stage 1 results

40 layers x 256 experts, Zipf(alpha=1.1) load, hot set rotated by 64 experts between snapshots:

| policy | Max/Mean | MaxLoad | plan ms | slots moved | base moved | GB copied |
|---|---|---|---|---|---|---|
| Static EP (no balancing) | 2.369 | 14584 | 0.0 | 0 | 0 | 0.00 |
| Official EPLB (reshuffle) | **1.000** | 6136 | 443.4 | 11520 | 0 | 70.78 |
| LiveEPLB (incremental) | 1.359 | 8336 | 187.9 | **1069** | **0** | **6.57** |

**10.8x less data moved, zero base-slot churn, at the cost of Max/Mean 1.00 -> 1.36.**

This is a trade-off, not a strict improvement. The official policy is free to pack any expert
anywhere and reaches the theoretical optimum; we deliberately give that up for boundedness.

```bash
python tutorials/01_eplb_diff.py          # reproduce the table
python tests/test_planner.py              # 13 invariant tests
```

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | Bounded incremental planner vs official EPLB | **done** |
| 2 | Single-GPU CUDA Graph in-place weight swap, zero re-capture | todo |
| 3 | CUDA IPC + `cudaMemcpyPeerAsync` NVLink P2P engine | todo |
| 4 | Async state machine + 8xH20 TPOT benchmark | todo |
| 5 | Real router traces from Qwen3.6-35B-A3B | todo |

## Layout

```
live_eplb/
  planner.py          Pinned-Base bounded planner
  eplb_official.py    vendored DeepSeek EPLB, baseline only (MIT, see EPLB_LICENSE)
tutorials/
  01_eplb_diff.py     official full reshuffle vs bounded incremental
tests/
  test_planner.py     13 invariant tests
docs/
  stage1_debug_notes.md   three real bugs and what caught them
```

## Prior art

This is not the first attempt at online expert rebalancing. Two systems solve the same problem:

- **[vllm-ascend SwiftBalancer](https://github.com/vllm-project/vllm/issues/22246)** — moves the
  EPLB algorithm, p2p op construction and log2phy conversion into a separate worker to hide ~1s of
  host-bound latency; transfers weights on an async stream inside the gap between iterations.
- **[xLLM EPLB](https://docs.xllm-ai.com/en/features/eplb/)** — keeps active tensor objects and
  updates weights / `log2phy` in place so the ACL Graph address stays stable, therefore weight
  switching does not trigger re-capture; two-phase prepare/activate state machine with a benefit
  gate.

Both **recompute the global placement and then diff the changed slots**. Migration volume therefore
depends on traffic characteristics — xLLM has an explicit branch for "changed slots >= half", which
implies that case does occur in practice.

LiveEPLB constrains the search space up front instead, which makes the worst case a constant and
removes the need for staging buffers and a two-phase commit. The cost is quantified above.

## License

MIT. `live_eplb/eplb_official.py` is vendored from deepseek-ai/EPLB and remains under its original
MIT license (`live_eplb/EPLB_LICENSE`).
