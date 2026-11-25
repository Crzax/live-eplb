# LiveEPLB

A from-scratch MoE expert live-migration engine: bounded-volume rebalancing with in-place CUDA
Graph weight swap.

When a Mixture-of-Experts model is served with expert parallelism, each GPU owns a slice of the
experts. Real request traffic is long-tailed — a handful of experts absorb most of the tokens — so
the GPU holding them becomes the bottleneck for the entire batch while its peers idle. The standard
fix is to replicate hot experts into spare slots and periodically recompute the placement. DeepSeek
published [EPLB](https://github.com/deepseek-ai/EPLB), which answers *where every expert should go*.

It does not answer *how to get there without stopping inference*. Two costs appear the moment you
try to apply a new placement online. First, the new layout bears no resemblance to the old one, so
almost every expert's weights must be copied across GPUs. Second, a CUDA Graph freezes every kernel
argument — including weight pointers — at capture time, so pointing the model at relocated weights
forces a graph re-capture. Both happen while the engine is stopped, and both show up as a latency
spike for whoever is being served at that moment.

LiveEPLB attacks the first cost by constraining the planner so that migration volume has a hard
upper bound, and the second by never moving a pointer at all.

## Architecture

```
                        load statistics (EMA, all-reduced)
                                      |
                                      v
                        +---------------------------+
                        |   Pinned-Base planner     |   deterministic, runs on
                        |   (live_eplb/planner.py)  |   every rank independently
                        +---------------------------+
                                      |
                       copy_tasks: only redundant slots
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
       v                              v                              v
  +---------+                    +---------+                    +---------+
  | GPU 0   |                    | GPU 1   |       ...          | GPU 7   |
  |         |   NVLink P2P       |         |                    |         |
  | W_pool  |<------------------>| W_pool  |<------------------>| W_pool  |
  | [36,..] |  transfer_stream   | [36,..] |                    | [36,..] |
  +---------+                    +---------+                    +---------+
    0..31  base slots   frozen, read-only for the whole service lifetime
    32..35 redundant    the only slots the planner may touch

  compute_stream:  graph.replay()  ->  slot = log2phy[expert, cursor % logcnt[expert]]
                                       matmul with W_pool[slot]
  transfer_stream: cudaMemcpyPeerAsync into W_pool[32..35]  ->  event.record()
```

The forward pass reads a *routing table* instead of a weight pointer. `W_pool` and `log2phy` are
allocated once and never reallocated, so the addresses baked into the captured graph stay valid
forever. Swapping an expert means writing bytes into a slot and then writing a new slot index into
the table — the graph never notices.

## Key design decisions

- **Pinned-Base layout** — GPU `r` permanently owns logical experts `[32r, 32r+32)`. Only the 4
  trailing redundant slots per GPU are planned, so migration volume is bounded by 32 slots per
  layer regardless of how traffic changes.
- **Peak-driven greedy** — a replica is placed only if it lowers the *global maximum rank load*.
  Under uniform traffic no candidate qualifies and nothing is copied, which is correct: replication
  cannot help a balanced distribution.
- **Constant copy sources** — expert `e` always has its base replica at rank `e // 32`, slot
  `e % 32`. No lookup, and no circular dependency between moves, so no staging buffer is needed to
  break cycles.
- **Empty slots receive no tokens** — an unused redundant slot holds `-1` and is unreachable from
  the routing table, so writing weights into it needs no mutual exclusion with the forward pass.
- **Base weights are read-only** for the entire service lifetime, which shrinks the window where a
  read could race a write from 100% of slots to 11%.
- **Deterministic planning** — identical statistics produce an identical plan, so all 8 ranks can
  plan independently and skip broadcasting the placement; only the load statistics need an
  all-reduce.
- **The trigger is per layer, and so is the re-plan** — each layer's redundant slots are its own, so
  40 layers are 40 independent subproblems. Gating on the stack-average imbalance hides a layer that
  is badly skewed behind 39 that are fine: measured, 11 skewed layers out of 40 cost 32% of step
  time while the average reads 1.32, below the 1.35 threshold. The controller selects the layers
  above threshold and the planner re-plans only those, keeping the rest byte-identical.
- **A re-plan may never make a layer worse** — the greedy is optimal for one layer in isolation but
  not monotone against the previous placement, so each layer's new peak rank load is compared with
  the old one on the same statistics and the old one is kept on a tie. In a full re-plan this
  rejects 164 copies that were pure regression.
- **No atomics in the commit kernel** — kernels on one stream are ordered, so "replay sees a
  complete table" follows from stream ordering plus event sync. Atomics would only cost speed.
- **Statistics are reduced off the GPU, on a fixed step cadence** — the payload is 1 KB, but an
  NCCL collective spins on the GPU until the slowest rank joins and steals SMs from the decode
  step; and a reduction gated by each rank's own state machine can deadlock. A gloo group inside
  a background thread removes both.
- **A redundant slot is unlinked from the routing table before it is overwritten** — "an unused
  slot receives no tokens" only holds for the *first* migration. From the second one on, the
  target slot is serving the previous plan's replica.

## Modules

| Path | Role |
|---|---|
| `live_eplb/planner.py` | Pinned-Base bounded planner: placement, inverse routing tables, copy-task diff |
| `live_eplb/eplb_official.py` | DeepSeek EPLB, vendored verbatim, used only as the comparison baseline |
| `tutorials/01_eplb_diff.py` | Official full reshuffle vs bounded incremental planning |
| `tutorials/01b_ablation.py` | Skew sweep, and the same-budget constraint-vs-algorithm ablation |
| `tests/test_planner.py` | 17 invariant tests for the planner |
| `live_eplb/moe_layer.py` | MoE shard with address-pinned `W_pool` / `log2phy`, graph capture, async staging |
| `tutorials/02_cudagraph_swap.py` | In-place weight swap under a live graph, plus commit-vs-recapture timing |
| `tests/test_moe_layer.py` | 17 tests for graph validity across swaps |
| `csrc/p2p_swapper.cu` | CUDA IPC handles, `cudaMemcpyPeerAsync`, single-launch routing commit |
| `live_eplb/p2p.py` | Handle exchange, coalesced peer pulls, task filtering per rank |
| `tutorials/03_p2p_bandwidth.py` | Cross-process correctness, NVLink bandwidth, HBM interference |
| `tests/test_p2p.py` | 11 multi-process tests |
| `live_eplb/controller.py` | Async state machine: IDLE -> PLANNING -> COPYING -> commit, all off the critical path |
| `benchmarks/run_synthetic_8gpu.py` | 8x H20 TPOT benchmark, three policies under a traffic shift |
| `tests/test_controller.py` | 11 state-machine tests, single GPU |

## Installation

Stage 1 is pure Python and needs no GPU:

```bash
git clone https://github.com/Crzax/live-eplb.git
cd live-eplb
pip install torch numpy
```

> Requires Python >= 3.10 and PyTorch >= 2.1. Later stages additionally need CUDA >= 12.0 with
> `nvcc` on `PATH`, and 8 NVLink-connected GPUs for the full benchmark.

## Quick Start

```bash
python tutorials/01_eplb_diff.py
```

```
Qwen3.6-35B-A3B: 40 MoE layers x 256 experts, EP=8, 32 base + 4 redundant slots/rank = 288 physical slots
expert slot = 6.29 MB BF16; traffic Zipf(alpha=1.1), hot set rotates by 64 experts between snapshots

policy                        Max/Mean    MaxLoad   plan ms  slots moved  base moved  GB copied
-----------------------------------------------------------------------------------------------
Static EP (no balancing)         2.369      14584       0.0            0           0       0.00
Official EPLB  plan@t0           1.000       6037     450.3            0           0       0.00
Official EPLB  reshuffle         1.000       6136     449.7        11520           0      70.78
LiveEPLB       plan@t0           1.090       6570     188.3            0           0       0.00
LiveEPLB       incremental       1.108       6796     188.9          984           0       6.05

--- copy plan for the traffic shift (snapshot t0 -> t1) ---
LiveEPLB :   984 tasks, 6.05 GB  (77% of redundant slots)
Official : 11520 tasks, 70.78 GB  (full reshuffle)

correctness: every expert reachable, per-layer token count conserved [OK]
```

Then the ablations, and the tests:

```bash
python tutorials/01b_ablation.py
python tests/test_planner.py
```

Stage 2 needs one GPU. It swaps an expert under a live CUDA Graph and checks the result against a
reference computed outside the graph:

```bash
python tutorials/02_cudagraph_swap.py
python tests/test_moe_layer.py
```

```
3. swap an expert into a redundant slot, zero re-capture
  40-layer batch (252 MB): overlapped 1 full decode steps, step time 3.97-3.97 ms
  expert 23: slot 23 -> slot 32
    token0 == new weights (hand-computed): True
    token0 != old weights                : True
    W_pool address unchanged             : 0x7f1668000000
    captures: 1 -> 1  (re-captures = 0)

5. in-place commit vs full re-capture
  graph.replay() GPU       2.265 ms   (one decode step, synced)
  commit_routing()         0.033 ms   (1.5% of a step)
  full re-capture         25.164 ms   (11.1x a step)
```

Stage 3 needs the CUDA extension and multiple processes:

```bash
pip install -e . --no-build-isolation
torchrun --nproc_per_node=8 tutorials/03_p2p_bandwidth.py
torchrun --nproc_per_node=4 tests/test_p2p.py
```

```
1. correctness
   ranks with bit-identical pull: 8/8
2. bandwidth (NV18 all-to-all, 18 links/pair)
   1 slot            6.29 MB    0.043 ms    147.6 GB/s
   4 contiguous     25.17 MB    0.092 ms    272.3 GB/s
   40 scattered     251.7 MB    1.507 ms    167.0 GB/s
3. commit_routing_table (single launch, no atomics)
   commit latency         : 12.6 us
4. interference with compute
   step alone                  : 0.280 ms
   step during 252 MB pull     : 0.667 ms  (2.38x)
   total extra latency         : 0.39 ms
```

Stage 4 is the end-to-end result: **40 MoE layers**, 300 decode steps on 8x H20, hot-expert set
rotated at step 80. Every layer gets its own hot set, so every layer is planned independently.

```bash
torchrun --nproc_per_node=8 benchmarks/run_synthetic_8gpu.py --layers 40
python benchmarks/plot_stage4.py 80
```

![TPOT under a traffic shift](results/synthetic_tpot_timeline.png)

```
policy      TPOT p50      p99    worst  Max/Mean last100  commits
-----------------------------------------------------------------
static        64.792   65.735   65.944             2.204        0
stw           64.616  121.822  138.711             1.174       15
live          64.581   65.904   66.143             1.166       12
```

All three policies share the same trigger, the same planner and the same copy tasks — only the
way a new placement is *applied* differs.

**LiveEPLB's worst step is within jitter of not rebalancing at all** — 66.143 ms vs static's
65.944, i.e. +0.199 ms, and that worst step was a *cooldown* step where the controller spent
0.07 ms, so the difference is not attributable to rebalancing. Stop-the-world's worst step is 110%
above. Over the 15 stalls, the stop-the-world cost is almost entirely graph re-capture:

```
   step  slots    plan   drain    copy   recap    TPOT  vs p50
     10    125    34.3    29.4     2.7    71.9   138.7   +73.9
     90    138    38.0    26.3     3.0    51.4   119.1   +54.3
    251      1     2.1    62.0     0.3    49.3   114.2   +49.4

sum(copy + recap)  811.6 ms      <- the whole stall
sum(TPOT - p50)    812.0 ms         re-capture is 98.6% of it
```

`plan + drain` is a constant ~64 ms on every row, which is one decode step: planning hides behind
the replay that step launched anyway, so it is free. What is left is re-capture. **Step 251
migrated a single expert and still paid 49.3 ms** — the stall barely depends on how much moved,
which is exactly what LiveEPLB sidesteps: it does not make re-capture faster, it makes it not
happen.

The trade is not free. Steady-state balance is the same (2.204 → 1.17 for both, a 47% cut in peak
rank load), but **for ~45 steps after the traffic shift LiveEPLB is measurably worse balanced than
stop-the-world** — 2.089 vs 1.283 at the peak (step 100), because its first correction lands 11
steps later and a per-layer re-plan needs several rounds to converge. It buys a flat tail and pays
for it in recovery speed; which side wins depends on whether the SLO is written against latency or
throughput, and this stage does not answer that.

Note that "N times better" is highly sensitive to layer count: the same experiment at 1 layer gives
38.7x (75.5 ms worst step vs 2.0 ms), because re-capture barely grows from 1 to 40 layers while a
decode step grows 39x. The scale-independent claim is the first one. Full analysis — including why
the per-layer trigger replaced a stack-average one, and four measurement bugs that each flattered
this project — is in [`docs/stage4_experiments.md`](docs/stage4_experiments.md).

## The trade-off

The planner is not strictly better than the official one — it occupies a different point:

```
Official EPLB:  optimal balance  +  unbounded migration  +  needs cycle-breaking buffers
LiveEPLB:       near balance     +  bounded migration    +  copies are independent writes
```

Concretely, at `alpha = 1.1`:

| | Official | LiveEPLB |
|---|---|---|
| Max/Mean | **1.000** | 1.108 |
| data moved per shift | 70.78 GB | **6.05 GB** |
| worst-case volume | unbounded (full reshuffle) | **7.86 GB, traffic-independent** |
| planning time | 450 ms | **188 ms** |

**11.7x less data moved for 10.8% worse balance.** And the balance gap is caused by the constraint,
not by a weaker algorithm — running the official replica-selection rule inside the same Pinned-Base
budget yields 1.166, which our peak-driven greedy beats at 1.108. Full numbers and methodology in
[`docs/stage1_experiments.md`](docs/stage1_experiments.md).

Why bounding matters more than averaging: the official policy's cost depends on traffic shape, and
its worst case is a complete reshuffle. xLLM carries an explicit code path for "changed slots >=
half", which implies that case occurs in production. Our cost is a constant you can put in a
latency budget.

## Status

| Stage | Scope | State |
|---|---|---|
| 1 | Bounded incremental planner vs official EPLB | **done** |
| 2 | Single-GPU CUDA Graph in-place swap, zero re-capture | **done** |
| 3 | CUDA IPC + `cudaMemcpyPeerAsync` NVLink P2P engine | **done** |
| 4 | Async state machine + 8x H20 TPOT benchmark | **done** |
| 5 | Real router traces from Qwen3.6-35B-A3B | todo |

Target model: `Qwen/Qwen3.6-35B-A3B` — 40 MoE layers, 256 routed experts, top-8 routing,
`moe_intermediate_size=512`, plus 1 shared expert. One routed expert is
`3 * 2048 * 512 = 3.15M` parameters, **6.29 MB** in BF16.

## Project Structure

```
live-eplb/
├── csrc/
│   └── p2p_swapper.cu          <- CUDA IPC, peer copy, routing commit kernel
├── live_eplb/
│   ├── planner.py              <- Pinned-Base planner, peak-driven greedy
│   ├── moe_layer.py            <- address-pinned MoE shard, graph capture, hot swap
│   ├── p2p.py                  <- cross-process pulls over NVLink
│   ├── controller.py           <- async state machine, background planning and staging
│   ├── eplb_official.py        <- vendored DeepSeek EPLB (MIT), baseline only
│   └── EPLB_LICENSE
├── tutorials/
│   ├── 01_eplb_diff.py         <- main comparison, writes results/stage1_plans.pt
│   ├── 01b_ablation.py         <- alpha sweep + equal-budget ablation
│   ├── 02_cudagraph_swap.py    <- in-place swap under a live graph, timing
│   └── 03_p2p_bandwidth.py     <- peer copy correctness, bandwidth, interference
├── benchmarks/
│   ├── run_synthetic_8gpu.py   <- 8x H20, three policies, 300 decode steps
│   └── plot_stage4.py          <- writes results/synthetic_tpot_timeline.png
├── tests/
│   ├── test_planner.py         <- 17 invariant tests
│   ├── test_moe_layer.py       <- 17 graph-validity tests
│   ├── test_p2p.py             <- 11 multi-process tests
│   └── test_controller.py      <- 11 state-machine tests
├── docs/
│   ├── stage1_experiments.md   <- workload model, all measurements, limitations
│   ├── stage1_debug_notes.md   <- four real bugs and what caught each one
│   ├── stage2_experiments.md   <- indirection design, the Python-int trap, timing
│   ├── stage2_debug_notes.md   <- async timing trap, shared-cursor design bug
│   ├── stage3_experiments.md   <- IPC, bandwidth vs coalescing, HBM contention
│   ├── stage3_debug_notes.md   <- parameter-semantics drift, a wrong hardware assumption
│   ├── stage4_experiments.md   <- the TPOT benchmark, what "zero downtime" costs
│   └── stage4_debug_notes.md   <- fifteen bugs, four rounds of chasing the same spike
├── setup.py
└── ROADMAP.md                  <- per-stage specs and known pitfalls
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/stage1_experiments.md`](docs/stage1_experiments.md) | Zipf workload model, main comparison, two ablations, correctness checks, known limitations |
| [`docs/stage1_debug_notes.md`](docs/stage1_debug_notes.md) | Four bugs found during Stage 1, including two that passed every structural test while silently producing wrong plans |
| [`docs/stage2_experiments.md`](docs/stage2_experiments.md) | Why a graph freezes pointers, the two-level indirection, the Python-int trap, commit vs re-capture timing |
| [`docs/stage2_debug_notes.md`](docs/stage2_debug_notes.md) | Two more bugs: async timing measured as sync, and a shared cursor that made replication a no-op within a step |
| [`docs/stage3_experiments.md`](docs/stage3_experiments.md) | CUDA IPC vs NCCL send/recv, why coalescing doubles bandwidth, how much a peer copy actually costs a decode step |
| [`docs/stage3_debug_notes.md`](docs/stage3_debug_notes.md) | A parameter whose meaning drifted, and a hardware assumption ("copy engine does not touch SMs, therefore no interference") that measured 2.4x wrong |
| [`docs/stage4_experiments.md`](docs/stage4_experiments.md) | The TPOT benchmark, why the baseline shares the trigger, where "zero downtime" actually lands, and one number left unexplained |
| [`docs/stage4_debug_notes.md`](docs/stage4_debug_notes.md) | Fifteen bugs: four rounds of chasing the same spike (critical path, GIL, NCCL SM contention, thread startup), a routing race under repeated migration, a workload model that quietly cancelled its own result, a trigger that averaged away the thing it was gating, and three CPU-side stalls that TPOT could not see |
| [`ROADMAP.md`](ROADMAP.md) | Technical spec for each stage, plus the pitfalls worth knowing before writing the code |

## Prior art

Online expert rebalancing is not a new idea, and two production systems solve the same problem:

- **[SwiftBalancer](https://github.com/vllm-project/vllm/issues/22246)** (vllm-ascend) moves the
  EPLB algorithm, p2p op construction and `log2phy` conversion into a separate worker to hide ~1s of
  host-bound latency, and transfers weights on an async stream inside the gap between iterations.
- **[xLLM EPLB](https://docs.xllm-ai.com/en/features/eplb/)** keeps active tensor objects and
  updates weights and `log2phy` in place so the ACL Graph's addresses stay stable, meaning a weight
  switch does not trigger re-capture. It drives migration through a two-phase prepare/activate
  state machine with a benefit gate.

Both **recompute the global placement, then diff the changed slots** — migration volume is reduced
after the fact, and therefore still depends on traffic. LiveEPLB constrains the search space before
planning instead, which turns the worst case into a constant and removes the need for staging
buffers and a two-phase commit. The price is the balance ceiling quantified above.

## License

MIT. `live_eplb/eplb_official.py` is vendored from
[deepseek-ai/EPLB](https://github.com/deepseek-ai/EPLB) and stays under its original MIT license
(see `live_eplb/EPLB_LICENSE`).
