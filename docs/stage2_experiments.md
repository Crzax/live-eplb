# Stage 2 Experiments

One H20, torch 2.7.0+cu126, CUDA 12.9.

```bash
python tutorials/02_cudagraph_swap.py    # four experiments + timing
python tests/test_moe_layer.py           # 11 tests
```

## The problem

A CUDA Graph records a fixed sequence of kernel launches *with their arguments already
materialised*. Weight pointers are arguments. So if swapping an expert means making the model read
from a different allocation, every kernel that touched the old pointer is now wrong and the graph
must be captured again.

Measured cost of that re-capture, against one decode step on the same shard:

| operation | time | relative |
|---|---|---|
| `graph.replay()` (GPU, synced) | 2.265 ms | 1 decode step |
| `graph.replay()` (CPU enqueue only) | 0.009 ms | — |
| `commit_routing()` in place | 0.033 ms | **1.5% of a step** |
| full re-capture | 25.164 ms | **11.1x a step** |

Re-capturing costs eleven decode steps. Worse, it is synchronous and lands on whichever request is
being served at that moment.

> The two `replay()` rows are a reminder for anyone benchmarking this: without a `synchronize()` per
> iteration you measure kernel launch (0.009 ms), not execution (2.265 ms). The first draft of this
> script reported "53 overlapped replays in 0.5 ms", which was 53 enqueues into a queue the GPU had
> barely started.

## The design

Nothing the graph touches is ever reallocated. Three tensors are allocated once:

```
W_pool  [36, 3, 512, 2048] BF16    0.23 GB    one slot = 6.29 MB
log2phy [256, 8]           int32
logcnt  [256]              int32
cursor  [1]                int32
```

The captured forward does not take a weight pointer. It takes a *token-to-expert* assignment and
looks the weights up:

```python
def _route(self, expert):
    e = expert.long()
    same = expert.unsqueeze(1) == expert.unsqueeze(0)
    rank = same.tril(diagonal=-1).sum(1)               # [T] earlier tokens on same expert
    cnt = self.logcnt[e].long()
    k = (self.cursor[e] + rank) % cnt.clamp(min=1)     # [T] which replica
    return self.log2phy[e].long().gather(1, k.unsqueeze(1)).squeeze(1)

w = self.w_pool[self._route(expert)]                   # gather, not pointer arithmetic
```

Every address baked into the graph — `w_pool.data_ptr()`, `log2phy.data_ptr()`,
`cursor.data_ptr()` — stays valid forever. Swapping an expert is two in-place writes:

1. new bytes into a free slot in `W_pool`
2. a new slot index into `log2phy`

The graph is never told, and never needs to be.

## Experiment 1: the graph reproduces the eager forward

```
captures=1 replays=1
graph == eager                                  : True
token0 (expert 23 -> slot 23) == hand-computed  : True
```

The second line is the one that matters. `graph == eager` only proves the two agree; both could be
wrong together. The hand-computed reference recomputes SwiGLU from raw weights outside any graph:

```python
def reference(x, w):
    xv = x.to(dtype)
    return w[DOWN].t() @ (F.silu(w[GATE] @ xv) * (w[UP] @ xv))
```

This is the "comparative layer" that Stage 1's bug 4 showed was missing — a check that can only
pass if the computation is genuinely correct, not merely self-consistent.

## Experiment 2: why `cursor` must live on the GPU

The specification originally wrote the replica selection as `step % logcnt[expert]`. If `step` is a
Python integer, this silently breaks. Demonstrated directly:

```
set py_slot 0 -> 5, replay without re-capture:
  output still equals slot 0 (frozen)   : True
  output equals slot 5 (what we wanted) : False
```

At capture time `pool[py_slot]` is evaluated by the Python interpreter. What gets recorded is a
kernel reading `pool.data_ptr() + 0 * slot_bytes` — an immediate value. Rebinding the Python name
afterwards changes nothing, because the graph holds no reference to it.

The fix is to keep the index in device memory. A control-flow probe confirmed the asymmetry:

| index lives in | changed between replays | graph sees the change |
|---|---|---|
| Python int | `py_idx = 2` | **no** (constant-folded) |
| GPU tensor | `gpu_idx.fill_(2)` | **yes** (re-read each replay) |

Same for the routing table itself: `log2phy[0,0] = 3` retargets the lookup with no re-capture, and
bumping `logcnt[0] = 2` makes the cursor alternate between two slots.

This failure mode is nasty because it produces no error — just stale results, forever.

## Experiment 3: hot swap, zero re-capture

```
single slot (6.29 MB): 0 replays overlapped (copy finished within one step)
40-layer batch (252 MB): overlapped 1 full decode steps, step time 3.97-3.97 ms
expert 23: slot 23 -> slot 32
  token0 == new weights (hand-computed) : True
  token0 != old weights                 : True
  W_pool address unchanged              : 0x7f1668000000
  captures: 1 -> 1  (re-captures = 0)
```

Three assertions, deliberately redundant:

- the output matches the **new** weights, computed by hand
- the output does **not** match the old weights (guards against a no-op "pass")
- `captures` did not increment

A single 6.29 MB copy finishes inside one decode step, so to show the non-blocking property the
script also queues what a real 40-layer rebalance costs: 252 MB. That copy overlapped a complete
decode step which still took its normal 3.97 ms — the staging work ran on `transfer_stream` via
the copy engine and did not contend for SMs.

The reason overlapping is safe at all is the Pinned-Base invariant from Stage 1. Slot 32 was
holding `-1`, meaning **no routing table entry pointed at it**, so no token could reach it. Writing
to an unreachable slot needs no mutual exclusion with the forward pass. xLLM, which recomputes the
global placement, cannot assume this: a slot it wants to overwrite may be live, which is why it
needs staging buffers plus a two-phase prepare/activate protocol. Freezing the base slots buys that
complexity away.

## Experiment 4: replica round-robin

```
cursor[23]=0 -> slot 32    matches hand-computed : True
cursor[23]=1 -> slot 33    matches hand-computed : True
64 tokens all on expert 23 -> slots [32, 33], counts [32, 32]
spread within one step (not just across steps): True
```

Stage 1's planner assumes an expert with `c` replicas carries `w/c` per replica. If a second replica
never receives tokens it relieves nothing, and every Stage 1 number would be wrong.

**The first implementation got this wrong.** `cursor` was a single scalar shared by all experts, so
within one step every token selected the same replica index — the whole batch piled onto one slot
and the load only averaged out across steps. Since a MoE step is gated by the slowest rank, that
averaging is worthless: each individual step stays skewed.

Two things make the correct version work on a GPU. First, the obvious sequential formulation
("read `cursor[e]`, use it, increment it") cannot be parallelised — all 64 tokens are routed
simultaneously, so they would all observe the same value. Making it sequential would need atomics,
whose ordering is non-deterministic and unreproducible inside a graph.

So `rank` — how many earlier tokens in this batch hit the same expert — is **computed** instead of
accumulated:

```python
same = expert.unsqueeze(1) == expert.unsqueeze(0)   # [T, T]
rank = same.tril(diagonal=-1).sum(1)                # [T]
```

Second, `cursor` becomes per-expert and carries the remainder. With 3 tokens and 2 replicas the
split is 2:1; advancing `cursor` by the token count makes the next step 1:2, so the leftover
alternates rather than always favouring replica 0.

`rank` spreads within a step, `cursor` spreads across steps. The measurement above shows 64 tokens
on one expert splitting exactly 32/32.

## Tests

`tests/test_moe_layer.py`, 14 tests.

| test | asserts |
|---|---|
| `test_graph_matches_eager` | captured output equals eager output |
| `test_graph_matches_hand_computed` | **every** token matches the reference, not just the aggregate |
| `test_swap_takes_effect_without_recapture` | post-swap output matches new weights, `captures` unchanged |
| `test_swap_actually_changes_output` | output differs before vs after (a no-op swap must fail) |
| `test_addresses_pinned_across_swaps` | all four `data_ptr()` values survive filling every redundant slot |
| `test_replica_round_robin` | cursor 0..3 alternates across two replicas correctly |
| `test_batch_spreads_across_replicas` | 8 tokens on one expert with 4 replicas split 2/2/2/2 **within one step** |
| `test_cursor_carries_remainder_across_steps` | a 3-token / 2-replica split flips from 2:1 to 1:2 after advancing |
| `test_cursor_is_per_expert` | cursor has one entry per expert; `set_cursor` does not reallocate |
| `test_advance_cursor_counts_tokens` | each expert's offset moves by exactly its token count |
| `test_async_stage_then_commit` | staging on `transfer_stream`, polling, then committing yields the new weights |
| `test_untouched_experts_unaffected` | swapping one expert leaves every other token's output bit-stable |
| `test_reload_routing_keeps_address` | installing a whole new table keeps `log2phy.data_ptr()` |
| `test_reallocation_is_detected` | the pin assertion fires when a pinned tensor is rebound |

The last one tests the guard rather than the feature. `assert_addresses_pinned()` is the only thing
standing between a careless `self.log2phy = something` and a graph that reads freed memory, so the
guard needs its own test.

`test_swap_actually_changes_output` exists because of Stage 1's bug 3: a check of the form "does the
result match X" can pass trivially if the operation did nothing and the result already matched X.
Asserting the output *changed* closes that hole.

## Known limitations

- **One expert per token.** The real model routes top-8. The indirection mechanism is identical for
  top-k — gather 8 slots instead of 1 — but the kernel here is a simple per-token `bmm`, not the
  grouped GEMM a real engine would use. Absolute latency numbers are therefore not representative;
  the *ratios* (commit vs re-capture) are.
- **The 2.265 ms step is not a real decode step.** It is 64 tokens through one MoE layer with a
  naive bmm. A real 40-layer decode step includes attention, all-to-all and the shared expert.
  Re-capture cost (25 ms) is the number worth trusting, since it scales with graph size rather than
  with this toy kernel.
- **Single GPU.** `stage_expert_async` copies from local memory. Stage 3 replaces that with
  `cudaMemcpyPeerAsync` over NVLink, where the source lives in another process's address space.
- **No concurrency stress.** The commit here happens between replays on one thread. Stage 4's state
  machine has to handle a commit arriving while a replay is in flight, which is a stream-ordering
  problem this stage does not exercise.
