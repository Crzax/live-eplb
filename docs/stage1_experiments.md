# Stage 1 Experiments

All numbers from one 8x H20 node (torch 2.7.0+cu126, CUDA 12.9). The planner itself is CPU-only,
so these are reproducible anywhere.

Reproduce:

```bash
python tutorials/01_eplb_diff.py     # main comparison
python tutorials/01b_ablation.py     # alpha sweep + equal-budget ablation
python tests/test_planner.py         # 14 invariant tests
```

## Workload model

Real MoE routing is long-tailed: a minority of experts absorb most tokens. We model that with a
Zipf distribution over per-expert token counts:

$$w_i \propto i^{-\alpha}$$

```python
base = 1.0 / torch.arange(1, E + 1).float() ** alpha
perm = torch.randperm(E)                      # hot experts differ per layer
perm = (perm + phase_shift) % E               # rotate the hot set = domain shift
w[l] = base[perm] * (0.6 + 0.8 * torch.rand(E))   # measurement noise
```

What `alpha` means in practice, for 256 experts:

| alpha | top-8 share | top-32 share | hottest / coldest |
|---|---|---|---|
| 0.5 | 14.3% | 32.5% | 16x |
| 0.8 | 30.2% | 52.2% | 84x |
| **1.1** | **51.9%** | **72.8%** | **446x** |
| 1.5 | 77.5% | 90.9% | 4096x |
| 2.0 | 93.1% | 98.4% | 65536x |

Three things are deliberate:

- **`randperm` per layer** — hot expert ids differ across the 40 layers, otherwise every layer
  would produce an identical plan and per-layer planning would go untested.
- **`0.6 + 0.8 * rand` noise** — pure Zipf is too clean; this prevents the algorithm from
  accidentally depending on the exact shape of the distribution.
- **`phase_shift`** — rotating the hot set simulates a domain shift (users move from code to
  math, a different set of experts lights up). The whole point of the comparison is how much data
  must move to follow that shift.

> `alpha = 1.1` is a guess. There is no real Qwen3.6 router statistic behind it, which is exactly
> why the sweep below exists, and why Stage 5 replaces this generator with traces captured from the
> real model.

## Main comparison

40 layers x 256 experts, `alpha = 1.1`, hot set rotated by 64 experts between the two snapshots.

| policy | Max/Mean | MaxLoad | plan ms | slots moved | base moved | GB copied |
|---|---|---|---|---|---|---|
| Static EP (no balancing) | 2.369 | 14584 | 0.0 | 0 | 0 | 0.00 |
| Official EPLB (plan @ t0) | 1.000 | 6037 | 450.3 | 0 | 0 | 0.00 |
| Official EPLB (reshuffle t0 -> t1) | **1.000** | 6136 | 449.7 | 11520 | 0 | 70.78 |
| LiveEPLB (plan @ t0) | 1.090 | 6570 | 188.3 | 0 | 0 | 0.00 |
| LiveEPLB (incremental t0 -> t1) | 1.108 | 6796 | **188.9** | **984** | **0** | **6.05** |

`slots moved` counts slots that need an actual weight copy — a slot changing from one expert to
another, or from empty to occupied. A slot that becomes empty costs nothing to "move".

Reading the table:

- Official EPLB reaches the optimum, 1.000. It is free to place any expert anywhere.
- Following the traffic shift costs it **70.78 GB** — all 11520 physical slots change.
- We pay 10.8% on Max/Mean (1.108 vs 1.000) and move **6.05 GB**, an 11.7x reduction.
- `base moved = 0` confirms the Pinned-Base invariant holds: no base slot ever changes.

### Why the official policy reshuffles everything

Layer 0 / GPU 0, before and after the hot set rotates:

```
t0: [138,  9, 86, 105, 253, 214, 251, 80, 67, 64, ...]
t1: [ 68, 68, 31, 160, 163, 219, 153, 118, 58, 77, ...]
```

All 36 slots differ. The cause is not a weak algorithm — it is that the objective function contains
**only the balance of the current snapshot**, with no term for distance from the previous layout.
Each call is solved as a fresh cold-start problem, so the solution is discontinuous: a small change
in input produces a completely different output.

A common misreading: `replicate_experts` initialises `phy2log` with `torch.arange`, which looks
like "the first N experts stay put". That is only an intermediate value — step 3's
`balanced_packing` sorts by load and re-packs every replica, so the final layout is unrelated to
the initialisation.

## Ablation A: does the advantage survive a different skew?

| alpha | static | official | ours | official GB | our GB | reduction | our cap |
|---|---|---|---|---|---|---|---|
| 0.5 | 1.242 | 1.000 | 1.047 | 70.64 | 5.21 | 13.6x | 7.86 |
| 0.8 | 1.634 | 1.000 | 1.073 | 70.77 | 5.66 | 12.5x | 7.86 |
| 1.1 | 2.369 | 1.000 | 1.108 | 70.78 | 6.05 | 11.7x | 7.86 |
| 1.5 | 3.673 | 1.000 | 1.073 | 70.78 | 6.54 | 10.8x | 7.86 |
| 2.0 | 5.123 | 1.000 | 1.112 | 70.75 | 6.77 | 10.5x | 7.86 |

Three observations:

1. **The reduction is stable at 10-14x** across the whole range. The advantage is not an artifact
   of picking `alpha = 1.1`.
2. **Our Max/Mean stays in 1.05-1.11** and does not degrade as skew worsens. At `alpha = 2.0`,
   where the 8 hottest experts absorb 93% of tokens and Static EP is at 5.123, we still reach
   1.112.
3. **The bound is real.** Our migration volume never exceeds 7.86 GB (1280 slots = 40 layers x 8
   ranks x 4 redundant slots) for any alpha, because base slots are not in the search space at all.
   The official policy sits at ~70.7 GB throughout — about **9x our hard cap** — and that number
   has no upper bound guarantee; it is simply "however much a full reshuffle costs".

This distinction matters more than the averages. For an online service, a worst case you can
compute is worth more than a better average you cannot bound.

## Ablation B: constraint or algorithm?

Our Max/Mean is worse than the official 1.000. Is that because our greedy is weaker, or because the
Pinned-Base constraint costs balance? Run the official *rule* (hottest per-replica expert, placed
on the least-loaded rank) inside our layout, with the same budget of 32 redundant slots:

| policy | Max/Mean | budget | plan ms |
|---|---|---|---|
| Static EP | 2.369 | no replicas | 0.0 |
| Official EPLB (unconstrained) | **1.000** | all 288 slots | 445.9 |
| Official rule under Pinned-Base | 1.166 | 32 redundant only | 3645.6 |
| Ours (peak-driven) under Pinned-Base | **1.108** | 32 redundant only | **179.9** |

- **Row 2 vs row 3** is the price of the constraint: 1.000 -> 1.166. This has nothing to do with
  our algorithm.
- **Row 3 vs row 4** is the algorithm comparison at equal budget: 1.166 vs **1.108**. Our
  peak-driven greedy is *better* than the official rule once both are restricted.

So the gap to 1.000 comes entirely from the design decision to freeze base slots, not from a weaker
planner. The 20x planning speedup (180 ms vs 3646 ms) is a side effect of expressing the decision
as one vectorised `argmin` over all `(expert, rank)` candidates instead of recomputing rank loads in
a Python loop every round.

## The greedy rule

Each round evaluates every `(expert e, rank r)` pair and asks: *if e gains a replica and it lands
on r, what is the resulting global maximum rank load?*

Adding a replica of `e` (currently `c` replicas) on rank `r`:

```
rank r gains              w[e] / (c+1)
every current holder loses w[e] / (c * (c+1))
```

The second term is easy to miss: replication does not only burden the recipient, it simultaneously
relieves every rank that already holds that expert.

```python
add  = w / (cnt + 1.0)                                    # [E]
rel  = w / (cnt * (cnt + 1.0))                            # [E]
after = load[None, :] - rel[:, None] * holds.T            # [E, R]
peak = np.maximum(after.max(axis=1)[:, None], after + add[:, None])   # [E, R]
peak[holds.T] = np.inf        # a rank never holds two replicas of one expert
peak[:, free == 0] = np.inf   # that rank has no redundant slot left
e, r = divmod(int(np.argmin(peak)), num_ranks)
if not np.isfinite(peak[e, r]) or peak[e, r] >= cur_max - 1e-9:
    break                     # no candidate lowers the peak: stop, leave slots empty
```

`after` is **not** a 2-D load table. Each row is an independent hypothetical: `after[e]` holds the
8 rank loads in the world where only expert `e` was replicated. Only one row survives each round;
the rest are discarded. `[E, R]` is the search space, `[R]` is the state.

The stopping rule is the important part. **A replica is placed only if it lowers the global peak.**
Under uniform load no candidate qualifies, so zero replicas are emitted and the redundant slots stay
empty — which is correct, since replication provably cannot help a balanced distribution. The
original specification (`argmax(w_e/c_e)` + least-loaded rank) instead fills all 32 slots and ends
up *worse* than doing nothing: measured `[20, 32, 34, 34, 34, 34, 34, 34]`, Max/Mean 1.062 against a
baseline of 1.000. See `stage1_debug_notes.md`.

## Correctness checks

`tests/test_planner.py`, 14 tests, organised by invariant rather than by function.

**Structural** — the layout must be legal:

| test | asserts |
|---|---|
| `test_base_slots_frozen` | all 40 layers x 8 ranks keep `arange` in their first 32 slots |
| `test_no_duplicate_on_rank` | no rank holds two replicas of the same expert |
| `test_inverse_consistency` | `phy2log` and `log2phy` invert each other, `logcnt` matches |
| `test_log2phy_helper_matches` | the standalone helper agrees with the planner |

**Semantic** — the result must be useful:

| test | asserts |
|---|---|
| `test_improves_imbalance` | Max/Mean drops by >=25% vs baseline **and** total load is conserved |
| `test_uniform_weight_no_replica` | uniform load emits zero copies and leaves slots at -1 |

The conservation check guards against a specific failure: if the `w/c` split inside `rank_load` is
wrong, balance can appear to improve while tokens quietly appear or vanish.

**Boundary:**

| test | scenario |
|---|---|
| `test_boundary_max_replicas` | one expert at 1e6, rest at 1.0 — spreads to all 8 ranks, peak drops below 1/7 |
| `test_single_hot_expert_rank_bound` | replica count cannot exceed the rank count |
| `test_negative_load_raises` | negative weights rejected |
| `test_shape_mismatch_raises` | `num_experts != ranks * experts_per_rank` rejected |

**Incremental** — Stage 3 and 4 depend on these:

| test | asserts |
|---|---|
| `test_copy_tasks_are_valid` | source is `(e//32, e%32)`, source != destination rank, `dst_slot` in `[32,36)`, target slot really holds `e`, no duplicate targets |
| `test_diff_mode_no_change` | identical input against itself yields zero tasks |
| `test_diff_mode_after_drift` | after a shift, task count is in `(0, 32*L]` |

Every run also verifies, end to end, that every logical expert remains reachable and that per-layer
token counts are conserved:

```
correctness: every expert reachable, per-layer token count conserved [OK]
```

## Known limitations

- **`alpha` is unvalidated.** Everything above is a relative comparison under a synthetic
  distribution. Stage 5 replaces it with real router traces.
- **Planning cost is `O(L * R * red * E * R)`** — 180 ms for 40 layers on one CPU core. That is
  fine because planning happens off the critical path (SwiftBalancer moves the equivalent work to a
  separate worker process for the same reason), but it is not free.
- **Balance ceiling.** An expert can have at most 8 replicas, one per rank, so Max/Mean can never
  reach 1.000 the way the unconstrained policy does.
- **The load model assumes tokens split evenly across replicas.** That is what `cursor % logcnt`
  round-robin in Stage 2 will actually do, but any other dispatch policy would invalidate the
  numbers here.
