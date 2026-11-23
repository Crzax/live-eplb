"""Stage 1: Pinned-Base incremental redundant-slot planner vs official EPLB."""
from typing import List, Optional, Tuple

import numpy as np
import torch

CopyTask = Tuple[int, int, int, int, int, int]  # (layer, src_rank, src_slot, dst_rank, dst_slot, expert)


def static_phy2log(num_layers: int, num_ranks: int = 8, experts_per_rank: int = 32,
                   redundant_per_rank: int = 4) -> torch.Tensor:
    """Baseline layout: rank r holds experts [r*epr, (r+1)*epr); redundant slots are -1.

    Shapes:
        return phy2log: [num_layers, num_ranks*(epr+red)] int64, slot -> expert (-1 = empty)
    """
    spr = experts_per_rank + redundant_per_rank
    phy2log = torch.full((num_layers, num_ranks * spr), -1, dtype=torch.int64)
    for r in range(num_ranks):
        phy2log[:, r * spr:r * spr + experts_per_rank] = torch.arange(
            r * experts_per_rank, (r + 1) * experts_per_rank)
    return phy2log


def log2phy_from_phy2log(phy2log: torch.Tensor, num_experts: int
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert the slot->expert table into the expert->slot table used at inference.

    Shapes (L layers, S slots, E experts, C = max replicas of any expert):
        phy2log: [L, S]       int64, slot -> expert (-1 = empty)
        return:  [L, S]       int64, phy2log unchanged (pass-through for convenience)
                 [L, E, C]    int32, log2phy, replica k of expert e -> slot (-1 = unused)
                 [L, E]       int64, logcnt, number of replicas of each expert
    """
    num_layers, num_slots = phy2log.shape
    valid = phy2log >= 0                                      # [L, S]
    logcnt = torch.zeros(num_layers, num_experts, dtype=torch.int64)    # [L, E]
    logcnt.scatter_add_(-1, phy2log.clamp(min=0), valid.to(torch.int64))
    maxcnt = max(int(logcnt.max()), 1)                        # C
    log2phy = torch.full((num_layers, num_experts, maxcnt), -1, dtype=torch.int32)
    for l in range(num_layers):
        cursor = [0] * num_experts
        for s in range(num_slots):
            e = int(phy2log[l, s])
            if e < 0:
                continue
            log2phy[l, e, cursor[e]] = s
            cursor[e] += 1
    return phy2log, log2phy, logcnt


def rank_load(phy2log: torch.Tensor, weight: torch.Tensor, num_ranks: int = 8) -> torch.Tensor:
    """Per-rank load; each replica of expert e carries weight[e]/logcnt[e].

    Shapes (L layers, S slots, R ranks, E experts):
        phy2log: [L, S]   int64, slot -> expert (-1 = empty)
        weight:  [L, E]   float, token count per logical expert
        return:  [L, R]   float, load carried by each rank
    """
    num_layers, num_slots = phy2log.shape
    slots = num_slots // num_ranks
    w = weight.float()                                        # [L, E]
    num_experts = w.shape[1]
    valid = phy2log >= 0                                      # [L, S]
    logcnt = torch.zeros(num_layers, num_experts)             # [L, E]
    logcnt.scatter_add_(-1, phy2log.clamp(min=0), valid.float())
    # per_replica[l, s] = load carried by slot s
    per_replica = (w / logcnt.clamp(min=1)).gather(-1, phy2log.clamp(min=0)) * valid   # [L, S]
    return per_replica.view(num_layers, num_ranks, slots).sum(-1)


def _plan_layer(w: np.ndarray, num_ranks: int, epr: int, red: int) -> List[int]:
    """Greedy plan for ONE layer; returns slot->expert (=-1 for unused redundant slots).

    Adding a replica of e to rank r costs w[e]/(c+1) on r and relieves
    w[e]/(c*(c+1)) on each rank already holding e. A move is accepted only if it
    lowers the global max rank load, so uniformly distributed traffic correctly
    gets no replicas (replication cannot help there) while skewed traffic does.

    Shapes (E = num_ranks*epr experts, R = num_ranks, S = R*(epr+red) slots):
        w:      [E]      float64, token count of each logical expert in this layer
        return: [S]       list[int], slot -> logical expert id (-1 = empty slot)
    """
    spr = epr + red                                   # slots per rank
    num_experts = num_ranks * epr                     # E
    cnt = np.ones(num_experts)                        # [E] replicas per expert
    holds = np.zeros((num_ranks, num_experts), dtype=bool)   # [R, E] rank holds expert?
    for r in range(num_ranks):
        holds[r, r * epr:(r + 1) * epr] = True
    load = holds.astype(np.float64) @ w               # [R] current load per rank

    for _ in range(num_ranks * red):
        cur_max = load.max()                                     # scalar
        add = w / (cnt + 1.0)                                    # [E] added to target rank
        rel = w / (cnt * (cnt + 1.0))                            # [E] relieved per holder
        # after[e] = [R] rank loads once e gains a replica, before adding it anywhere
        after = load[None, :] - rel[:, None] * holds.T           # [E, R]
        after_max = after.max(axis=1)                            # [E]
        # peak[e, r] = global max rank load if e's new replica lands on rank r
        peak = np.maximum(after_max[:, None], after + add[:, None])   # [E, R]
        peak[holds.T] = np.inf                                   # rank already has e
        flat = int(np.argmin(peak))                              # scalar, index into [E, R]
        e, r = divmod(flat, num_ranks)
        if not np.isfinite(peak[e, r]) or peak[e, r] >= cur_max - 1e-9:
            break
        load = after[e].copy()                                   # [R]
        load[r] += add[e]
        holds[r, e] = True
        cnt[e] += 1

    row = [-1] * (num_ranks * spr)
    for r in range(num_ranks):
        row[r * spr:r * spr + epr] = list(range(r * epr, (r + 1) * epr))
        extra = [e for e in np.flatnonzero(holds[r]) if not r * epr <= e < (r + 1) * epr]
        for i, e in enumerate(extra[:red]):
            row[r * spr + epr + i] = int(e)
    return row


def _diff_tasks(phy2log: torch.Tensor, prev: Optional[torch.Tensor], num_ranks: int,
                epr: int, red: int) -> List[CopyTask]:
    """Emit a copy task for every redundant slot whose content differs from `prev`.

    Base slots are never scanned (they cannot change). The source is a constant:
    expert e always has its base replica at rank e//epr, slot e%epr.

    Shapes:
        phy2log: [L, S] int64
        prev:    [L, S] int64 or None (None = every occupied redundant slot is new)
        return:  list of (layer, src_rank, src_slot, dst_rank, dst_slot, expert);
                 dst_slot is rank-local (in [epr, epr+red)), not a global slot id
    """
    spr = epr + red
    tasks: List[CopyTask] = []
    for l in range(phy2log.shape[0]):
        for r in range(num_ranks):
            for i in range(red):
                e = int(phy2log[l, r * spr + epr + i])
                if e < 0:
                    continue
                if prev is not None and int(prev[l, r * spr + epr + i]) == e:
                    continue
                tasks.append((l, e // epr, e % epr, r, epr + i, e))
    return tasks


def plan_incremental_redundant(weight: torch.Tensor, num_ranks: int = 8,
                               experts_per_rank: int = 32, redundant_per_rank: int = 4,
                               prev_phy2log: Optional[torch.Tensor] = None
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[CopyTask]]:
    """Pinned-Base incremental planner.

    Base slots are frozen: rank r keeps logical experts [r*epr, (r+1)*epr), so the
    base mapping never changes and never needs a copy. Only the trailing
    redundant_per_rank slots of each rank are used, giving at most num_ranks
    replicas per expert (one per rank). Each layer is planned independently.

    Shapes (L layers, E = num_ranks*experts_per_rank experts,
            S = num_ranks*(experts_per_rank+redundant_per_rank) slots,
            C = max replicas of any expert):
        weight:       [L, E]  float, token count per logical expert
        prev_phy2log: [L, S]  int64 or None, previous layout used to diff copy tasks
        return:       [L, S]     int64, phy2log, slot -> expert (-1 = empty)
                      [L, E, C]  int32, log2phy, replica k of expert e -> slot
                      [L, E]     int64, logcnt, replicas per expert
                      list       copy tasks (see _diff_tasks)
    """
    weight = weight.float().cpu()
    num_layers, num_experts = weight.shape
    assert num_experts == num_ranks * experts_per_rank, (
        f"num_experts {num_experts} != num_ranks*experts_per_rank {num_ranks * experts_per_rank}")
    assert (weight >= 0).all(), "expert load must be non-negative"
    spr = experts_per_rank + redundant_per_rank

    phy2log = static_phy2log(num_layers, num_ranks, experts_per_rank, redundant_per_rank)
    for l in range(num_layers):
        row = _plan_layer(weight[l].numpy().astype(np.float64), num_ranks,
                          experts_per_rank, redundant_per_rank)
        phy2log[l] = torch.tensor(row, dtype=torch.int64)

    tasks = _diff_tasks(phy2log, prev_phy2log, num_ranks, experts_per_rank, redundant_per_rank)
    phy2log, log2phy, logcnt = log2phy_from_phy2log(phy2log, num_experts)
    return phy2log, log2phy, logcnt, tasks
