"""Stage 1: Pinned-Base incremental redundant-slot planner vs official EPLB."""
from typing import List, Optional, Tuple

import numpy as np
import torch

CopyTask = Tuple[int, int, int, int, int, int]  # (layer, src_rank, src_slot, dst_rank, dst_slot, expert)


def static_phy2log(num_layers: int, num_ranks: int = 8, experts_per_rank: int = 32,
                   redundant_per_rank: int = 4) -> torch.Tensor:
    """Baseline layout -> [L, R*(epr+red)]: rank r holds [r*epr, (r+1)*epr), rest -1."""
    spr = experts_per_rank + redundant_per_rank
    phy2log = torch.full((num_layers, num_ranks * spr), -1, dtype=torch.int64)
    for r in range(num_ranks):
        phy2log[:, r * spr:r * spr + experts_per_rank] = torch.arange(
            r * experts_per_rank, (r + 1) * experts_per_rank)
    return phy2log


def log2phy_from_phy2log(phy2log: torch.Tensor, num_experts: int
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """phy2log [L, S] -> (phy2log, log2phy [L, E, C] int32, logcnt [L, E])."""
    num_layers, num_slots = phy2log.shape
    valid = phy2log >= 0
    logcnt = torch.zeros(num_layers, num_experts, dtype=torch.int64)
    logcnt.scatter_add_(-1, phy2log.clamp(min=0), valid.to(torch.int64))
    maxcnt = max(int(logcnt.max()), 1)
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
    """phy2log [L, S], weight [L, E] -> load [L, R]; replica of e carries w[e]/logcnt[e]."""
    num_layers, num_slots = phy2log.shape
    slots = num_slots // num_ranks
    w = weight.float()
    valid = phy2log >= 0
    logcnt = torch.zeros(num_layers, w.shape[1])
    logcnt.scatter_add_(-1, phy2log.clamp(min=0), valid.float())
    per_replica = (w / logcnt.clamp(min=1)).gather(-1, phy2log.clamp(min=0)) * valid
    return per_replica.view(num_layers, num_ranks, slots).sum(-1)


def _plan_layer(w: np.ndarray, num_ranks: int, epr: int, red: int) -> List[int]:
    """Greedy plan for ONE layer; returns slot->expert (=-1 for unused redundant slots).

    Adding a replica of e to rank r costs w[e]/(c+1) on r and relieves
    w[e]/(c*(c+1)) on each rank already holding e. A move is accepted only if it
    lowers the global max rank load, so uniformly distributed traffic correctly
    gets no replicas (replication cannot help there) while skewed traffic does.

    w is [E]; returns [S] with slot -> expert.
    """
    spr = epr + red
    num_experts = num_ranks * epr
    cnt = np.ones(num_experts)
    holds = np.zeros((num_ranks, num_experts), dtype=bool)
    for r in range(num_ranks):
        holds[r, r * epr:(r + 1) * epr] = True
    load = holds.astype(np.float64) @ w               # [R]
    free = np.full(num_ranks, red, dtype=np.int64)

    for _ in range(num_ranks * red):
        cur_max = load.max()
        add = w / (cnt + 1.0)
        rel = w / (cnt * (cnt + 1.0))
        # after[e] is an independent hypothetical: rank loads if only e were replicated
        after = load[None, :] - rel[:, None] * holds.T           # [E, R]
        # peak[e, r] = global max rank load if e's new replica lands on r
        peak = np.maximum(after.max(axis=1)[:, None], after + add[:, None])
        peak[holds.T] = np.inf
        peak[:, free == 0] = np.inf
        e, r = divmod(int(np.argmin(peak)), num_ranks)
        if not np.isfinite(peak[e, r]) or peak[e, r] >= cur_max - 1e-9:
            break
        load = after[e].copy()
        load[r] += add[e]
        holds[r, e] = True
        cnt[e] += 1
        free[r] -= 1

    row = [-1] * (num_ranks * spr)
    for r in range(num_ranks):
        row[r * spr:r * spr + epr] = list(range(r * epr, (r + 1) * epr))
        extra = [e for e in np.flatnonzero(holds[r]) if not r * epr <= e < (r + 1) * epr]
        assert len(extra) <= red, f"rank {r} overflowed: {len(extra)} > {red}"
        for i, e in enumerate(extra):
            row[r * spr + epr + i] = int(e)
    return row


def _diff_tasks(phy2log: torch.Tensor, prev: Optional[torch.Tensor], num_ranks: int,
                epr: int, red: int) -> List[CopyTask]:
    """Copy task per redundant slot differing from `prev`. Base slots are never
    scanned; the source is a constant, e's base replica at rank e//epr slot e%epr.
    dst_slot is rank-local, in [epr, epr+red)."""
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
    """Pinned-Base planner. Base slots are frozen (rank r keeps [r*epr, (r+1)*epr)),
    so only the trailing redundant slots are planned and an expert gets at most
    num_ranks replicas. Layers are independent.

    weight [L, E] -> phy2log [L, S], log2phy [L, E, C], logcnt [L, E], copy tasks.
    """
    weight = weight.float().cpu()
    num_layers, num_experts = weight.shape
    assert num_experts == num_ranks * experts_per_rank, (
        f"num_experts {num_experts} != num_ranks*experts_per_rank {num_ranks * experts_per_rank}")
    assert (weight >= 0).all(), "expert load must be non-negative"

    phy2log = static_phy2log(num_layers, num_ranks, experts_per_rank, redundant_per_rank)
    for l in range(num_layers):
        row = _plan_layer(weight[l].numpy().astype(np.float64), num_ranks,
                          experts_per_rank, redundant_per_rank)
        phy2log[l] = torch.tensor(row, dtype=torch.int64)

    tasks = _diff_tasks(phy2log, prev_phy2log, num_ranks, experts_per_rank, redundant_per_rank)
    phy2log, log2phy, logcnt = log2phy_from_phy2log(phy2log, num_experts)
    return phy2log, log2phy, logcnt, tasks
