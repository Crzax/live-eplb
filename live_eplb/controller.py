"""Async rebalance state machine: IDLE -> PLANNING -> COPYING -> commit.

Shapes, used throughout:
  L  num_layers
  E  num_experts = num_ranks * experts_per_rank
  S  global slots = num_ranks * (experts_per_rank + redundant_per_rank)
  C  max replicas of one expert addressable by a shard

The planner works on all layers at once and in global slot ids; a MoEStack's
tables are the same shape but hold layer-local ones:

  planner   phy2log [L, S], log2phy [L, E, C] with ids in [0, R*spr)
  stack     log2phy [L, E, C], logcnt [L, E]  with ids in [0, spr)

Copy tasks are the third space: they address the whole pool, so their slot ids
are layer * spr + layer-local. _retract is the one place all three meet.
"""
import threading
from enum import Enum
from typing import List, Optional, Set, Tuple

import torch
import torch.distributed as dist

from .p2p import P2PSwapper
from .planner import plan_incremental_redundant, rank_load, static_phy2log


def rank_local_tables(log2phy: torch.Tensor, rank: int, slots_per_rank: int,
                      out_l: torch.Tensor, out_c: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Global slot ids -> layer-local ones, written in place.

    log2phy   [L, E, C], ids in [0, num_ranks * slots_per_rank)
    out_l     [L, E, C'] written with this rank's slots, left-packed, -1 padded
    out_c     [L, E] written with how many of them each expert got
    -> (out_l, out_c)
    """
    src = log2phy.cpu()
    assert src.shape[:2] == out_l.shape[:2], (src.shape, out_l.shape)
    keep = (src >= 0) & (src.div(slots_per_rank, rounding_mode="floor") == rank)
    local = torch.where(keep, src % slots_per_rank, torch.full_like(src, -1))
    packed = local.gather(-1, torch.argsort((~keep).int(), dim=-1, stable=True))
    width = min(packed.shape[-1], out_l.shape[-1])
    out_l.fill_(-1)
    out_l[..., :width] = packed[..., :width].int()
    out_c.copy_(keep.sum(-1).clamp(max=width).int())
    return out_l, out_c


class State(Enum):
    IDLE = 0
    PLANNING = 1
    COPYING = 2


class LiveEPLBController:
    """Per-rank driver: decides when to rebalance, stages in the background,
    commits the routing table in place.

    The planner is deterministic, so every rank derives an identical global
    placement from the same reduced statistics and keeps only its own share of
    it; the placement itself is never broadcast. The statistics are the one
    thing reduced, on a fixed step cadence and off the decode stream.
    """

    def __init__(self, swapper: P2PSwapper, shard, num_layers: int = 1,
                 num_ranks: int = 8, experts_per_rank: int = 32,
                 redundant_per_rank: int = 4, threshold: float = 1.35,
                 ema_decay: float = 0.9, warmup_steps: int = 20,
                 layers_per_step: int = 4, cooldown_steps: int = 20,
                 check_every: int = 8, snapshot_buffers: int = 4,
                 group: Optional[dist.ProcessGroup] = None):
        """swapper and shard belong to this rank; shard is a MoEStack, or
        anything exposing log2phy [L, E, C] and logcnt [L, E] over a pool the
        swapper owns.

        threshold        Max/Mean rank load of the *worst* layer above which a
                         plan is started; a layer-averaged test would let a few
                         badly skewed layers hide behind the balanced ones
        ema_decay        ema = decay * ema + (1 - decay) * counts
        warmup_steps     steps to fill the EMA before the machine arms
        cooldown_steps   minimum gap between two commits
        check_every      statistics reduction cadence, in steps
        snapshot_buffers host buffers cycled between the caller and the
                         reduction thread; the caller blocks once that many
                         reductions are outstanding
        layers_per_step  pulls issued per step, as layers * redundant_per_rank
        group            defaults to a fresh gloo group; a collective kernel on
                         the GPU would spin on SMs the decode step needs
        """
        self.sw, self.shard = swapper, shard
        self.num_layers, self.num_ranks = num_layers, num_ranks
        self.epr, self.red = experts_per_rank, redundant_per_rank
        self.threshold, self.decay = threshold, ema_decay
        self.warmup, self.layers_per_step = warmup_steps, layers_per_step
        self.cooldown = cooldown_steps
        self.check_every = check_every
        self.group = group or (dist.new_group(backend="gloo")
                               if dist.is_initialized() else None)

        self.ema = torch.zeros(num_layers, num_ranks * experts_per_rank)
        # [L, E] each, and deliberately not the same tensor: record() keeps the
        # EMA where counts live while stats is read by CPU code, so aliasing
        # them makes the gate fire off unreduced statistics whenever the two
        # happen to share a device
        self.stats = self.ema.clone()
        # [L, E] each, handed to the reduction thread in turn. A plain .cpu()
        # here would block the caller until the decode stream drained.
        self.snap_ring = [torch.empty_like(self.ema).pin_memory()
                          for _ in range(snapshot_buffers)]
        self.snap_i = 0
        self.snap_free = threading.Semaphore(snapshot_buffers)
        self.stats_q: List[Tuple[torch.Tensor, torch.cuda.Event]] = []
        self.stats_tick = threading.Semaphore(0)
        if self.group is not None:
            threading.Thread(target=self._stats_loop, daemon=True).start()
        self.phy2log = static_phy2log(num_layers, num_ranks, experts_per_rank,
                                      redundant_per_rank)          # [L, S]
        self.planned_phy2log = self.phy2log      # [L, S] adopted at commit
        # [L, 1, 1] layer-local slot id + this = pool slot id
        self.slot_offsets = (torch.arange(num_layers, dtype=torch.int32)
                             * (experts_per_rank + redundant_per_rank)).view(-1, 1, 1)
        self.state = State.IDLE
        self.step = 0
        self.last_commit = -cooldown_steps
        # (dst_slot, src_rank, src_slot), all three pool ids
        self.pending: List[Tuple[int, int, int]] = []
        self.event: Optional[torch.cuda.Event] = None
        self.plan_result: Optional[tuple] = None
        self.plan_job: Optional[tuple] = None
        self.plan_tick = threading.Semaphore(0)
        self.plan_done = threading.Event()
        threading.Thread(target=self._plan_loop, daemon=True).start()
        # pinned up front: the first pin_memory() costs tens of ms in the driver
        self.host_l = torch.full(tuple(shard.log2phy.shape), -1,
                                 dtype=torch.int32).pin_memory()   # [L, E, C] planned
        self.host_c = torch.zeros(tuple(shard.logcnt.shape),
                                  dtype=torch.int32).pin_memory()  # [L, E]
        self.hold_l = self.host_l.clone().pin_memory()  # [L, E, C] held while copying
        self.hold_c = self.host_c.clone().pin_memory()  # [L, E]
        self.live_l = shard.log2phy.cpu()               # [L, E, C] mirror, in force
        self.live_c = shard.logcnt.cpu()                # [L, E]
        # warm up first-use costs (the commit kernel's module load, the EMA's
        # pageable H2D; S4-15): each would otherwise wait out a full replay
        self.sw.commit(shard.log2phy, shard.log2phy.clone(),
                       shard.logcnt, shard.logcnt.clone())
        self.record(torch.zeros(self.ema.shape, device=shard.log2phy.device))
        self.layer_imbalance()
        self.commits = 0
        self.plans = 0

    def record(self, counts: torch.Tensor) -> None:
        """Fold counts [L, E], tokens per expert this step, into the EMA. Stays
        on counts' device; a .cpu() here would sync on the decode step."""
        if self.ema.device != counts.device:
            self.ema = self.ema.to(counts.device)
        self.ema.mul_(self.decay).add_(counts.float(), alpha=1.0 - self.decay)

    def layer_imbalance(self) -> torch.Tensor:
        """-> [L] of max rank load / mean rank load, under the placement in
        force and the latest reduced statistics.

        Each layer's redundant slots are its own, so these are L independent
        problems and the trigger reads the worst entry, not the average.
        """
        load = rank_load(self.phy2log, self.stats, self.num_ranks)   # [L, R]
        return load.max(-1).values / load.mean(-1).clamp(min=1e-9)

    def imbalance(self) -> float:
        """-> mean of layer_imbalance(), which is how much longer a decode step
        takes than a perfectly balanced one.

        The two coincide because every layer routes the same token count, so
        its mean rank load is the same, and a step costs the sum over layers of
        the slowest rank. This is a cost measure, not the trigger.
        """
        return self.layer_imbalance().mean().item()

    def _stats_loop(self) -> None:
        """Body of a resident thread; never call from the main thread.

        Reduces queued snapshots in issue order, so every rank runs the same
        sequence of collectives even when their state machines drift apart.

        Rebinds self.stats rather than writing into it: a reader in imbalance()
        then sees either the whole old tensor or the whole new one.

        Only started when a group exists -- the whole body is a collective, and
        without a group _reduce_stats assigns self.stats itself and never
        releases the semaphore, so the thread would block forever.
        """
        while True:
            self.stats_tick.acquire()
            buf, ev = self.stats_q.pop(0)
            ev.synchronize()        # the D2H is async; wait off the main thread
            snap = buf.clone()      # own it: the ring slot will be reused
            self.snap_free.release()
            dist.all_reduce(snap, group=self.group)
            self.stats = snap

    def _reduce_stats(self) -> None:
        """Hand a host snapshot [L, E] of this rank's EMA to the reduction
        thread, which sums it across ranks into self.stats.

        The copy is issued async into a pinned ring slot and only an event is
        queued with it, so nothing here waits on the decode stream.
        """
        if self.group is None:
            self.stats = self.ema.cpu()
            return
        # back-pressure rather than skipping: a skip is a local decision, and
        # one rank skipping a collective the others run is a deadlock. Blocking
        # keeps the sequence identical on every rank; with the default ring and
        # cadence it never actually waits.
        self.snap_free.acquire()
        buf = self.snap_ring[self.snap_i]
        self.snap_i = (self.snap_i + 1) % len(self.snap_ring)
        buf.copy_(self.ema, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.stats_q.append((buf, ev))
        self.stats_tick.release()

    def _plan_loop(self) -> None:
        """Body of a resident thread; never call from the main thread. Started
        unconditionally: planning is local CPU work, with or without a group.

        Turns each queued job into plan_result,
        (phy2log [L, S], log2phy [L, E, C], logcnt [L, E], pulls, retract_needed),
        the two tables already layer-local and the pulls already this rank's."""
        while True:
            self.plan_tick.acquire()
            stats, prev, hot = self.plan_job
            phy2log, log2phy, _, tasks = plan_incremental_redundant(
                stats, self.num_ranks, self.epr, self.red, prev_phy2log=prev,
                layers=hot)
            # the planner's logcnt counts replicas cluster-wide; a stack needs
            # how many of its own slots serve each expert, which this recomputes
            host_l, host_c = rank_local_tables(log2phy, self.sw.rank,
                                               self.epr + self.red,
                                               self.host_l, self.host_c)
            mine = self.sw.local_tasks(tasks, self.epr + self.red)
            retract = self._retract({dst for dst, _, _ in mine})
            self.plan_result = (phy2log, host_l, host_c, mine, retract)
            self.plan_done.set()

    def _start_planning(self, hot: List[int]) -> None:
        """Queue (statistics [L, E], placement in force [L, S], layers to
        re-plan) for the planner thread. The first two are cloned: it runs for
        milliseconds while they keep changing."""
        self.plan_job = (self.stats.clone(), self.phy2log.clone(), hot)
        self.plan_result = None
        self.plan_done.clear()
        self.plan_tick.release()

    def _retract(self, dst: Set[int]) -> bool:
        """Build into hold_l [L, E, C] / hold_c [L, E] the routing that survives
        unlinking the pool slots `dst`, i.e. what stays in force while they are
        overwritten; -> whether any of them was in use.

        A redundant slot may still serve a replica an earlier plan put there,
        and overwriting it while routable hands tokens half-written weights.

        The table is layer-local and `dst` is not, so the comparison is done in
        pool ids. Padding (-1) biased by an offset can collide with a real slot
        of the layer below, which the `l >= 0` term rules out.
        """
        l = self.live_l
        coded = l + self.slot_offsets
        keep = (l >= 0) & ~torch.isin(coded, torch.tensor(sorted(dst), dtype=l.dtype))
        order = torch.argsort((~keep).int(), dim=-1, stable=True)
        self.hold_l.copy_(torch.where(keep, l, torch.full_like(l, -1)).gather(-1, order))
        self.hold_c.copy_(keep.sum(-1).int())
        return bool((keep != (l >= 0)).any())

    def _absorb_plan(self) -> bool:
        """Install the finished plan: tables to the device, pulls into pending.
        -> False if it matches the placement already in force.

        phy2log is only adopted once the plan is committed: until then the old
        one is what serves, and reporting the new one would credit the balance
        to steps that are still running on the old layout."""
        phy2log, host_l, host_c, mine, retract = self.plan_result
        self.plan_result = None
        self.plans += 1
        if not mine and torch.equal(host_l, self.live_l):
            self.phy2log = phy2log
            return False
        self.planned_phy2log = phy2log
        dev = self.shard.log2phy.device
        if retract:
            # left on the current stream on purpose: it orders against both the
            # replays already enqueued there and the pull_batch that waits on
            # it, so no replay reaching a doomed slot can straddle the copy
            self.sw.commit(self.shard.log2phy, self.hold_l.to(dev, non_blocking=True),
                           self.shard.logcnt, self.hold_c.to(dev, non_blocking=True))
        self.new_log2phy = host_l.to(dev, non_blocking=True)   # [L, E, C]
        self.new_logcnt = host_c.to(dev, non_blocking=True)    # [L, E]
        self.pending = mine
        return True

    def _issue(self) -> None:
        """Pull the next batch and record the event to poll. The cap is
        layers_per_step layers' worth at red slots per layer, spreading a
        plan's bandwidth over several steps instead of one burst."""
        batch = self.pending[:self.layers_per_step * self.red]
        self.pending = self.pending[len(batch):]
        self.event = self.sw.pull_batch(batch) if batch else None

    def on_step(self, counts: torch.Tensor) -> str:
        """Advance one decode step with counts [L, E]; -> the action taken, one
        of warmup / cooldown / idle / planning / copying / commit.

        Never blocks: planning runs on a thread and staging is polled, so the
        old routing table keeps serving until the new one is complete.

        Must be called on the same stream as replay(), unwrapped: table edits
        are ordered against decode by stream order alone.
        """
        self.record(counts)
        self.step += 1
        if self.step < self.warmup:
            return "warmup"
        # unconditional: a reduction gated by the state machine would deadlock
        # once two ranks disagree on whether to enter it
        if self.step % self.check_every == 0:
            self._reduce_stats()

        if self.state is State.IDLE:
            if self.step - self.last_commit < self.cooldown:
                return "cooldown"
            hot = (self.layer_imbalance() > self.threshold).nonzero().flatten().tolist()
            if not hot:
                return "idle"
            self._start_planning(hot)
            self.state = State.PLANNING
            return "planning"

        if self.state is State.PLANNING:
            if not self.plan_done.is_set():
                return "planning"
            if not self._absorb_plan():
                self.state = State.IDLE
                self.last_commit = self.step
                return "idle"
            self._issue()
            self.state = State.COPYING
            return "copying"

        if self.event is not None and not self.event.query():
            return "copying"
        if self.pending:
            self._issue()
            return "copying"

        self.sw.commit(self.shard.log2phy, self.new_log2phy,
                       self.shard.logcnt, self.new_logcnt)
        self.phy2log = self.planned_phy2log
        self.live_l.copy_(self.host_l)
        self.live_c.copy_(self.host_c)
        self.commits += 1
        self.last_commit = self.step
        self.state = State.IDLE
        self.event = None
        return "commit"
