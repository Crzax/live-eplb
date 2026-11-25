"""Cross-process expert migration over NVLink, built on CUDA IPC."""
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

import live_eplb_cuda as _C

# (layer, src_rank, src_slot, dst_rank, dst_slot, expert); slots are rank-local
CopyTask = Tuple[int, int, int, int, int, int]


class P2PSwapper:
    """Pulls expert slots from peer ranks' weight pools into the local pool.

    Each rank exports an IPC handle for its own pool once; every other rank maps
    it and keeps the pointer. A migration is then a plain peer copy on the copy
    engine, with no participation from the source rank.
    """

    def __init__(self, pool: torch.Tensor, rank: int, world_size: int,
                 group: Optional[dist.ProcessGroup] = None):
        assert pool.is_cuda and pool.is_contiguous()
        assert pool.storage_offset() == 0, "pool must own its storage"
        self.pool = pool
        self.rank, self.world_size = rank, world_size
        self.group = group
        self.slot_bytes = pool[0].numel() * pool.element_size()
        self.peers: List[int] = []
        self.stream = torch.cuda.Stream()

    def exchange_handles(self) -> None:
        local = _C.get_ipc_handle(self.pool)
        gathered: List[Optional[List[int]]] = [None] * self.world_size
        dist.all_gather_object(gathered, list(local), group=self.group)
        for r, blob in enumerate(gathered):
            if r == self.rank:
                continue
            if not _C.enable_peer_access(r):
                continue
            _C.open_ipc_handle(list(blob), r)
            self.peers.append(r)

    def pull(self, dst_slot: int, src_rank: int, src_slot: int, n_slots: int = 1,
             stream: Optional[torch.cuda.Stream] = None) -> None:
        """Pull n_slots consecutive slots. Merging runs matters: one 252 MB copy
        reaches 376 GB/s where 40 separate 6.29 MB copies only reach 192."""
        s = stream or self.stream
        _C.async_copy_expert_slot(self.pool, dst_slot, src_rank, src_slot,
                                  self.slot_bytes * n_slots, s.cuda_stream)

    @staticmethod
    def _coalesce(tasks: Sequence[Tuple[int, int, int]]
                  ) -> List[Tuple[int, int, int, int]]:
        """Merge (dst, src_rank, src_slot) triples into contiguous runs."""
        if not tasks:
            return []
        out: List[Tuple[int, int, int, int]] = []
        for dst, rk, src in sorted(tasks, key=lambda t: (t[1], t[2], t[0])):
            if out:
                pdst, prk, psrc, n = out[-1]
                if rk == prk and src == psrc + n and dst == pdst + n:
                    out[-1] = (pdst, prk, psrc, n + 1)
                    continue
            out.append((dst, rk, src, 1))
        return out

    def pull_batch(self, tasks: Sequence[Tuple[int, int, int]],
                   stream: Optional[torch.cuda.Stream] = None) -> torch.cuda.Event:
        """tasks = [(dst_slot, src_rank, src_slot)]; returns an event to poll.

        Waits on the current stream first, so anything already enqueued there
        -- a replay, a commit -- completes before a slot is overwritten. The
        caller must therefore issue this from the stream it replays on.
        """
        s = stream or self.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for dst, rk, src, n in self._coalesce(tasks):
                self.pull(dst, rk, src, n, s)
            ev = torch.cuda.Event()
            ev.record(s)
        return ev

    def local_tasks(self, copy_tasks: Sequence[CopyTask], slots_per_layer: int
                    ) -> List[Tuple[int, int, int]]:
        """Pulls this rank owns -> [(dst_slot, src_rank, src_slot)].

        copy_tasks       [(layer, src_rank, src_slot: L, dst_rank, dst_slot: L, expert)]
                         with both slots layer-local (rank-lane slot ids)
        slots_per_layer  spr, the stride that folds a layer-local slot into a
                         pool slot id (global). Output ids are pool-wide and
                         unique per (layer, slot) pair, so layers cannot collide

        The planner never emits src_rank == dst_rank: a rank holding e in a base
        slot is excluded as a replica target, so every task is a real peer copy.
        """
        out = []
        for l, src_rank, src_slot, dst_rank, dst_slot, _ in copy_tasks:
            if dst_rank != self.rank:
                continue
            assert src_rank != dst_rank, f"self-copy in task {(l, src_rank, dst_rank)}"
            off = l * slots_per_layer
            out.append((off + dst_slot, src_rank, off + src_slot))
        return out

    def commit(self, log2phy: torch.Tensor, new_log2phy: torch.Tensor,
               logcnt: torch.Tensor, new_logcnt: torch.Tensor,
               stream: Optional[torch.cuda.Stream] = None) -> None:
        """Overwrite the routing table in place, keeping its addresses.

        log2phy / new_log2phy  [L, E, C]
        logcnt / new_logcnt    [L, E]

        Defaults to the current stream, which must be the one replay() uses:
        ordering against in-flight replays is what makes a mid-decode table
        swap safe, and a separate stream would drop that guarantee.
        """
        s = stream or torch.cuda.current_stream()
        _C.commit_routing_table(log2phy, new_log2phy, logcnt, new_logcnt,
                                s.cuda_stream)

    def close(self) -> None:
        _C.close_ipc_handles()
        self.peers.clear()
