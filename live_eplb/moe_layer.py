"""Stage 2: an EP=8 MoE shard whose expert weights can be swapped under a live CUDA Graph.

The whole design exists to satisfy one constraint: a CUDA Graph bakes every kernel
argument, including weight pointers, into the graph at capture time. So nothing the
graph touches may ever be reallocated. Two allocations are made once and never
replaced:

    W_pool  [slots, 3, moe_inter, hidden]   expert weights, contents mutable
    log2phy [experts, max_replicas]         expert -> slot, contents mutable
    logcnt  [experts]                       replicas per expert

The captured forward reads a *slot index* out of log2phy and gathers from W_pool.
Swapping an expert is therefore two in-place writes (new bytes into a free slot, new
index into the table) and the graph never notices.
"""
from typing import Optional, Tuple

import torch

# Qwen3.6-35B-A3B routed expert: gate / up / down, each [moe_inter, hidden]
GATE, UP, DOWN = 0, 1, 2


class GraphSafeMoEShard:
    """One GPU's shard of a MoE layer, with address-pinned weights and routing table.

    Args:
        experts: number of logical experts the routing table can address
        slots: physical slots held by this shard (base + redundant)
        hidden: model hidden size
        moe_inter: expert intermediate size
        max_replicas: last dim of log2phy; an expert may have at most this many replicas
    """

    def __init__(self, experts: int = 256, slots: int = 36, hidden: int = 2048,
                 moe_inter: int = 512, max_replicas: int = 8,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.experts, self.slots = experts, slots
        self.hidden, self.moe_inter = hidden, moe_inter
        self.device, self.dtype = torch.device(device), dtype

        # --- the three address-pinned allocations ---------------------------------
        self.w_pool = torch.zeros(slots, 3, moe_inter, hidden, device=device, dtype=dtype)
        self.log2phy = torch.full((experts, max_replicas), -1, dtype=torch.int32, device=device)
        self.logcnt = torch.zeros(experts, dtype=torch.int32, device=device)
        # cursor must live on the GPU: a Python int would be constant-folded at capture
        self.cursor = torch.zeros(1, dtype=torch.int32, device=device)

        self._base_ptrs = (self.w_pool.data_ptr(), self.log2phy.data_ptr(),
                           self.logcnt.data_ptr(), self.cursor.data_ptr())

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.replays = 0
        self.captures = 0

    # ------------------------------------------------------------------ bookkeeping
    def assert_addresses_pinned(self) -> None:
        """Any reallocation of these four tensors silently invalidates the graph."""
        now = (self.w_pool.data_ptr(), self.log2phy.data_ptr(),
               self.logcnt.data_ptr(), self.cursor.data_ptr())
        assert now == self._base_ptrs, (
            f"pinned tensor was reallocated: {self._base_ptrs} -> {now}")

    def load_routing(self, log2phy: torch.Tensor, logcnt: torch.Tensor) -> None:
        """Install a routing table in place (shapes must match the pinned allocations)."""
        assert log2phy.shape == self.log2phy.shape, (log2phy.shape, self.log2phy.shape)
        assert logcnt.shape == self.logcnt.shape, (logcnt.shape, self.logcnt.shape)
        self.log2phy.copy_(log2phy)
        self.logcnt.copy_(logcnt)
        self.assert_addresses_pinned()

    def load_expert(self, slot: int, w: torch.Tensor) -> None:
        """Write one expert's weights into a slot, in place."""
        assert w.shape == (3, self.moe_inter, self.hidden), w.shape
        self.w_pool[slot].copy_(w)

    # ----------------------------------------------------------------- the forward
    def _route(self, expert: torch.Tensor) -> torch.Tensor:
        """expert [T] int32 -> slot [T] int64, all of it graph-capturable.

        Round-robin across replicas: replica k = cursor % logcnt[e]. This is what makes
        the planner's "each replica carries w/c" load model true at runtime.
        """
        e = expert.long()
        cnt = self.logcnt.index_select(0, e).long()             # [T]
        k = (self.cursor.long() % cnt.clamp(min=1))             # [T] broadcast over cursor[1]
        row = self.log2phy.index_select(0, e).long()            # [T, max_replicas]
        return row.gather(1, k.unsqueeze(1)).squeeze(1)         # [T]

    def forward(self, x: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        """x [T, hidden], expert [T] int32 -> y [T, hidden]. SwiGLU, one expert per token."""
        slot = self._route(expert)                              # [T]
        w = self.w_pool.index_select(0, slot)                   # [T, 3, moe_inter, hidden]
        # per-token matvec: [T, moe_inter, hidden] @ [T, hidden, 1]
        xv = x.unsqueeze(-1).to(w.dtype)                        # [T, hidden, 1]
        gate = torch.bmm(w[:, GATE], xv).squeeze(-1)            # [T, moe_inter]
        up = torch.bmm(w[:, UP], xv).squeeze(-1)                # [T, moe_inter]
        act = (torch.nn.functional.silu(gate) * up).unsqueeze(-1)   # [T, moe_inter, 1]
        return torch.bmm(w[:, DOWN].transpose(1, 2), act).squeeze(-1)   # [T, hidden]

    # ------------------------------------------------------------------- the graph
    def capture(self, x: torch.Tensor, expert: torch.Tensor, warmup: int = 3
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Capture forward into a CUDA Graph. Returns the static input/output buffers.

        Writing into the returned input buffer and calling replay() runs the graph on
        the new data. The buffers, like everything else here, are never reallocated.
        """
        self.static_x = x.clone()
        self.static_expert = expert.clone().to(torch.int32)

        # Warm up on a side stream first: cuBLAS/cuDNN lazy init and the caching
        # allocator's first-touch work must not land inside the capture.
        s = torch.cuda.Stream(device=self.device)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.forward(self.static_x, self.static_expert)
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_y = self.forward(self.static_x, self.static_expert)
        self.captures += 1
        self.assert_addresses_pinned()
        return self.static_x, self.static_y

    def replay(self) -> torch.Tensor:
        assert self.graph is not None, "capture() first"
        self.graph.replay()
        self.replays += 1
        return self.static_y

    # ---------------------------------------------------------------- weight swap
    def stage_expert_async(self, slot: int, w: torch.Tensor) -> torch.cuda.Event:
        """Copy an expert into `slot` on transfer_stream; returns an event to poll.

        Safe to overlap with replay() as long as `slot` is currently unreachable from
        log2phy, which is exactly what an unused redundant slot is.
        """
        assert w.shape == (3, self.moe_inter, self.hidden), w.shape
        self.transfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.transfer_stream):
            self.w_pool[slot].copy_(w, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.transfer_stream)
        return ev

    def commit_routing(self, expert: int, slot: int, replica: int = 0,
                       new_logcnt: Optional[int] = None) -> None:
        """Point expert -> slot in place. No new allocation, so the graph stays valid."""
        self.log2phy[expert, replica] = slot
        if new_logcnt is not None:
            self.logcnt[expert] = new_logcnt
        self.assert_addresses_pinned()

    def set_cursor(self, value: int) -> None:
        """Advance the round-robin cursor. Must go through copy_, not reassignment."""
        self.cursor.fill_(value)
