"""MoE shard whose expert weights can be swapped under a live CUDA Graph.

A graph bakes kernel arguments, including weight pointers, at capture time. So
W_pool / log2phy / logcnt / cursor are allocated once and only ever mutated in
place; the captured forward reads a slot index from log2phy instead of closing
over a weight address.
"""
from typing import Optional, Tuple

import torch

GATE, UP, DOWN = 0, 1, 2


class GraphSafeMoEShard:
    def __init__(self, experts: int = 256, slots: int = 36, hidden: int = 2048,
                 moe_inter: int = 512, max_replicas: int = 8,
                 device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.experts, self.slots = experts, slots
        self.hidden, self.moe_inter = hidden, moe_inter
        self.device, self.dtype = torch.device(device), dtype

        self.w_pool = torch.zeros(slots, 3, moe_inter, hidden, device=device, dtype=dtype)
        self.log2phy = torch.full((experts, max_replicas), -1, dtype=torch.int32, device=device)
        self.logcnt = torch.zeros(experts, dtype=torch.int32, device=device)
        # on-device: a Python int would be constant-folded into the graph
        self.cursor = torch.zeros(1, dtype=torch.int32, device=device)

        self._base_ptrs = (self.w_pool.data_ptr(), self.log2phy.data_ptr(),
                           self.logcnt.data_ptr(), self.cursor.data_ptr())

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.transfer_stream = torch.cuda.Stream(device=device)
        self.replays = 0
        self.captures = 0

    def assert_addresses_pinned(self) -> None:
        now = (self.w_pool.data_ptr(), self.log2phy.data_ptr(),
               self.logcnt.data_ptr(), self.cursor.data_ptr())
        assert now == self._base_ptrs, f"pinned tensor reallocated: {self._base_ptrs} -> {now}"

    def load_routing(self, log2phy: torch.Tensor, logcnt: torch.Tensor) -> None:
        assert log2phy.shape == self.log2phy.shape, (log2phy.shape, self.log2phy.shape)
        assert logcnt.shape == self.logcnt.shape, (logcnt.shape, self.logcnt.shape)
        self.log2phy.copy_(log2phy)
        self.logcnt.copy_(logcnt)
        self.assert_addresses_pinned()

    def load_expert(self, slot: int, w: torch.Tensor) -> None:
        assert w.shape == (3, self.moe_inter, self.hidden), w.shape
        self.w_pool[slot].copy_(w)

    def _route(self, expert: torch.Tensor) -> torch.Tensor:
        """expert [T] -> slot [T]; round-robin so each replica carries w/logcnt."""
        e = expert.long()
        cnt = self.logcnt.index_select(0, e).long()
        k = self.cursor.long() % cnt.clamp(min=1)
        row = self.log2phy.index_select(0, e).long()
        return row.gather(1, k.unsqueeze(1)).squeeze(1)

    def forward(self, x: torch.Tensor, expert: torch.Tensor) -> torch.Tensor:
        """x [T, hidden], expert [T] -> [T, hidden]. SwiGLU, one expert per token."""
        w = self.w_pool.index_select(0, self._route(expert))
        xv = x.unsqueeze(-1).to(w.dtype)
        gate = torch.bmm(w[:, GATE], xv).squeeze(-1)
        up = torch.bmm(w[:, UP], xv).squeeze(-1)
        act = (torch.nn.functional.silu(gate) * up).unsqueeze(-1)
        return torch.bmm(w[:, DOWN].transpose(1, 2), act).squeeze(-1)

    def capture(self, x: torch.Tensor, expert: torch.Tensor, warmup: int = 3
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.static_x = x.clone()
        self.static_expert = expert.clone().to(torch.int32)

        # warm up off-stream, or cuBLAS lazy init lands inside the capture
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

    def stage_expert_async(self, slot: int, w: torch.Tensor) -> torch.cuda.Event:
        """Copy into `slot` on transfer_stream. Safe to overlap replay() while the
        slot is unreachable from log2phy."""
        assert w.shape == (3, self.moe_inter, self.hidden), w.shape
        self.transfer_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.transfer_stream):
            self.w_pool[slot].copy_(w, non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(self.transfer_stream)
        return ev

    def commit_routing(self, expert: int, slot: int, replica: int = 0,
                       new_logcnt: Optional[int] = None) -> None:
        self.log2phy[expert, replica] = slot
        if new_logcnt is not None:
            self.logcnt[expert] = new_logcnt
        self.assert_addresses_pinned()

    def set_cursor(self, value: int) -> None:
        self.cursor.fill_(value)
