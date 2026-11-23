"""Stage 3: cross-process expert migration over NVLink.

torchrun --nproc_per_node=8 tutorials/03_p2p_bandwidth.py

Four checks:
  1. correctness  - pulled bytes are bit-identical to the source rank's
  2. bandwidth    - single slot and 40-layer batch, against nvidia-smi topo
  3. commit       - routing table swapped by the CUDA kernel, no atomics
  4. overlap      - a pull does not stall replay on the compute stream
"""
import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from live_eplb.p2p import P2PSwapper

SLOTS, HIDDEN, MOE_INTER = 36, 2048, 512
BASE, RED = 32, 4
EXPERTS, MAX_REP = 256, 8
SLOT_MB = 3 * MOE_INTER * HIDDEN * 2 / 1e6


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    # each rank fills its pool with a rank-identifiable pattern
    pool = torch.empty(SLOTS, 3, MOE_INTER, HIDDEN, device="cuda", dtype=torch.bfloat16)
    for s in range(SLOTS):
        pool[s] = float(rank * 100 + s) / 1000.0

    sw = P2PSwapper(pool, rank, world)
    sw.exchange_handles()

    log(rank, f"world={world} slot={SLOT_MB:.2f} MB pool={pool.nbytes/1e9:.2f} GB")
    log(rank, f"rank0 peers mapped: {sw.peers}")

    # ---------------------------------------------------------- 1. correctness
    src_rank = (rank + 1) % world
    src_slot = 7
    sw.pull(BASE, src_rank, src_slot)
    sw.stream.synchronize()
    want = float(src_rank * 100 + src_slot) / 1000.0
    got = pool[BASE].flatten()[0].item()
    exact = bool((pool[BASE] == torch.tensor(want, dtype=torch.bfloat16,
                                             device="cuda")).all())
    ok = torch.tensor([1.0 if exact else 0.0], device="cuda")
    dist.all_reduce(ok)
    log(rank, "")
    log(rank, "1. correctness")
    log(rank, f"   rank0 pulled slot {src_slot} from rank {src_rank}: "
              f"got {got:.3f} want {want:.3f}")
    log(rank, f"   ranks with bit-identical pull: {int(ok.item())}/{world}")

    # ------------------------------------------------------------ 2. bandwidth
    def bench(tasks, iters=20):
        ev = sw.pull_batch(tasks); ev.synchronize()
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        for _ in range(iters):
            sw.pull_batch(tasks).synchronize()
        dt = (time.perf_counter() - t0) / iters
        mb = len(tasks) * SLOT_MB
        return dt * 1e3, mb / 1e3 / dt

    one_ms, one_gb = bench([(BASE, src_rank, 3)])
    # 40 layers, one slot each: scattered sources, cannot be coalesced
    scattered = [(BASE + (i % RED), (rank + 1 + i % (world - 1)) % world, i % BASE)
                 for i in range(40)]
    sc_ms, sc_gb = bench(scattered)
    # same volume as one contiguous run from a single peer
    contig = [(i % SLOTS, src_rank, i % SLOTS) for i in range(RED)]
    ct_ms, ct_gb = bench(contig)
    log(rank, "")
    log(rank, "2. bandwidth (NV18 all-to-all, 18 links/pair)")
    log(rank, f"   1 slot          {SLOT_MB:6.2f} MB  {one_ms:7.3f} ms  {one_gb:7.1f} GB/s")
    log(rank, f"   {RED} contiguous   {RED*SLOT_MB:6.2f} MB  {ct_ms:7.3f} ms  {ct_gb:7.1f} GB/s")
    log(rank, f"   40 scattered   {40*SLOT_MB:6.1f} MB  {sc_ms:7.3f} ms  {sc_gb:7.1f} GB/s")
    log(rank, f"   small copies are launch-bound; coalescing consecutive slots is")
    log(rank, f"   what closes the gap to the ~376 GB/s single-pair ceiling")

    # --------------------------------------------------------------- 3. commit
    log2phy = torch.full((EXPERTS, MAX_REP), -1, dtype=torch.int32, device="cuda")
    logcnt = torch.zeros(EXPERTS, dtype=torch.int32, device="cuda")
    for e in range(BASE):
        log2phy[e, 0] = e
        logcnt[e] = 1
    new_log2phy = log2phy.clone()
    new_logcnt = logcnt.clone()
    new_log2phy[5, 1] = BASE
    new_logcnt[5] = 2
    ptr_before = (log2phy.data_ptr(), logcnt.data_ptr())

    sw.commit(log2phy, new_log2phy, logcnt, new_logcnt)
    torch.cuda.synchronize()
    applied = bool((log2phy == new_log2phy).all() and (logcnt == new_logcnt).all())
    log(rank, "")
    log(rank, "3. commit_routing_table (single launch, no atomics)")
    log(rank, f"   table applied          : {applied}")
    log(rank, f"   addresses unchanged    : "
              f"{(log2phy.data_ptr(), logcnt.data_ptr()) == ptr_before}")

    t0 = time.perf_counter()
    for _ in range(50):
        sw.commit(log2phy, new_log2phy, logcnt, new_logcnt)
        torch.cuda.synchronize()
    log(rank, f"   commit latency         : {(time.perf_counter()-t0)/50*1e6:.1f} us")

    # -------------------------------------------------------------- 4. overlap
    # A peer copy does NOT run for free: it contends for memory bandwidth, and
    # the receiving SMs stall on the same HBM the GEMM is reading. Measure how
    # much, rather than assuming the copy engine makes it invisible.
    x = torch.randn(4096, HIDDEN, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(HIDDEN, HIDDEN, device="cuda", dtype=torch.bfloat16)

    def step():
        torch.mm(x, w)

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        step(); torch.cuda.synchronize()
    solo = (time.perf_counter() - t0) / 20 * 1e3

    ev = sw.pull_batch(scattered)
    spins, during = 0, []
    while not ev.query():
        t1 = time.perf_counter()
        step(); torch.cuda.synchronize()
        during.append((time.perf_counter() - t1) * 1e3)
        spins += 1
    ev.synchronize()
    dist.barrier()
    log(rank, "")
    log(rank, "4. interference with compute")
    log(rank, f"   step alone                  : {solo:.3f} ms")
    if during:
        avg = sum(during) / len(during)
        log(rank, f"   step during {40*SLOT_MB:.0f} MB pull    : {avg:.3f} ms "
                  f"over {spins} steps ({avg/solo:.2f}x)")
        log(rank, f"   -> the copy is not free. It never blocks the stream, but it")
        log(rank, f"      does compete for HBM, so a step slows while it runs.")
    else:
        log(rank, f"   pull finished before the first step completed")
    if during:
        extra = sum(during) - solo * len(during)
        log(rank, f"   total extra latency for the whole migration: {extra:.2f} ms")

    sw.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
