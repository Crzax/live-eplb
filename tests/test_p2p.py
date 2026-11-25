"""Stage 3 tests: cross-process peer copy.

torchrun --nproc_per_node=2 tests/test_p2p.py
"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist

from live_eplb.p2p import P2PSwapper

SLOTS, D = 12, 64
BASE, RED = 8, 4


def make(rank):
    pool = torch.empty(SLOTS, D, D, device="cuda", dtype=torch.bfloat16)
    for s in range(SLOTS):
        pool[s] = float(rank * 100 + s) / 1000.0
    sw = P2PSwapper(pool, rank, int(os.environ["WORLD_SIZE"]))
    sw.exchange_handles()
    return sw, pool


def test_handles_exchanged(sw, pool, rank, world):
    assert len(sw.peers) == world - 1, (sw.peers, world)


def test_pull_is_bit_exact(sw, pool, rank, world):
    peer = (rank + 1) % world
    sw.pull(BASE, peer, 3)
    sw.stream.synchronize()
    want = torch.full_like(pool[BASE], float(peer * 100 + 3) / 1000.0)
    assert torch.equal(pool[BASE], want), (pool[BASE].flatten()[0], want.flatten()[0])


def test_pull_leaves_other_slots_alone(sw, pool, rank, world):
    before = pool[:BASE].clone()
    sw.pull(BASE, (rank + 1) % world, 0)
    sw.stream.synchronize()
    assert torch.equal(pool[:BASE], before)


def test_multi_slot_pull(sw, pool, rank, world):
    peer = (rank + 1) % world
    sw.pull(BASE, peer, 0, n_slots=RED)
    sw.stream.synchronize()
    for i in range(RED):
        want = float(peer * 100 + i) / 1000.0
        assert torch.equal(pool[BASE + i],
                           torch.full_like(pool[BASE + i], want)), i


def test_coalesce_merges_runs(sw, pool, rank, world):
    f = P2PSwapper._coalesce
    assert f([(8, 1, 0), (9, 1, 1), (10, 1, 2)]) == [(8, 1, 0, 3)]
    assert f([(8, 1, 0), (9, 1, 5)]) == [(8, 1, 0, 1), (9, 1, 5, 1)]
    assert f([(8, 1, 0), (9, 2, 1)]) == [(8, 1, 0, 1), (9, 2, 1, 1)]
    assert f([]) == []
    merged = f([(9, 1, 1), (8, 1, 0)])
    assert merged == [(8, 1, 0, 2)], merged


def test_pull_batch_correct(sw, pool, rank, world):
    peer = (rank + 1) % world
    tasks = [(BASE + i, peer, i) for i in range(RED)]
    ev = sw.pull_batch(tasks)
    ev.synchronize()
    for i in range(RED):
        want = float(peer * 100 + i) / 1000.0
        assert torch.equal(pool[BASE + i], torch.full_like(pool[BASE + i], want)), i


def test_commit_applies_table(sw, pool, rank, world):
    log2phy = torch.full((16, 4), -1, dtype=torch.int32, device="cuda")
    logcnt = torch.zeros(16, dtype=torch.int32, device="cuda")
    new_l = log2phy.clone(); new_c = logcnt.clone()
    new_l[3, 0] = 9; new_c[3] = 1
    ptr = (log2phy.data_ptr(), logcnt.data_ptr())
    sw.commit(log2phy, new_l, logcnt, new_c)
    torch.cuda.synchronize()
    assert torch.equal(log2phy, new_l) and torch.equal(logcnt, new_c)
    assert (log2phy.data_ptr(), logcnt.data_ptr()) == ptr


def test_commit_rejects_bad_dtype(sw, pool, rank, world):
    bad = torch.zeros(16, 4, dtype=torch.int64, device="cuda")
    cnt = torch.zeros(16, dtype=torch.int32, device="cuda")
    try:
        sw.commit(bad, bad.clone(), cnt, cnt.clone())
    except RuntimeError:
        return
    raise AssertionError("int64 table should be rejected")


def test_local_tasks_folds_the_layer_into_the_slot(sw, pool, rank, world):
    """Same layer-local dst slot in two layers must come back as two different
    pool slots, or one layer's pull would overwrite the other's."""
    tasks = [(0, 1, 1, 0, 8, 33), (0, 0, 2, 1, 8, 2), (1, 1, 3, 0, 8, 35)]
    mine = sw.local_tasks(tasks, SLOTS)
    expect = ([(8, 1, 1), (SLOTS + 8, 1, SLOTS + 3)] if rank == 0
              else [(8, 0, 2)] if rank == 1 else [])
    assert mine == expect, (rank, mine)


def test_local_tasks_rejects_self_copy(sw, pool, rank, world):
    """The planner cannot emit src_rank == dst_rank; assert we would notice."""
    bad = [(0, rank, 1, rank, 8, 1)]
    try:
        sw.local_tasks(bad, SLOTS)
    except AssertionError:
        return
    raise AssertionError("self-copy went undetected")


def test_oob_slot_rejected(sw, pool, rank, world):
    try:
        sw.pull(SLOTS, (rank + 1) % world, 0)
    except RuntimeError:
        return
    raise AssertionError("out-of-range dst_slot should be rejected")


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    sw, pool = make(rank)
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        for s in range(BASE, SLOTS):          # reset scratch slots between tests
            pool[s] = float(rank * 100 + s) / 1000.0
        torch.cuda.synchronize()
        try:
            fn(sw, pool, rank, world)
            if rank == 0:
                print(f"PASS {name}", flush=True)
        except Exception as ex:
            failed += 1
            if rank == 0:
                print(f"FAIL {name}: {type(ex).__name__}: {ex}", flush=True)
    if rank == 0:
        print(f"\n{len(fns) - failed}/{len(fns)} passed", flush=True)

    sw.close()
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(1 if failed else 0)
