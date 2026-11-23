"""Stage 1 tests: plan_incremental_redundant invariants.

Run: python tests/test_planner.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from live_eplb.planner import (
    plan_incremental_redundant,
    rank_load,
    static_phy2log,
    log2phy_from_phy2log,
)

R, EPR, RED = 8, 32, 4
SPR = EPR + RED
E = R * EPR
NPHY = R * SPR


def skewed(num_layers, alpha=1.1, seed=0):
    """Zipf-skewed load, like a real hot-expert distribution."""
    g = torch.Generator().manual_seed(seed)
    base = 1.0 / torch.arange(1, E + 1).float() ** alpha
    w = torch.empty(num_layers, E)
    for l in range(num_layers):
        perm = torch.randperm(E, generator=g)
        w[l] = base[perm] * (0.5 + torch.rand(E, generator=g))
    return w * 1e4


def check_inverse(phy2log, log2phy, logcnt):
    for l in range(phy2log.shape[0]):
        for s in range(phy2log.shape[1]):
            e = int(phy2log[l, s])
            if e < 0:
                continue
            reps = [int(x) for x in log2phy[l, e] if x >= 0]
            assert s in reps, f"slot {s} (expert {e}) missing in log2phy"
        for e in range(logcnt.shape[1]):
            reps = [int(x) for x in log2phy[l, e] if x >= 0]
            assert len(reps) == int(logcnt[l, e]), (e, reps, int(logcnt[l, e]))
            assert len(set(reps)) == len(reps)
            for s in reps:
                assert int(phy2log[l, s]) == e


def test_base_slots_frozen():
    w = skewed(40)
    phy2log, _, _, _ = plan_incremental_redundant(w)
    assert phy2log.shape == (40, NPHY)
    for r in range(R):
        block = phy2log[:, r * SPR:r * SPR + EPR]
        expect = torch.arange(r * EPR, (r + 1) * EPR).expand(40, EPR)
        assert torch.equal(block, expect), f"base slots of rank {r} moved"


def test_inverse_consistency():
    phy2log, log2phy, logcnt, _ = plan_incremental_redundant(skewed(6))
    check_inverse(phy2log, log2phy, logcnt)


def test_improves_imbalance():
    w = skewed(40)
    l0 = rank_load(static_phy2log(40), w)
    phy2log, _, _, _ = plan_incremental_redundant(w)
    l1 = rank_load(phy2log, w)
    r0 = (l0.max(-1).values / l0.mean(-1)).mean()
    r1 = (l1.max(-1).values / l1.mean(-1)).mean()
    assert r1 < r0 * 0.75, (r0.item(), r1.item())
    assert torch.allclose(l1.sum(-1), l0.sum(-1), rtol=1e-5)


def test_no_duplicate_on_rank():
    phy2log, _, _, _ = plan_incremental_redundant(skewed(40))
    for r in range(R):
        block = phy2log[:, r * SPR:(r + 1) * SPR]
        for l in range(block.shape[0]):
            vals = [int(v) for v in block[l] if v >= 0]
            assert len(set(vals)) == len(vals), f"rank {r} layer {l} duplicate"


def test_copy_tasks_are_valid():
    phy2log, _, _, tasks = plan_incremental_redundant(skewed(4))
    assert len(tasks) > 0
    seen = set()
    for l, src_rank, src_slot, dst_rank, dst_slot, e in tasks:
        assert src_rank == e // EPR and src_slot == e % EPR
        assert src_rank != dst_rank
        assert EPR <= dst_slot < SPR
        assert int(phy2log[l, dst_rank * SPR + dst_slot]) == e
        key = (l, dst_rank, dst_slot)
        assert key not in seen
        seen.add(key)


def test_diff_mode_no_change():
    phy2log, _, _, _ = plan_incremental_redundant(skewed(4))
    _, _, _, tasks2 = plan_incremental_redundant(skewed(4), prev_phy2log=phy2log)
    assert tasks2 == []


def test_diff_mode_after_drift():
    phy2log1, _, _, _ = plan_incremental_redundant(skewed(4, seed=1))
    _, _, _, tasks = plan_incremental_redundant(skewed(4, seed=2), prev_phy2log=phy2log1)
    assert 0 < len(tasks) <= R * RED * 4


def test_uniform_weight_no_replica():
    """Replication cannot help uniform load, so nothing should move."""
    w = torch.ones(3, E)
    phy2log, log2phy, logcnt, tasks = plan_incremental_redundant(w)
    assert tasks == []
    assert (logcnt == 1).all()
    for r in range(R):
        assert (phy2log[:, r * SPR:r * SPR + EPR] >= 0).all()
        assert (phy2log[:, r * SPR + EPR:(r + 1) * SPR] == -1).all(), "unused slots must stay -1"
    l = rank_load(phy2log, w)
    assert torch.allclose(l, torch.full_like(l, l[0, 0]))


def test_boundary_max_replicas():
    """An ultra-hot expert spreads to every other rank; peak load keeps dropping."""
    w = torch.zeros(2, E)
    w[:, 0] = 1e6
    w[:, 1:] = 1.0
    phy2log, log2phy, logcnt, _ = plan_incremental_redundant(w)
    assert (logcnt[:, 0] == R).all(), logcnt[:, 0].tolist()
    for r in range(R):
        vals = [int(v) for v in phy2log[0, r * SPR:(r + 1) * SPR] if v >= 0]
        assert len(set(vals)) == len(vals)
    base = rank_load(static_phy2log(2), w)
    after = rank_load(phy2log, w)
    assert after[0, 0] < base[0, 0] / 7


def test_single_hot_expert_rank_bound():
    """All redundant slots used on the single hottest expert still cannot exceed 8 replicas."""
    w = torch.zeros(1, E)
    w[0, 0] = 1e7
    phy2log, _, logcnt, tasks = plan_incremental_redundant(w)
    assert int(logcnt[0, 0]) == R
    assert len(tasks) <= R * RED


def test_shape_mismatch_raises():
    try:
        plan_incremental_redundant(torch.ones(2, E - 1))
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on wrong num_experts")


def test_log2phy_helper_matches():
    phy2log, log2phy, logcnt, _ = plan_incremental_redundant(skewed(2))
    p2l, l2p, lc = log2phy_from_phy2log(phy2log, E)
    assert torch.equal(p2l, phy2log)
    assert l2p.shape == log2phy.shape
    assert torch.equal(l2p[l2p >= 0], log2phy[l2p >= 0])
    assert torch.equal(lc, logcnt)


def test_negative_load_raises():
    w = torch.full((1, E), -1.0)
    try:
        plan_incremental_redundant(w)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on negative weight")


if __name__ == "__main__":
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as ex:
            failed += 1
            print(f"FAIL {name}: {type(ex).__name__}: {ex}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
