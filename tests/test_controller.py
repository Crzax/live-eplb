"""Stage 4 tests: the controller state machine, on one GPU with a fake swapper.

Run: python tests/test_controller.py
"""
import os, sys, time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from live_eplb.controller import LiveEPLBController, State
from live_eplb.p2p import P2PSwapper

R, EPR, RED = 8, 32, 4
E, SPR = R * EPR, EPR + RED
RANK = 1


class FakeSwapper:
    """P2PSwapper's interface, with pulls as local fills so one process is
    enough. delay_cycles keeps the copy stream busy to exercise COPYING; a
    pulled slot ends up holding src_rank * 100 + src_slot.

    Records every batch, whether each commit saw a completed event, and any
    destination slot that was still routable when it was written.
    """
    local_tasks = P2PSwapper.local_tasks

    def __init__(self, shard, delay_cycles=0):
        self.rank, self.shard = RANK, shard
        self.stream = torch.cuda.Stream()
        self.delay = delay_cycles
        self.batches, self.commits, self.unsafe = [], [], []
        self.event = None

    def pull_batch(self, tasks):
        torch.cuda.synchronize()
        # tasks name pool slots, the table layer-local ones; compare in pool ids
        l2p = self.shard.log2phy.cpu()
        off = (torch.arange(l2p.shape[0]) * SPR).view(-1, 1, 1)
        live = set(torch.where(l2p >= 0, l2p + off, -1).flatten().tolist())
        self.unsafe += [dst for dst, _, _ in tasks if dst in live]
        self.batches.append(list(tasks))
        s = self.stream
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            if self.delay:
                torch.cuda._sleep(self.delay)
            for dst, rk, src in tasks:
                self.shard.pool[dst].fill_(rk * 100 + src)
            self.event = torch.cuda.Event()
            self.event.record(s)
        return self.event

    def commit(self, l, nl, c, nc, stream=None):
        self.commits.append(self.event is None or self.event.query())
        l.copy_(nl)
        c.copy_(nc)


def make(delay_cycles=0, layers=1, **kw):
    """-> (controller, swapper, shard) for RANK, every layer holding its own
    base experts. kw overrides the controller's constructor arguments."""
    l2p = torch.full((layers, E, 8), -1, dtype=torch.int32)
    cnt = torch.zeros(layers, E, dtype=torch.int32)
    mine = torch.arange(RANK * EPR, (RANK + 1) * EPR)
    l2p[:, mine, 0] = torch.arange(EPR, dtype=torch.int32)
    cnt[:, mine] = 1
    shard = SimpleNamespace(log2phy=l2p.cuda(), logcnt=cnt.cuda(),
                            pool=torch.zeros(layers * SPR, 4, device="cuda"))
    sw = FakeSwapper(shard, delay_cycles)
    args = dict(num_layers=layers, num_ranks=R, experts_per_rank=EPR,
                redundant_per_rank=RED, ema_decay=0.0, warmup_steps=2,
                cooldown_steps=3)
    args.update(kw)
    ctrl = LiveEPLBController(sw, shard, **args)
    sw.commits.clear()  # the constructor fires a no-op one, to load the kernel
    return ctrl, sw, shard


def hot(lo, hi, layers=1, w=100.0):
    """-> counts [layers, E] with experts [lo, hi) carrying w times the rest."""
    c = torch.ones(layers, E)
    c[:, lo:hi] = w
    return c


def run_until(ctrl, counts, want="commit", limit=400):
    """Step with counts [L, E] until `want` is returned; -> actions seen."""
    seen = []
    for _ in range(limit):
        seen.append(ctrl.on_step(counts))
        if seen[-1] == want:
            return seen
        time.sleep(0.002)
    raise AssertionError(f"no {want} in {limit} steps, tail {seen[-10:]}")


def test_uniform_never_plans():
    ctrl, sw, _ = make()
    acts = [ctrl.on_step(torch.ones(1, E)) for _ in range(100)]
    assert set(acts) <= {"warmup", "idle"}, set(acts)
    assert ctrl.plans == 0 and sw.batches == []


def test_skew_goes_through_every_state_in_order():
    # idle may lead: the gate reads reduced statistics, and nothing is reduced
    # until the first check_every boundary
    ctrl, sw, shard = make()
    acts = run_until(ctrl, hot(0, 16))
    order = [a for i, a in enumerate(acts) if i == 0 or a != acts[i - 1]]
    assert order[0] == "warmup" and set(order[:2]) <= {"warmup", "idle"}, order
    assert [a for a in order if a not in ("warmup", "idle")] == [
        "planning", "copying", "commit"], order
    assert ctrl.state is State.IDLE and ctrl.commits == 1


def test_commit_installs_rank_local_table():
    ctrl, sw, shard = make()
    run_until(ctrl, hot(0, 16))
    l2p, cnt = shard.log2phy[0].cpu(), shard.logcnt[0].cpu()
    assert l2p.max() < SPR, l2p.max()
    for e in range(RANK * EPR, (RANK + 1) * EPR):
        assert l2p[e, 0] == e - RANK * EPR
    extra = [(e, int(s)) for e in range(E) for s in l2p[e, :cnt[e]] if s >= EPR]
    assert extra, "rank 1 got no replica of the hot experts"
    for e, s in extra:
        assert e < 16, (e, s)
        assert shard.pool[s].eq(0 * 100 + e).all(), (e, s)


def test_copying_never_blocks():
    ctrl, sw, shard = make(delay_cycles=int(4e8))
    before = shard.log2phy.clone()
    acts, worst, snaps = [], 0.0, []
    for _ in range(2000):
        t0 = time.perf_counter()
        a = ctrl.on_step(hot(0, 16))
        dt = time.perf_counter() - t0
        acts.append(a)
        if a == "copying":
            worst = max(worst, dt)
            # clone, do not compare: torch.equal syncs, which would wait out the
            # very copy this test is trying to observe in flight
            snaps.append(shard.log2phy.clone())
        if a == "commit":
            break
        time.sleep(0.001)
    assert all(torch.equal(s, before) for s in snaps), "table changed mid-copy"
    assert acts[-1] == "commit", acts[-10:]
    assert acts.count("copying") > 10, acts.count("copying")
    assert worst < 5e-3, f"on_step blocked {worst * 1e3:.2f} ms while copying"
    assert sw.commits == [True], "committed before the copy event completed"


def test_cooldown_rate_limits_planning():
    # threshold 1.0 keeps the layout above trigger forever, so only the cooldown
    # decides when planning may happen again
    ctrl, _, _ = make(cooldown_steps=60, threshold=1.0)
    run_until(ctrl, hot(0, 16))
    after_commit = ctrl.step
    for _ in range(40):
        ctrl.on_step(hot(64, 80))
        time.sleep(0.001)
    assert ctrl.plans == 1, ctrl.plans
    for _ in range(60):
        ctrl.on_step(hot(64, 80))
        time.sleep(0.001)
    assert ctrl.plans == 2 and ctrl.step - after_commit >= 60, (ctrl.plans, ctrl.step)


def test_unchanged_plan_copies_nothing():
    ctrl, sw, _ = make(cooldown_steps=1, threshold=1.0)
    run_until(ctrl, hot(0, 16))
    n = len(sw.batches)
    for _ in range(100):
        ctrl.on_step(hot(0, 16))
        time.sleep(0.001)
    assert ctrl.plans > 1
    assert len(sw.batches) == n and ctrl.commits == 1, (len(sw.batches), ctrl.commits)


def test_occupied_slot_is_unlinked_before_overwrite():
    """A second plan rewrites redundant slots that still serve the first plan's
    replicas; none of them may be routable while the copy is in flight."""
    ctrl, sw, shard = make()
    run_until(ctrl, hot(0, 16))
    first = {s for s in shard.log2phy.flatten().tolist() if s >= EPR}
    run_until(ctrl, hot(64, 80))
    rewritten = {dst for dst, _, _ in sw.batches[-1]} & first
    assert rewritten, "test is vacuous: no occupied slot was rewritten"
    assert sw.unsafe == [], f"slots {sw.unsafe} were live while being overwritten"
    l2p = shard.log2phy[0].cpu()
    for s in rewritten:
        e = int((l2p == s).nonzero()[0, 0])
        assert 64 <= e < 80 and shard.pool[s].eq(2 * 100 + e - 64).all(), (s, e)


def layered_hot(layers, width=8):
    """-> counts [layers, E] whose per-layer hot set is a different slice of
    rank 0's experts, so this rank is a replica target in all of them."""
    return torch.cat([hot(width * l, width * (l + 1)) for l in range(layers)])


def test_multilayer_pulls_land_in_their_own_layer():
    """Every layer plans its own hot set and their replica slots share the same
    layer-local ids; a pull must still write only its own layer's pool range."""
    L = 4
    ctrl, sw, shard = make(layers=L)
    run_until(ctrl, layered_hot(L))
    l2p, cnt = shard.log2phy.cpu(), shard.logcnt.cpu()
    assert l2p.max() < SPR, "table should stay layer-local"
    replicas = [(l, e, int(s)) for l in range(L) for e in range(E)
                for s in l2p[l, e, :cnt[l, e]] if s >= EPR]
    assert len({l for l, _, _ in replicas}) == L, f"only {replicas} came back"
    for l, e, s in replicas:
        # FakeSwapper stamped src_rank * 100 + the source pool slot
        want = (e // EPR) * 100 + l * SPR + e % EPR
        assert shard.pool[l * SPR + s].eq(want).all(), (l, e, s)
    assert sw.unsafe == [], sw.unsafe


def test_one_skewed_layer_among_balanced_ones_triggers():
    """One bad layer in 16 must still be rebalanced. Averaging over layers
    buries it: the mean sits near 1.07 while the layer itself is above 2."""
    L = 16
    counts = torch.ones(L, E)
    counts[0, :16] = 10.0
    ctrl, _, shard = make(layers=L)
    for _ in range(ctrl.check_every):           # first reduction, before any plan
        ctrl.on_step(counts)
    before = ctrl.layer_imbalance()
    assert before.max() > 2.0, before.max()
    assert before.mean() < 1.2, f"test is vacuous, mean is {before.mean():.3f}"

    run_until(ctrl, counts)
    l2p, cnt = shard.log2phy.cpu(), shard.logcnt.cpu()
    got = {l for l in range(L) for e in range(E)
           for s in l2p[l, e, :cnt[l, e]].tolist() if s >= EPR}
    assert got == {0}, f"layers {got} got replicas, want only layer 0"
    assert ctrl.layer_imbalance()[0] < before.max()


def test_multilayer_pulls_are_paced():
    """A stack's pulls exceed the per-step cap, so COPYING spans several steps
    and each slot is pulled exactly once."""
    L = 4
    ctrl, sw, _ = make(layers=L, layers_per_step=1)
    acts = run_until(ctrl, layered_hot(L))
    issued = [t for b in sw.batches for t in b]
    assert len(issued) > RED, f"test is vacuous: only {len(issued)} pulls"
    assert max(len(b) for b in sw.batches) <= RED, [len(b) for b in sw.batches]
    assert acts.count("copying") > 1, acts
    assert len(issued) == len(set(issued)), "a slot was pulled twice"


def test_imbalance_reports_the_layout_in_force_while_copying():
    """phy2log, which imbalance() reads, must not adopt the new placement until
    it is committed -- otherwise the reported balance improves on steps that are
    still serving from the old table."""
    ctrl, _, _ = make(delay_cycles=200_000_000, layers=1)
    counts = hot(0, 16)
    before = ctrl.phy2log.clone()
    seen = []
    for _ in range(2000):
        a = ctrl.on_step(counts)
        seen.append(a)
        if a == "copying":
            assert torch.equal(ctrl.phy2log, before), "adopted before commit"
        if a == "commit":
            break
        time.sleep(0.001)
    assert seen[-1] == "commit", seen[-10:]
    assert seen.count("copying") > 10, seen.count("copying")
    assert not torch.equal(ctrl.phy2log, before), "never adopted"


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: no GPU")
        sys.exit(0)
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
