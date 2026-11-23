"""Stage 2 tests: the graph must stay valid across weight swaps.

Run: python tests/test_moe_layer.py     (needs one GPU)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from live_eplb.moe_layer import GraphSafeMoEShard, GATE, UP, DOWN

E, SLOTS, H, M, BASE = 32, 12, 128, 64, 8      # small but structurally identical
RED = SLOTS - BASE
T = 8


def ref(x, w):
    """Hand-computed SwiGLU: the control layer. If the graph matches this, the
    swap really happened — as opposed to merely not crashing."""
    xv = x.to(w.dtype)
    return w[DOWN].t() @ (torch.nn.functional.silu(w[GATE] @ xv) * (w[UP] @ xv))


def make(seed=0):
    torch.manual_seed(seed)
    sh = GraphSafeMoEShard(E, SLOTS, H, M, max_replicas=4)
    l2p = torch.full((E, 4), -1, dtype=torch.int32)
    cnt = torch.zeros(E, dtype=torch.int32)
    for e in range(BASE):
        l2p[e, 0] = e
        cnt[e] = 1
    sh.load_routing(l2p.cuda(), cnt.cuda())
    for s in range(BASE):
        sh.load_expert(s, torch.randn(3, M, H, device="cuda", dtype=torch.bfloat16) * 0.05)
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    expert = torch.arange(T, device="cuda", dtype=torch.int32) % BASE
    return sh, x, expert


def rand_w():
    return torch.randn(3, M, H, device="cuda", dtype=torch.bfloat16) * 0.05


def test_graph_matches_eager():
    sh, x, e = make()
    eager = sh.forward(x, e).clone()
    sh.capture(x, e)
    y = sh.replay()
    torch.cuda.synchronize()
    assert torch.allclose(y.float(), eager.float(), atol=5e-2)


def test_graph_matches_hand_computed():
    """Every token, not just the aggregate, must match the reference."""
    sh, x, e = make()
    sh.capture(x, e)
    y = sh.replay()
    torch.cuda.synchronize()
    for t in range(T):
        slot = int(sh.log2phy[int(e[t]), 0])
        want = ref(x[t], sh.w_pool[slot])
        assert torch.allclose(y[t].float(), want.float(), atol=5e-2), f"token {t}"


def test_swap_takes_effect_without_recapture():
    sh, x, e = make()
    sh.capture(x, e)
    sh.replay(); torch.cuda.synchronize()
    captures = sh.captures

    hot = int(e[0])
    new_w = rand_w()
    sh.load_expert(BASE, new_w)
    sh.commit_routing(hot, BASE)
    y = sh.replay()
    torch.cuda.synchronize()

    assert sh.captures == captures, "graph was re-captured"
    assert torch.allclose(y[0].float(), ref(x[0], new_w).float(), atol=5e-2)


def test_swap_actually_changes_output():
    """Guards against a no-op swap silently 'passing' the previous test."""
    sh, x, e = make()
    sh.capture(x, e)
    before = sh.replay().clone()
    torch.cuda.synchronize()
    sh.load_expert(BASE, rand_w())
    sh.commit_routing(int(e[0]), BASE)
    after = sh.replay().clone()
    torch.cuda.synchronize()
    assert not torch.allclose(before[0].float(), after[0].float(), atol=5e-2)


def test_addresses_pinned_across_swaps():
    sh, x, e = make()
    ptrs = (sh.w_pool.data_ptr(), sh.log2phy.data_ptr(), sh.logcnt.data_ptr(),
            sh.cursor.data_ptr())
    sh.capture(x, e)
    for i in range(RED):
        sh.load_expert(BASE + i, rand_w())
        sh.commit_routing(i, BASE + i)
        sh.set_cursor(0)
        sh.replay()
    torch.cuda.synchronize()
    assert ptrs == (sh.w_pool.data_ptr(), sh.log2phy.data_ptr(), sh.logcnt.data_ptr(),
                    sh.cursor.data_ptr())
    sh.assert_addresses_pinned()


def test_replica_round_robin():
    """cursor must select between replicas, which is what makes the planner's
    'each replica carries w/c' load model true at runtime."""
    sh, x, e = make()
    sh.capture(x, e)
    hot = int(e[0])
    wa, wb = rand_w(), rand_w()
    sh.load_expert(BASE, wa)
    sh.load_expert(BASE + 1, wb)
    sh.commit_routing(hot, BASE, replica=0)
    sh.commit_routing(hot, BASE + 1, replica=1, new_logcnt=2)
    for c, w in [(0, wa), (1, wb), (2, wa), (3, wb)]:
        sh.set_cursor(c, expert=hot)
        y = sh.replay()
        torch.cuda.synchronize()
        assert torch.allclose(y[0].float(), ref(x[0], w).float(), atol=5e-2), f"cursor {c}"


def test_batch_spreads_across_replicas():
    """Within ONE step, tokens hitting the same expert must land on different
    replicas. A global cursor would send them all to the same slot, so the
    planner's w/c load model would hold only when averaged over steps."""
    sh, _, _ = make()
    x = torch.randn(8, H, device="cuda", dtype=torch.bfloat16)
    e = torch.zeros(8, device="cuda", dtype=torch.int32)      # all on expert 0
    for i in range(4):
        sh.load_expert(BASE + i, rand_w())
        sh.commit_routing(0, BASE + i, replica=i)
    sh.logcnt[0] = 4

    slots = sh._route(e).tolist()
    assert sorted(slots) == [BASE, BASE, BASE + 1, BASE + 1,
                             BASE + 2, BASE + 2, BASE + 3, BASE + 3], slots

    sh.capture(x, e)
    y = sh.replay()
    torch.cuda.synchronize()
    for t in range(8):
        want = ref(x[t], sh.w_pool[slots[t]])
        assert torch.allclose(y[t].float(), want.float(), atol=5e-2), f"token {t}"


def test_cursor_carries_remainder_across_steps():
    """An uneven split must not always favour the same replica."""
    sh, _, _ = make()
    e = torch.zeros(3, device="cuda", dtype=torch.int32)      # 3 tokens, 2 replicas
    sh.commit_routing(0, BASE, replica=0)
    sh.commit_routing(0, BASE + 1, replica=1, new_logcnt=2)

    first = sh._route(e).tolist()
    sh.advance_cursor(e)
    second = sh._route(e).tolist()
    assert first.count(BASE) == 2 and first.count(BASE + 1) == 1, first
    assert second.count(BASE) == 1 and second.count(BASE + 1) == 2, second


def test_cursor_is_per_expert():
    sh, _, _ = make()
    assert sh.cursor.is_cuda and sh.cursor.numel() == E
    before = sh.cursor.data_ptr()
    sh.set_cursor(7, expert=3)
    assert int(sh.cursor[3]) == 7 and int(sh.cursor[0]) == 0
    sh.set_cursor(0)
    assert int(sh.cursor.sum()) == 0
    assert sh.cursor.data_ptr() == before, "set_cursor reallocated the tensor"


def test_advance_cursor_counts_tokens():
    sh, _, _ = make()
    e = torch.tensor([0, 0, 0, 1], device="cuda", dtype=torch.int32)
    sh.advance_cursor(e)
    assert int(sh.cursor[0]) == 3 and int(sh.cursor[1]) == 1
    assert int(sh.cursor[2]) == 0


def test_async_stage_then_commit():
    sh, x, e = make()
    sh.capture(x, e)
    new_w = rand_w()
    ev = sh.stage_expert_async(BASE, new_w)
    spins = 0
    while not ev.query():
        sh.replay()
        spins += 1
    ev.synchronize()
    sh.commit_routing(int(e[0]), BASE)
    y = sh.replay()
    torch.cuda.synchronize()
    assert torch.allclose(y[0].float(), ref(x[0], new_w).float(), atol=5e-2)
    assert torch.equal(sh.w_pool[BASE], new_w)


def test_untouched_experts_unaffected():
    """Swapping one expert must not disturb the others."""
    sh, x, e = make()
    sh.capture(x, e)
    before = sh.replay().clone()
    torch.cuda.synchronize()
    hot = int(e[0])
    sh.load_expert(BASE, rand_w())
    sh.commit_routing(hot, BASE)
    after = sh.replay().clone()
    torch.cuda.synchronize()
    for t in range(T):
        if int(e[t]) == hot:
            continue
        assert torch.allclose(before[t].float(), after[t].float(), atol=5e-2), f"token {t}"


def test_reload_routing_keeps_address():
    sh, x, e = make()
    sh.capture(x, e)
    ptr = sh.log2phy.data_ptr()
    l2p = torch.full((E, 4), -1, dtype=torch.int32)
    cnt = torch.zeros(E, dtype=torch.int32)
    for i in range(BASE):
        l2p[i, 0] = (i + 1) % BASE
        cnt[i] = 1
    sh.load_routing(l2p.cuda(), cnt.cuda())
    assert sh.log2phy.data_ptr() == ptr
    y = sh.replay()
    torch.cuda.synchronize()
    slot = int(sh.log2phy[int(e[0]), 0])
    assert torch.allclose(y[0].float(), ref(x[0], sh.w_pool[slot]).float(), atol=5e-2)


def test_reallocation_is_detected():
    """The pin assertion must actually fire if someone rebinds a pinned tensor."""
    sh, x, e = make()
    sh.capture(x, e)
    sh.log2phy = sh.log2phy.clone()       # the mistake we want to catch
    try:
        sh.assert_addresses_pinned()
    except AssertionError:
        return
    raise AssertionError("reallocation went undetected")


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
