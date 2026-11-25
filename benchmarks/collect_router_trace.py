"""Collect real router decisions from Qwen3.6-35B-A3B on two prompt domains.

Phase A = code generation (English), phase B = math + Chinese encyclopedia.
Hook every Qwen3_5MoeTopKRouter; its forward returns
(router_logits, router_scores, router_indices); [2] is [tokens, 8] and is all
we keep. Output counts [L=40, E=256] per phase -> results/router_traces.pt.

--prompts-dir reads results/prompts_<phase>.jsonl written by download_prompts.py
(large standard-dataset sets); omit it to use the 48 hand-written prompts.
Batches by token count: sort by length, pack similar lengths, count only
non-pad positions via the attention mask.

Run on the GPU box:
    python benchmarks/collect_router_trace.py --prompts-dir results --batch 6
"""
import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "MODEL_PATH"

PHASE_A = [
    "Write a Python function that finds the longest palindromic substring in a string.",
    "Implement a thread-safe LRU cache in C++ with O(1) get and put.",
    "Write a regex that matches valid IPv4 addresses and explain each group.",
    "Give me a bash script that archives old nginx logs and deletes files older than 30 days.",
    "Write a SQL query to find the top 3 customers by revenue per region, with ties.",
    "Implement Dijkstra's shortest path in Rust using a binary heap.",
    "Create a React component for an infinite-scrolling photo gallery with lazy loading.",
    "Write a Go HTTP middleware that enforces token-bucket rate limiting per client IP.",
    "Implement a tokenizer for a small expression language in Python, with unit tests.",
    "Write a Dockerfile for a multi-stage build of a Java Spring Boot service.",
    "Implement consistent hashing in Python for distributing keys across cache nodes.",
    "Write an asyncio web scraper that respects robots.txt and retries with backoff.",
    "Implement a bloom filter in C with bit packing and configurable error rate.",
    "Write a Terraform module that provisions an autoscaling group behind an ALB.",
    "Implement merge sort and binary search in Haskell, with quickcheck properties.",
    "Write a git pre-commit hook that blocks commits containing AWS secret keys.",
    "Implement a sliding-window log parser in Python that emits per-minute p99 latency.",
    "Write a TypeScript function that deeply merges two JSON objects with arrays unioned.",
    "Implement a priority scheduler for tasks with dependencies in Python.",
    "Write a Kubernetes NetworkPolicy that only allows ingress from the api namespace.",
    "Implement undo/redo for a text editor using the command pattern in Java.",
    "Write a Matplotlib script that plots a confusion matrix with per-class accuracy.",
    "Implement a WebSocket chat server in Node.js with rooms and heartbeat.",
    "Write a recursive descent parser for arithmetic with operator precedence in C.",
]

PHASE_B = [
    "证明柯西中值定理，并说明它与拉格朗日中值定理的关系。",
    "计算定积分：从 0 到 1 的 x^x，给出级数展开方法。",
    "简述牛顿三定律在现代力学中的表述方式及其适用边界。",
    "解方程组：3x + 2y - z = 5, x - y + 4z = 2, 2x + 3y + z = 12，写出步骤。",
    "介绍光合作用的光反应和暗反应各自发生的位置与产物。",
    "证明：任意大于 2 的偶数都可以表示为两个素数之和仍然成立的已知部分结果。",
    "简述秦始皇统一度量衡的具体措施及其历史影响。",
    "计算矩阵 [[2,1],[1,2]] 的特征值和特征向量，并说明对角化过程。",
    "介绍板块构造学说如何解释喜马拉雅山脉的形成。",
    "求函数 f(x)=x^3-6x^2+9x+1 的极值点与拐点。",
    "简述RNA转录过程中启动子、增强子与转录因子的作用。",
    "用数学归纳法证明 1^2+2^2+...+n^2 = n(n+1)(2n+1)/6。",
    "介绍唐代科举制度的科目设置与其对宋代文官政治的影响。",
    "计算半径为 r 的球冠体积，并给出积分推导。",
    "简述免疫系统如何区分自身抗原与外来抗原。",
    "求解微分方程 y''+3y'+2y=0 的通解，并讨论特征根情形。",
    "介绍可燃冰的化学组成、形成条件与开采难点。",
    "证明根号 2 是无理数，并推广到根号 p（p 为素数）。",
    "简述三江源地区的生态系统类型及其保护意义。",
    "计算二人连续掷骰子，先掷出总点数 7 者获胜的概率。",
    "介绍敦煌莫高窟的开凿历史与各朝代壁画风格差异。",
    "证明 n 个不同元素的全排列数为 n!，并说明错排数的递推关系。",
    "简述黑洞视界、光子球与吸积盘的物理含义。",
    "求极限 lim(x->0) (sin x - x) / x^3，并给出洛必达与泰勒两种方法。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="results/router_traces.pt")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prompts-dir")
    ap.add_argument("--max-prompts", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeTopKRouter)

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.time()
    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto")
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    model.eval()

    routers = [(n, m) for n, m in model.named_modules()
               if isinstance(m, Qwen3_5MoeTopKRouter)]
    print(f"{len(routers)} MoE routers hooked:", flush=True)
    for n, _ in routers[:3] + routers[-2:]:
        print("  ", n, flush=True)
    assert len(routers) == 40, f"expected 40 MoE routers, got {len(routers)}"

    picked, hooks = [], []
    for _, m in routers:
        hooks.append(m.register_forward_hook(
            lambda _m, _i, o: picked.append(o[2].detach().to("cpu"))))

    def load_prompts(phase, fallback):
        if not args.prompts_dir:
            return fallback
        path = os.path.join(args.prompts_dir, f"prompts_{phase}.jsonl")
        out = [json.loads(l)["text"] for l in open(path)]
        if args.max_prompts:
            out = out[:args.max_prompts]
        return out

    dev = next(model.parameters()).device
    phases = {}
    for name, prompts in (("phaseA_code", load_prompts("phaseA_code", PHASE_A)),
                          ("phaseB_math_wiki",
                           load_prompts("phaseB_math_wiki", PHASE_B))):
        # sort by length so a batch pads as little as possible
        prompts = sorted(prompts, key=len)
        counts = None
        ntok = 0
        n_batches = (len(prompts) + args.batch - 1) // args.batch
        for bi in range(n_batches):
            chunk = prompts[bi * args.batch:(bi + 1) * args.batch]
            picked.clear()
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len).to(dev)
            with torch.no_grad():
                model(input_ids=enc.input_ids,
                      attention_mask=enc.attention_mask)
            assert len(picked) == 40, len(picked)
            keep = enc.attention_mask.flatten().bool().cpu()   # [B*T]
            c = torch.stack([
                torch.bincount(sel[keep].flatten().long(),
                               minlength=256).float()
                for sel in picked])
            counts = c if counts is None else counts + c
            ntok += int(keep.sum()) * 8
            if bi % 40 == 0:
                print(f"{name} {bi + 1}/{n_batches} batches", flush=True)
        phases[name] = dict(counts=counts, prompts=prompts, routed_tokens=ntok)
        print(f"{name}: {ntok} routed assignments over {len(prompts)} prompts",
              flush=True)

    for h in hooks:
        h.remove()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(dict(phases=phases, layers=40, experts=256, top_k=8,
                    model=args.model), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
