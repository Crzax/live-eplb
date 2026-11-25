"""Build large, standard-dataset prompt sets for router-trace collection.

Needs external access (HF hub) -> export the proxy first, e.g.
    export https_proxy=http://PROXY_HOST:PORT
    export http_proxy=...same...
and unset it afterwards. Writes one JSON record per line:

    results/prompts_phaseA_code.jsonl        {"text": ...}
    results/prompts_phaseB_math_wiki.jsonl   {"text": ...}
"""
import argparse, json, os, sys

SOURCES = dict(
    phaseA_code=[
        # competitive-programming problem statements, English, dense code domain
        ("deepmind/code_contests", "train", "description"),
    ],
    phaseB_math_wiki=[
        # GSM8K translated to Chinese
        ("swulling/gsm8k_chinese", "train", "question_zh-cn"),
        # Chinese elementary math competition problems (has no train split)
        ("weitianwen/cmath", "validation", "question"),
        ("weitianwen/cmath", "test", "question"),
    ],
)


def take(name, split, field, limit, seed=1234):
    from datasets import load_dataset
    ds = load_dataset(name, split=split, streaming=True)
    out, seen = [], 0
    buf = []
    for row in ds:
        txt = (row.get(field) or "").strip().replace("\r", "")
        if 40 <= len(txt):
            buf.append(txt)
        seen += 1
        if seen >= limit * 4 and len(buf) >= limit:
            break
    import random
    random.Random(seed).shuffle(buf)
    return buf[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=2000)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    for phase, specs in SOURCES.items():
        recs = []
        for name, split, field in specs:
            n = max(args.per_source // len(specs), 200)
            try:
                got = take(name, split, field, n)
            except Exception as ex:
                print(f"  SKIP {name}: {type(ex).__name__}: {ex}", flush=True)
                continue
            print(f"  {name}: got {len(got)}", flush=True)
            recs += got
        path = os.path.join(args.out_dir, f"prompts_{phase}.jsonl")
        with open(path, "w") as f:
            for txt in recs:
                f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")
        print(f"{phase}: {len(recs)} prompts -> {path}", flush=True)


if __name__ == "__main__":
    main()
