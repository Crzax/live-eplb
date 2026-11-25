"""Standard-dataset prompt sets -> results/prompts_<phase>.jsonl ({"text": ...}).

Requires external access to the HF hub.
"""
import argparse, json, os, random

from datasets import load_dataset

SOURCES = dict(
    phaseA_code=[
        ("deepmind/code_contests", "train", "description"),
    ],
    phaseB_math_wiki=[
        ("swulling/gsm8k_chinese", "train", "question_zh-cn"),
        ("weitianwen/cmath", "validation", "question"),
        ("weitianwen/cmath", "test", "question"),
    ],
)


def take(name, split, field, limit, seed=1234):
    """-> list[str] of prompts, len >= 40 chars, deterministic shuffle."""
    buf = []
    for i, row in enumerate(load_dataset(name, split=split, streaming=True)):
        txt = (row.get(field) or "").strip().replace("\r", "")
        if len(txt) >= 40:
            buf.append(txt)
        if i >= limit * 4 and len(buf) >= limit:
            break
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
