#!/usr/bin/env python3
"""Load-test an OpenAI-compatible vLLM server and report latency and throughput.

Metrics (per concurrency level):
  - output tokens/sec  : total generated tokens / wall-clock time for the whole level
  - TTFT p50 / p95     : time to first streamed token
  - ITL  mean          : mean inter-token latency after the first token
  - E2E  p50 / p95     : end-to-end request latency
Each request uses a unique prompt so prefix caching does not inflate results.
"""
import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

WORDS = ("gpu kernel memory tensor latency batch scheduler cache throughput "
         "model token inference cluster network driver compiler runtime").split()


def make_prompt(n_words: int, uid: int) -> str:
    rng = random.Random(uid)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return (f"[request {uid}] Summarize the following notes in detail, "
            f"then expand on each idea:\n{body}")


async def one_request(session, url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft, chunks, usage_tokens = None, 0, None
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens")
            choices = obj.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                chunks += 1
    total = time.perf_counter() - t0
    tokens = usage_tokens if usage_tokens is not None else chunks
    return {"ttft": ttft if ttft is not None else total, "e2e": total, "tokens": tokens}


def pct(values, p):
    values = sorted(values)
    if not values:
        return float("nan")
    k = min(len(values) - 1, max(0, int(round(p / 100 * (len(values) - 1)))))
    return values[k]


async def run_level(args, concurrency):
    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    total_requests = concurrency * args.rounds
    sem = asyncio.Semaphore(concurrency)
    timeout = aiohttp.ClientTimeout(total=None)
    results, failures = [], 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def worker(i):
            nonlocal failures
            async with sem:
                try:
                    results.append(await one_request(
                        session, url, args.model,
                        make_prompt(args.input_words, i + concurrency * 10_000),
                        args.output_tokens))
                except Exception as exc:  # keep going, report failures
                    failures += 1
                    print(f"  request {i} failed: {exc}")

        t0 = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(total_requests)))
        wall = time.perf_counter() - t0

    tokens = sum(r["tokens"] for r in results)
    itls = [(r["e2e"] - r["ttft"]) / max(r["tokens"] - 1, 1) for r in results]
    return {
        "concurrency": concurrency,
        "requests_ok": len(results),
        "requests_failed": failures,
        "output_tok_per_s": round(tokens / wall, 1) if wall else 0.0,
        "ttft_p50_ms": round(pct([r["ttft"] for r in results], 50) * 1000, 1),
        "ttft_p95_ms": round(pct([r["ttft"] for r in results], 95) * 1000, 1),
        "itl_mean_ms": round(statistics.mean(itls) * 1000, 2) if itls else float("nan"),
        "e2e_p50_s": round(pct([r["e2e"] for r in results], 50), 2),
        "e2e_p95_s": round(pct([r["e2e"] for r in results], 95), 2),
    }


async def main_async(args):
    # warm-up so compile/graph capture does not pollute the first level
    print("Warm-up...")
    await run_level(argparse.Namespace(**{**vars(args), "rounds": 1}), 1)

    rows = []
    for c in args.concurrency:
        print(f"Concurrency {c} ({c * args.rounds} requests)...")
        row = await run_level(args, c)
        print("  ", row)
        rows.append(row)
    return rows


def write_outputs(rows, args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = [f"# Benchmark: {args.model}", "",
          f"input ~{args.input_words} words, output max {args.output_tokens} tokens, "
          f"{args.rounds} rounds per concurrency level", "",
          "| " + " | ".join(cols) + " |",
          "|" + "|".join("---" for _ in cols) + "|"]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"Wrote {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--rounds", type=int, default=4, help="requests per concurrent slot")
    p.add_argument("--input-words", type=int, default=256)
    p.add_argument("--output-tokens", type=int, default=256)
    p.add_argument("--out", default="results/bench")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = asyncio.run(main_async(args))
    write_outputs(rows, args)
