#!/usr/bin/env python3
"""
End-to-end verification for the reproduced C2C runtime.

Loads the smallest published C2C fuser pair (Qwen3-0.6B receiver +
Qwen2.5-0.5B-Instruct sharer), then answers one prompt twice: once with the
sharer's KV-cache fused into the receiver (`c2c`) and once with the receiver
alone (`baseline`). This is the reproduction check for the Cache-to-Cache
claim.

Run it with the C2C environment's interpreter:

    /path/to/c2c/.venv/bin/python dsh-plugin-c2c/python/verify_c2c.py \
        --repo-root /path/to/c2c

Exit code 0 means the runtime genuinely generated text under both conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the reproduced C2C runtime end to end")
    parser.add_argument("--repo-root", required=True, help="reproduced thu-nics/C2C checkout")
    parser.add_argument("--pair", default="qwen3_0.6b+qwen2.5_0.5b", help="registry pair name")
    parser.add_argument("--device", default="auto", help="auto, mps, cuda, or cpu")
    parser.add_argument("--dtype", default="auto", help="auto, float16, bfloat16, or float32")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--prompt",
        default="What is the capital of France? Answer in one short sentence.",
        help="prompt answered under both conditions",
    )
    args = parser.parse_args(argv)

    daemon_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(daemon_dir))
    from c2c_daemon import C2CRuntime, LoadRequest, resolve_device, resolve_dtype

    repo_root = Path(args.repo_root).expanduser().resolve()
    if not (repo_root / "rosetta").is_dir():
        print(f"FAIL: {repo_root}/rosetta not found — is --repo-root an actual C2C checkout?")
        return 2

    print(f"repo root : {repo_root}")
    print(f"python    : {sys.executable}")
    try:
        import torch
        import transformers

        print(f"torch     : {torch.__version__}")
        print(f"transformers: {transformers.__version__}")
    except ImportError as exc:
        print(f"FAIL: missing dependency: {exc}")
        return 2

    device = resolve_device(args.device)
    print(f"device    : {device} (dtype {resolve_dtype(args.dtype, device)})")
    print()

    runtime = C2CRuntime(repo_root=str(repo_root), idle_unload_seconds=0)
    try:
        print(">>> loading pair", args.pair)
        started = time.time()
        status = runtime.load(
            LoadRequest(pair=args.pair, device=args.device, dtype=args.dtype, repo_root=str(repo_root))
        )
        print(f"    loaded in {status['load_seconds']}s: "
              f"{status['base_model']} + {status['teacher_model']}, "
              f"{status['num_projectors']} projectors, {status['dtype']}")
        print()

        results = {}
        for mode in ("baseline", "c2c"):
            print(f">>> generating (mode={mode})")
            outcome = runtime.generate(
                {"prompt": args.prompt, "mode": mode, "max_new_tokens": args.max_new_tokens}
            )
            results[mode] = outcome
            print(f"    {outcome['generated_tokens']} tokens in {outcome['seconds']}s "
                  f"({outcome['tokens_per_second']} tok/s)")
            print(f"    text: {outcome['text']!r}")
            print()

        print("=" * 72)
        print("PROMPT:", args.prompt)
        print("=" * 72)
        print("\n--- baseline (receiver alone) ---")
        print(results["baseline"]["text"])
        print("\n--- c2c (sharer KV-cache fused) ---")
        print(results["c2c"]["text"])
        print("=" * 72)

        # A successful run is one where both conditions produced text. Whether the
        # fused answer beats the baseline on a single prompt is an anecdote; a real
        # accuracy comparison needs a benchmark subset.
        ok = all(r["generated_tokens"] > 0 for r in results.values())
        identical = results["baseline"]["text"] == results["c2c"]["text"]
        print(f"both conditions generated text : {ok}")
        print(f"answers identical              : {identical}")
        print()
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'text'}
                          for k, v in results.items()}, indent=2))
        return 0 if ok else 1

    except Exception:
        print("FAIL: unhandled exception during load/generate")
        traceback.print_exc()
        return 1
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
