from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def assess_output(text: str) -> dict[str, Any]:
    """Heuristic output checks. These flag suspicious output; they do not prove correctness."""
    cleaned = text.strip()
    flags: list[str] = []
    if not cleaned:
        flags.append("empty_output")
        return {"status": "fail", "flags": flags, "character_count": 0}

    visible = [c for c in cleaned if not c.isspace()]
    if visible:
        suspicious = sum(c in "?" for c in visible) / len(visible)
        if suspicious >= 0.30:
            flags.append("question_mark_heavy_output")

    tokens = re.findall(r"\w+|[^\w\s]", cleaned.lower())
    if len(tokens) >= 8:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.25:
            flags.append("high_repetition")

    if len(cleaned) < 8:
        flags.append("very_short_output")

    return {
        "status": "fail" if flags else "pass_heuristic",
        "flags": flags,
        "character_count": len(cleaned),
        "token_count_estimate": len(tokens),
    }


def parse_speed(text: str, label: str) -> float | None:
    match = re.search(
        rf"{re.escape(label)}:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", text, re.IGNORECASE
    )
    return float(match.group(1)) if match else None


def run_inference(
    cli: str,
    model: str,
    prompt: str,
    threads: int = 4,
    context_size: int = 128,
    predict: int = 16,
    timeout: int = 300,
) -> dict[str, Any]:
    command = [
        cli, "--model", model, "--threads", str(threads),
        "--ctx-size", str(context_size), "--n-gpu-layers", "0",
        "--n-predict", str(predict), "--prompt", prompt,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
        elapsed = time.perf_counter() - started
        combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
        # Keep the full output in the record so the run is auditable.
        assessment = assess_output(combined)
        return {
            "model": str(model),
            "cli": str(cli),
            "prompt": prompt,
            "settings": {
                "threads": threads, "context_size": context_size,
                "n_predict": predict, "gpu_layers": 0,
            },
            "elapsed_seconds": round(elapsed, 3),
            "exit_code": completed.returncode,
            "prompt_tokens_per_second": parse_speed(combined, "Prompt"),
            "generation_tokens_per_second": parse_speed(combined, "Generation"),
            "assessment": assessment,
            "raw_output": combined,
            "command": command,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        partial = (exc.stdout or "") + "\n" + (exc.stderr or "")
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return {
            "model": str(model), "cli": str(cli), "prompt": prompt,
            "settings": {
                "threads": threads, "context_size": context_size,
                "n_predict": predict, "gpu_layers": 0,
            },
            "elapsed_seconds": round(elapsed, 3),
            "exit_code": None,
            "prompt_tokens_per_second": None,
            "generation_tokens_per_second": None,
            "assessment": assess_output(partial),
            "raw_output": partial,
            "command": command,
            "timed_out": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a BitNet inference check and save a JSON record.")
    parser.add_argument("--cli", required=True, help="Path to llama-cli.exe")
    parser.add_argument("--model", required=True, help="Path to a GGUF model")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=128)
    parser.add_argument("--n-predict", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", default="results/benchmark_log.jsonl")
    args = parser.parse_args()

    result = run_inference(
        args.cli, args.model, args.prompt, args.threads,
        args.ctx_size, args.n_predict, args.timeout,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(json.dumps({k: v for k, v in result.items() if k != "raw_output"}, indent=2))
    print(f"\nFull record appended to: {output}")
    return 0 if result["exit_code"] == 0 and not result["timed_out"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
