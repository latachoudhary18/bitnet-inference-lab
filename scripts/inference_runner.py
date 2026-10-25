import argparse
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from scripts.result_schema import build_record, append_jsonl

DEFAULT_CLI = (
    Path.home() / "source" / "repos" / "BitNet" / "build" / "bin"
    / "Release" / "llama-cli.exe"
)
DEFAULT_MODEL = (
    Path.home() / "source" / "repos" / "BitNet" / "models"
    / "BitNet-b1.58-2B-4T-gguf" / "ggml-model-i2_s.gguf"
)


def _decode(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value



def _finalize(output_file, legacy, execution_status, exit_code, stdout,
              stderr, elapsed, prompt, error_message=None):
    """Write one record with Phase 5 and legacy-compatible fields."""
    config = {
        "cli": legacy["cli"],
        "model": legacy["model"],
        "threads": legacy["threads"],
        "ctx_size": legacy["ctx_size"],
        "prompt": prompt,
        "requested_tokens": legacy["requested_tokens"],
        "timeout_seconds": legacy["timeout_seconds"],
    }

    # Metrics remain unavailable unless the backend reports them.
    schema_record = build_record(
        run_id=str(uuid.uuid4()),
        config=config,
        execution_status=execution_status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        total_seconds=elapsed,
        prompt=prompt,
        performance={
            "prompt_tps": None,
            "gen_tps": None,
            "peak_memory_mb": None,
        },
        error_message=error_message,
    )

    output = (stdout or "").strip()
    legacy.update({
        "status": {
            "success": "completed",
            "timeout": "timeout",
            "error": "execution_error",
            "missing_file": "configuration_error",
        }.get(execution_status, execution_status),
        "return_code": exit_code,
        "elapsed_seconds": round(elapsed, 3) if elapsed is not None else None,
        "stdout": stdout or "",
        "stderr": stderr or "",
        "combined_log": (stdout or "") + "\n" + (stderr or ""),
        "output_characters": len(output),
        "output_nonempty": bool(output),
        "question_mark_only": bool(output) and all(
            c.isspace() or c == "?" for c in output
        ),
        "execution_status": execution_status,
        "exit_code": exit_code,
        "error": error_message,
    })

    # Keep the schema fields and nested output while preserving legacy aliases.
    schema_record.update(legacy)
    schema_record["output"] = {
        "stdout": stdout or "",
        "stderr": stderr or "",
    }

    append_jsonl(str(output_file), schema_record)
    return schema_record

def run_inference(cli, model, prompt, threads, ctx_size,
                  predict, timeout, output_file):
    cli = Path(cli).expanduser().resolve()
    model = Path(model).expanduser().resolve()
    output_file = Path(output_file)
    legacy = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli": str(cli), "model": str(model), "threads": threads,
        "ctx_size": ctx_size, "requested_tokens": predict,
        "timeout_seconds": timeout,
    }

    if not cli.is_file():
        return _finalize(output_file, legacy, "missing_file", None, "", "",
                         0.0, prompt, f"CLI not found: {cli}")
    if not model.is_file():
        return _finalize(output_file, legacy, "missing_file", None, "", "",
                         0.0, prompt, f"Model not found: {model}")

    command = [
        str(cli), "--model", str(model), "--threads", str(threads),
        "--ctx-size", str(ctx_size), "--n-predict", str(predict),
        "--prompt", prompt, "--no-conversation", "--simple-io",
        "--no-display-prompt", "--color", "off", "--n-gpu-layers", "0",
    ]
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            timeout=timeout, check=False,
        )
        elapsed = time.perf_counter() - start
        status = "success" if result.returncode == 0 else "error"
        return _finalize(
            output_file, legacy, status, result.returncode,
            result.stdout or "", result.stderr or "", elapsed, prompt,
            None if result.returncode == 0 else
            f"CLI exited with code {result.returncode}",
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout, stderr = _decode(exc.stdout), _decode(exc.stderr)
        return _finalize(
            output_file, legacy, "timeout", None, stdout, stderr, elapsed,
            prompt, f"Inference exceeded {timeout} seconds",
        )
    except OSError as exc:
        elapsed = time.perf_counter() - start
        return _finalize(
            output_file, legacy, "error", None, "", "", elapsed, prompt,
            str(exc),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run BitNet inference and record reliability-aware results."
    )
    parser.add_argument("--cli", default=str(DEFAULT_CLI))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--prompt", default="Explain what a 1-bit language model is in simple terms.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default="results/inference_runs.jsonl")
    args = parser.parse_args()
    if args.threads < 1 or args.ctx_size < 1:
        parser.error("--threads and --ctx-size must be positive")
    if args.predict < 1 or args.timeout < 1:
        parser.error("--predict and --timeout must be positive")
    record = run_inference(
        args.cli, args.model, args.prompt, args.threads, args.ctx_size,
        args.predict, args.timeout, args.output,
    )
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
