
import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_CLI = (
    Path.home()
    / "source"
    / "repos"
    / "BitNet"
    / "build"
    / "bin"
    / "Release"
    / "llama-cli.exe"
)

DEFAULT_MODEL = (
    Path.home()
    / "source"
    / "repos"
    / "BitNet"
    / "models"
    / "BitNet-b1.58-2B-4T-gguf"
    / "ggml-model-i2_s.gguf"
)


def run_inference(cli, model, prompt, threads, ctx_size,
                  predict, timeout, output_file):
    cli = Path(cli).expanduser().resolve()
    model = Path(model).expanduser().resolve()
    output_file = Path(output_file)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli": str(cli),
        "model": str(model),
        "threads": threads,
        "ctx_size": ctx_size,
        "requested_tokens": predict,
        "timeout_seconds": timeout,
    }

    if not cli.is_file():
        record.update(status="configuration_error",
                      error=f"CLI not found: {cli}")
        save_result(output_file, record)
        return record

    if not model.is_file():
        record.update(status="configuration_error",
                      error=f"Model not found: {model}")
        save_result(output_file, record)
        return record

    command = [
        str(cli),
        "--model", str(model),
        "--threads", str(threads),
        "--ctx-size", str(ctx_size),
        "--n-predict", str(predict),
        "--prompt", prompt,
        "--no-conversation",
        "--simple-io",
        "--no-display-prompt",
        "--color", "off",
        "--n-gpu-layers", "0",
    ]

    start = time.perf_counter()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            check=False,
        )
        elapsed = time.perf_counter() - start

        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        output = (result.stdout or "").strip()

        record.update({
            "status": "completed" if result.returncode == 0
                      else "process_error",
            "return_code": result.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "output_characters": len(output),
            "output_nonempty": bool(output),
            "question_mark_only": bool(output)
                and all(c.isspace() or c == "?" for c in output),
            "combined_log": combined,
        })

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        record.update({
            "status": "timeout",
            "elapsed_seconds": round(elapsed, 3),
            "stdout": str(exc.stdout or ""),
            "stderr": str(exc.stderr or ""),
            "error": f"Inference exceeded {timeout} seconds",
        })

    except OSError as exc:
        record.update({
            "status": "execution_error",
            "error": str(exc),
        })

    save_result(output_file, record)
    return record


def save_result(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run BitNet inference and record diagnostic results."
    )
    parser.add_argument("--cli", default=str(DEFAULT_CLI))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument(
        "--prompt",
        default="Explain what a 1-bit language model is in simple terms."
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx-size", type=int, default=512)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--output",
        default="results/inference_runs.jsonl"
    )
    args = parser.parse_args()

    if args.threads < 1 or args.ctx_size < 1:
        parser.error("--threads and --ctx-size must be positive")
    if args.predict < 1 or args.timeout < 1:
        parser.error("--predict and --timeout must be positive")

    record = run_inference(
        args.cli,
        args.model,
        args.prompt,
        args.threads,
        args.ctx_size,
        args.predict,
        args.timeout,
        args.output,
    )

    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
