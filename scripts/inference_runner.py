"""BitNet inference runner with robust subprocess management and reliability tracking.

Executes llama-cli / bitnet.cpp with non-interactive single-turn flags, captures
stdout/stderr separately, prevents stdin blocking, enforces timeouts, handles process
termination, and produces validated v1.1 experiment records.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.result_schema import build_record, validate_record, append_jsonl

DEFAULT_CLI = Path(
    os.environ.get(
        "BITNET_CLI",
        str(
            Path.home()
            / "source"
            / "repos"
            / "BitNet"
            / "build"
            / "bin"
            / "Release"
            / "llama-cli.exe"
        ),
    )
)

DEFAULT_MODEL = Path(
    os.environ.get(
        "BITNET_MODEL",
        str(
            Path.home()
            / "source"
            / "repos"
            / "BitNet"
            / "models"
            / "BitNet-b1.58-2B-4T-gguf"
            / "ggml-model-i2_s.gguf"
        ),
    )
)

DEFAULT_PROMPT = "Explain what a 1-bit language model is in simple terms."


def _decode(value: Any) -> str:
    """Safely decode process output to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _kill_process_tree(proc: subprocess.Popen[Any] | None) -> None:
    """Forcefully terminate a process and its child tree to avoid orphaned processes."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _format_error_message(exit_code: int, stderr: str) -> str:
    """Format non-zero exit codes with hex representation and trailing stderr context."""
    hex_repr = f" (0x{exit_code & 0xFFFFFFFF:08X})" if exit_code < 0 or exit_code > 255 else ""
    msg = f"CLI exited with code {exit_code}{hex_repr}"
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    if lines:
        msg += f": {' | '.join(lines[-3:])}"
    return msg


def parse_performance_metrics(stdout: str, stderr: str) -> dict[str, float | None]:
    """Parse generation and prompt speeds from backend performance logs.

    Extracts reported throughput without estimating or fabricating missing values.
    """
    combined = (stdout or "") + "\n" + (stderr or "")
    prompt_tps: float | None = None
    gen_tps: float | None = None

    # Format 1: [ Prompt: 23.3 t/s | Generation: 5.2 t/s ]
    m_prompt = re.search(r"Prompt:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", combined, re.IGNORECASE)
    if m_prompt:
        prompt_tps = float(m_prompt.group(1))

    m_gen = re.search(r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", combined, re.IGNORECASE)
    if m_gen:
        gen_tps = float(m_gen.group(1))

    # Format 2: llama.cpp standard timing lines
    # "prompt eval time =   ... ms / ... tokens ( ... ms per token,    81.00 tokens per second)"
    if prompt_tps is None:
        m_llama_prompt = re.search(
            r"prompt eval time\s*=.*?([0-9]+(?:\.[0-9]+)?)\s*tokens per second",
            combined,
            re.IGNORECASE,
        )
        if m_llama_prompt:
            prompt_tps = float(m_llama_prompt.group(1))

    # "eval time =   ... ms / ... tokens ( ... ms per token,    70.06 tokens per second)"
    if gen_tps is None:
        m_llama_gen = re.search(
            r"(?<!prompt )eval time\s*=.*?([0-9]+(?:\.[0-9]+)?)\s*tokens per second",
            combined,
            re.IGNORECASE,
        )
        if m_llama_gen:
            gen_tps = float(m_llama_gen.group(1))

    return {
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "peak_memory_mb": None,
    }


def _finalize_record(
    output_file: Path | str | None,
    config: dict[str, Any],
    execution_status: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    elapsed: float,
    prompt: str,
    error_message: str | None = None,
    benchmark: dict[str, Any] | None = None,
    data_source: str = "real",
) -> dict[str, Any]:
    """Build, validate, and optionally persist an experiment record."""
    perf = parse_performance_metrics(stdout, stderr)

    schema_record = build_record(
        run_id=str(uuid.uuid4()),
        config=config,
        execution_status=execution_status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        total_seconds=round(elapsed, 4) if elapsed is not None else 0.0,
        prompt=prompt,
        performance=perf,
        error_message=error_message,
        benchmark=benchmark,
        data_source=data_source,
        requested_tokens=config.get("requested_tokens"),
    )

    clean_output = (stdout or "").strip()

    # Legacy flat aliases for backward compatibility with existing readers
    legacy_aliases = {
        "cli": config.get("cli"),
        "model": config.get("model"),
        "threads": config.get("threads"),
        "ctx_size": config.get("ctx_size"),
        "requested_tokens": config.get("requested_tokens"),
        "timeout_seconds": config.get("timeout_seconds"),
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
        "output_characters": len(clean_output),
        "output_nonempty": bool(clean_output),
        "question_mark_only": bool(clean_output) and all(
            c.isspace() or c == "?" for c in clean_output
        ),
        "execution_status": execution_status,
        "exit_code": exit_code,
        "error": error_message,
    }

    schema_record.update(legacy_aliases)
    schema_record["output"] = {
        "stdout": stdout or "",
        "stderr": stderr or "",
    }

    problems = validate_record(schema_record)
    if problems:
        raise ValueError(f"Generated record failed schema validation: {problems}")

    if output_file:
        append_jsonl(str(output_file), schema_record)

    return schema_record


def build_command_args(
    model: str | Path,
    threads: int = 4,
    ctx_size: int = 512,
    predict: int = 32,
    temp: float = 0.0,
    seed: int = 42,
    prompt: str = DEFAULT_PROMPT,
) -> list[str]:
    """Construct verified CLI argument list for single-turn non-interactive inference."""
    return [
        "--model", str(model),
        "--threads", str(threads),
        "--ctx-size", str(ctx_size),
        "--n-predict", str(predict),
        "--temp", str(temp),
        "--seed", str(seed),
        "--prompt", prompt,
        "--single-turn",
        "--simple-io",
        "--no-display-prompt",
        "--color", "off",
        "--n-gpu-layers", "0",
    ]


def run_inference(
    cli: str | Path | None = None,
    model: str | Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    threads: int = 4,
    ctx_size: int = 512,
    predict: int = 32,
    timeout: int = 180,
    output_file: str | Path | None = None,
    temp: float = 0.0,
    seed: int = 42,
    benchmark: dict[str, Any] | None = None,
    data_source: str = "real",
    dry_run: bool = False,
    command_prefix: list[str] | None = None,
) -> dict[str, Any]:
    """Run BitNet/llama-cli inference and return a validated experiment record.

    Parameters
    ----------
    cli : str | Path | None
        Path to the llama-cli executable.
    model : str | Path | None
        Path to the GGUF model file.
    prompt : str
        Input prompt for inference.
    threads : int
        Number of CPU threads to allocate.
    ctx_size : int
        Model context size.
    predict : int
        Number of tokens to generate.
    timeout : int
        Process timeout ceiling in seconds.
    output_file : str | Path | None
        Optional path to append the JSONL record.
    temp : float
        Generation temperature (default: 0.0 for deterministic evaluation).
    seed : int
        Random seed (default: 42).
    benchmark : dict[str, Any] | None
        Optional metadata block for sweep tracking.
    data_source : str
        'real' for actual hardware/model runs, 'synthetic' for stubs/mocks.
    dry_run : bool
        If True, returns the formatted command without launching a process.
    command_prefix : list[str] | None
        Injectable command prefix for testing stubs without real binaries.
    """
    cli_path = Path(cli or DEFAULT_CLI).expanduser().resolve()
    model_path = Path(model or DEFAULT_MODEL).expanduser().resolve()

    config = {
        "cli": str(cli_path),
        "model": str(model_path),
        "threads": threads,
        "ctx_size": ctx_size,
        "prompt": prompt,
        "requested_tokens": predict,
        "timeout_seconds": timeout,
        "temp": temp,
        "seed": seed,
    }

    # Handle dry-run mode
    cmd_args = build_command_args(
        model=model_path,
        threads=threads,
        ctx_size=ctx_size,
        predict=predict,
        temp=temp,
        seed=seed,
        prompt=prompt,
    )
    full_command = (list(command_prefix) if command_prefix else [str(cli_path)]) + cmd_args

    if dry_run:
        return {
            "dry_run": True,
            "command": full_command,
            "config": config,
        }

    # Pre-flight missing file check (bypassed if custom test command_prefix is used)
    if not command_prefix:
        if not cli_path.is_file():
            return _finalize_record(
                output_file=output_file,
                config=config,
                execution_status="missing_file",
                exit_code=None,
                stdout="",
                stderr="",
                elapsed=0.0,
                prompt=prompt,
                error_message=f"CLI executable not found: {cli_path}",
                benchmark=benchmark,
                data_source=data_source,
            )
        if not model_path.is_file():
            return _finalize_record(
                output_file=output_file,
                config=config,
                execution_status="missing_file",
                exit_code=None,
                stdout="",
                stderr="",
                elapsed=0.0,
                prompt=prompt,
                error_message=f"Model file not found: {model_path}",
                benchmark=benchmark,
                data_source=data_source,
            )

    start = time.perf_counter()
    proc: subprocess.Popen[str] | None = None

    try:
        proc = subprocess.Popen(
            full_command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = proc.communicate(timeout=timeout)
        elapsed = time.perf_counter() - start

        exit_code = proc.returncode
        status = "success" if exit_code == 0 else "error"
        err_msg = None if exit_code == 0 else _format_error_message(exit_code, stderr)

        return _finalize_record(
            output_file=output_file,
            config=config,
            execution_status=status,
            exit_code=exit_code,
            stdout=stdout or "",
            stderr=stderr or "",
            elapsed=elapsed,
            prompt=prompt,
            error_message=err_msg,
            benchmark=benchmark,
            data_source=data_source,
        )

    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        _kill_process_tree(proc)
        raw_out, raw_err = (proc.communicate() if proc else (None, None))
        stdout = _decode(raw_out or exc.stdout)
        stderr = _decode(raw_err or exc.stderr)

        return _finalize_record(
            output_file=output_file,
            config=config,
            execution_status="timeout",
            exit_code=None,
            stdout=stdout,
            stderr=stderr,
            elapsed=elapsed,
            prompt=prompt,
            error_message=f"Inference exceeded timeout ceiling of {timeout} seconds",
            benchmark=benchmark,
            data_source=data_source,
        )

    except FileNotFoundError as exc:
        elapsed = time.perf_counter() - start
        return _finalize_record(
            output_file=output_file,
            config=config,
            execution_status="missing_file",
            exit_code=None,
            stdout="",
            stderr="",
            elapsed=elapsed,
            prompt=prompt,
            error_message=f"Command execution failed: {exc}",
            benchmark=benchmark,
            data_source=data_source,
        )

    except OSError as exc:
        elapsed = time.perf_counter() - start
        return _finalize_record(
            output_file=output_file,
            config=config,
            execution_status="error",
            exit_code=None,
            stdout="",
            stderr="",
            elapsed=elapsed,
            prompt=prompt,
            error_message=f"OS process error: {exc}",
            benchmark=benchmark,
            data_source=data_source,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BitNet inference and record reliability-aware results."
    )
    parser.add_argument("--cli", default=str(DEFAULT_CLI), help="Path to llama-cli.exe")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to GGUF model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt text")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--ctx-size", type=int, default=512, help="Context size")
    parser.add_argument("--predict", type=int, default=32, help="Tokens to generate")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--output", default="results/inference_runs.jsonl", help="JSONL output path")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")

    args = parser.parse_args()

    if args.threads < 1 or args.ctx_size < 1:
        parser.error("--threads and --ctx-size must be positive")
    if args.predict < 1 or args.timeout < 1:
        parser.error("--predict and --timeout must be positive")

    record = run_inference(
        cli=args.cli,
        model=args.model,
        prompt=args.prompt,
        threads=args.threads,
        ctx_size=args.ctx_size,
        predict=args.predict,
        timeout=args.timeout,
        output_file=args.output if not args.dry_run else None,
        temp=args.temp,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
