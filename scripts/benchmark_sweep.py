
"""
Benchmark sweep for BitNet inference.

Runs a warmup and repeated measurements for each
(thread_count, context_size) configuration.

This script reuses scripts.inference_runner.run_inference().
It does not estimate unavailable performance metrics.
"""

import argparse
import json
import statistics
import tempfile
from pathlib import Path

from scripts.inference_runner import (
    DEFAULT_CLI,
    DEFAULT_MODEL,
    run_inference,
)


DEFAULT_PROMPT = (
    "Explain what a 1-bit language model is in simple terms."
)


def append_jsonl(path, record):
    """Append one JSON record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one(cli, model, prompt, threads, ctx_size,
            predict, timeout, is_warmup, output_file,                        data_source="real_cli_attempt",):
    """Run one inference and return its result with sweep metadata."""

    # The existing runner writes its own record.
    # Use a temporary file so we can add sweep metadata
    # before appending the record to the main sweep log.
    with tempfile.TemporaryDirectory() as temp_dir:
        temporary_output = Path(temp_dir) / "single_run.jsonl"

        record = run_inference(
            cli=cli,
            model=model,
            prompt=prompt,
            threads=threads,
            ctx_size=ctx_size,
            predict=predict,
            timeout=timeout,
            output_file=str(temporary_output),
        )

    record["benchmark"] = {
        "threads": threads,
        "ctx_size": ctx_size,
        "is_warmup": is_warmup,
        "data_source": data_source,
        "is_synthetic": data_source == "synthetic",
    }

    append_jsonl(output_file, record)

    label = "WARMUP" if is_warmup else "MEASURED"
    status = record.get("execution_status", "unknown")

    print(
        f"{label:8} threads={threads:<2} "
        f"ctx={ctx_size:<3} status={status}"
    )

    return record


def calculate_statistics(records):
    """Calculate latency statistics for measured successful runs."""

    values = []

    for record in records:
        benchmark = record.get("benchmark", {})

        if benchmark.get("is_warmup", False):
            continue

        if record.get("execution_status") != "success":
            continue

        if record.get("exit_code") != 0:
            continue

        timing = record.get("timing", {})
        elapsed = timing.get("total_seconds")

        if isinstance(elapsed, (int, float)):
            values.append(elapsed)

    if not values:
        return {
            "measurement_count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "std_dev_seconds": None,
        }

    return {
        "measurement_count": len(values),
        "mean_seconds": round(statistics.mean(values), 4),
        "median_seconds": round(statistics.median(values), 4),
        "std_dev_seconds": (
            round(statistics.stdev(values), 4)
            if len(values) > 1 else None
        ),
    }


def build_summary(records):
    """Group measured records by thread and context configuration."""

    configurations = {}

    for record in records:
        benchmark = record.get("benchmark", {})

        if benchmark.get("is_warmup", False):
            continue

        key = (
            benchmark.get("threads"),
            benchmark.get("ctx_size"),
        )

        configurations.setdefault(key, []).append(record)

    results = []

    for (threads, ctx_size), group in sorted(configurations.items()):
        successful = [
            row for row in group
            if row.get("execution_status") == "success"
            and row.get("exit_code") == 0
        ]

        reliable = [
            row for row in successful
            if isinstance(row.get("reliability"), dict)
            and row["reliability"].get("status") == "PASS"
        ]

        latency = calculate_statistics(group)

        results.append({
            "threads": threads,
            "ctx_size": ctx_size,
            "total_measured_runs": len(group),
            "successful_runs": len(successful),
            "reliability_passes": len(reliable),
            "latency": latency,
            "mean_prompt_tps": None,
            "mean_generation_tps": None,
            "peak_memory_mb": None,
            "data_source": "real_cli_attempt",
            "note": (
                "Unavailable performance metrics remain null. "
                "No throughput or memory values are estimated."
            ),
        })

    return {
        "total_configurations": len(results),
        "configurations": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run repeated BitNet inference benchmarks."
    )

    parser.add_argument("--cli", default=str(DEFAULT_CLI))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--threads", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--predict", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--output",
        default="results/sweep_runs.jsonl",
    )
    parser.add_argument(
        "--summary",
        default="results/sweep_summary.json",
    )

    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    if args.predict < 1 or args.timeout < 1:
        parser.error("--predict and --timeout must be positive")

    if any(value < 1 for value in args.threads + args.contexts):
        parser.error("Thread counts and context sizes must be positive")

    output_file = Path(args.output)
    summary_file = Path(args.summary)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Start a fresh sweep log to avoid mixing separate sweep sessions.
    if output_file.exists():
        output_file.unlink()

    all_records = []

    total_configurations = len(args.threads) * len(args.contexts)

    print(f"Configurations: {total_configurations}")
    print(f"Warmups: {total_configurations}")
    print(f"Measured repetitions: {total_configurations * args.repetitions}")
    print("Starting benchmark sweep...\n")

    for threads in args.threads:
        for ctx_size in args.contexts:

            print(
                f"\nConfiguration: threads={threads}, "
                f"context={ctx_size}"
            )

            # One warmup run per configuration.
            warmup = run_one(
                args.cli, args.model, args.prompt,
                threads, ctx_size, args.predict,
                args.timeout, True, output_file,
            )
            all_records.append(warmup)

            # Repeated measured runs.
            for repetition in range(1, args.repetitions + 1):
                print(f"Repetition {repetition}/{args.repetitions}")

                record = run_one(
                    args.cli, args.model, args.prompt,
                    threads, ctx_size, args.predict,
                    args.timeout, False, output_file,
                )
                all_records.append(record)

    summary = build_summary(all_records)

    summary["prompt"] = args.prompt
    summary["repetitions_per_configuration"] = args.repetitions
    summary["warmup_runs_per_configuration"] = 1
    summary["data_source"] = "real_cli_attempt"

    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with summary_file.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nBenchmark sweep finished.")
    print(f"Run records: {output_file}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
