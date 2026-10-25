from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL records and reject malformed lines with a useful line number."""
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            records.append(row)
    return records


def _status(row: dict[str, Any]) -> str:
    """Support both inference_runner and the older inference_check log schema."""
    if row.get("status"):
        return str(row["status"])
    if row.get("timed_out", False):
        return "timeout"
    code = row.get("exit_code", row.get("return_code"))
    if code == 0:
        return "completed"
    if code is not None:
        return "process_error"
    return "unknown"


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [_status(row) for row in records]
    completed = [row for row, status in zip(records, statuses) if status == "completed"]
    timed_out = [row for row, status in zip(records, statuses) if status == "timeout"]
    process_failures = [
        row for row, status in zip(records, statuses)
        if status in {"process_error", "execution_error", "configuration_error"}
    ]

    assessments = [
        row.get("assessment", {}).get("status")
        for row in records if isinstance(row.get("assessment"), dict)
    ]
    reliability_passes = assessments.count("pass_heuristic")
    reliability_failures = assessments.count("fail")

    eligible = []
    for row in records:
        assessment = row.get("assessment", {})
        speed = row.get("generation_tokens_per_second")
        if (
            _status(row) == "completed"
            and isinstance(assessment, dict)
            and assessment.get("status") == "pass_heuristic"
            and isinstance(speed, (int, float))
        ):
            eligible.append(row)

    fastest = max(eligible, key=lambda row: row["generation_tokens_per_second"], default=None)

    def numeric_values(key: str) -> list[float]:
        return [
            float(row[key]) for row in records
            if isinstance(row.get(key), (int, float))
        ]

    elapsed = numeric_values("elapsed_seconds")
    prompt_speeds = numeric_values("prompt_tokens_per_second")
    generation_speeds = numeric_values("generation_tokens_per_second")
    load_times = numeric_values("model_load_ms")

    return {
        "total_runs": len(records),
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "successful_processes": len(completed),
        "timed_out_runs": len(timed_out),
        "process_failures": len(process_failures),
        "reliability_passes": reliability_passes,
        "reliability_failures": reliability_failures,
        "runs_without_assessment": len(records) - len(assessments),
        "elapsed_seconds": {
            "measurement_count": len(elapsed),
            "mean": round(mean(elapsed), 3) if elapsed else None,
        },
        "model_load_ms": {
            "measurement_count": len(load_times),
            "mean": round(mean(load_times), 3) if load_times else None,
        },
        "prompt_tokens_per_second": {
            "measurement_count": len(prompt_speeds),
            "mean": round(mean(prompt_speeds), 3) if prompt_speeds else None,
        },
        "generation_tokens_per_second": {
            "measurement_count": len(generation_speeds),
            "mean": round(mean(generation_speeds), 3) if generation_speeds else None,
        },
        "fastest_eligible_run": fastest,
        "selection_note": (
            "Fastest run selected only from completed runs that pass the heuristic "
            "gate and contain an explicit generation-speed measurement."
            if fastest is not None else
            "No eligible configuration: require a completed process, a passing "
            "heuristic assessment, and an explicit generation-speed measurement."
        ),
        "limitations": [
            "A completed process and heuristic pass do not establish semantic correctness.",
            "Token throughput is reported only when parsed from explicit backend performance logs; it is never estimated from output characters.",
            "CPU and memory usage are not measured by this runner.",
            "Missing metrics mean the backend did not expose a recognized measurement.",
            "Legacy log schemas are normalized where possible; absent fields remain unavailable.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize BitNet JSON Lines benchmark records.")
    parser.add_argument("--input", default="results/inference_runs.jsonl",
                        help="Path to the inference runner JSONL log")
    parser.add_argument("--output", default="results/benchmark_summary.json",
                        help="Path to write the summary JSON")
    args = parser.parse_args()
    try:
        records = load_records(Path(args.input))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    summary = summarize(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
