from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            records.append(record)
    return records


def _execution_status(row):
    if row.get("execution_status"):
        return row["execution_status"]
    status = row.get("status")
    return {
        "completed": "success", "process_error": "error",
        "execution_error": "error", "configuration_error": "missing_file",
    }.get(status, status)


def _reliability_status(row):
    reliability = row.get("reliability")
    if isinstance(reliability, dict) and reliability.get("status"):
        return reliability["status"]
    assessment = row.get("assessment")
    if isinstance(assessment, dict):
        old = assessment.get("status")
        return {"pass_heuristic": "PASS", "fail": "FAIL"}.get(old, old)
    return None


def _elapsed(row):
    timing = row.get("timing")
    if isinstance(timing, dict) and isinstance(timing.get("total_seconds"), (int, float)):
        return timing["total_seconds"]
    value = row.get("elapsed_seconds")
    return value if isinstance(value, (int, float)) else None


def _metric(row, schema_key, legacy_key):
    performance = row.get("performance")
    if isinstance(performance, dict):
        value = performance.get(schema_key)
        if isinstance(value, (int, float)):
            return value
    value = row.get(legacy_key)
    return value if isinstance(value, (int, float)) else None


def _stats(values):
    return {
        "measurement_count": len(values),
        "mean": round(sum(values) / len(values), 4) if values else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = [_execution_status(r) for r in records]
    successful = [r for r in records if _execution_status(r) == "success"
                  and r.get("exit_code", r.get("return_code")) == 0]
    timed_out = [r for r in records if _execution_status(r) == "timeout"]
    process_failures = [r for r in records if _execution_status(r) == "error"]
    reliability = [_reliability_status(r) for r in records]
    passes = reliability.count("PASS") + reliability.count("pass_heuristic")
    warnings = reliability.count("WARNING") + reliability.count("warning")
    failures = reliability.count("FAIL") + reliability.count("fail")

    prompt_speeds = [v for r in records if (v := _metric(r, "prompt_tps", "prompt_tokens_per_second")) is not None]
    generation_speeds = [v for r in records if (v := _metric(r, "gen_tps", "generation_tokens_per_second")) is not None]
    elapsed_values = [v for r in records if (v := _elapsed(r)) is not None]
    model_load = [r["model_load_ms"] for r in records if isinstance(r.get("model_load_ms"), (int, float))]

    eligible = []
    for row in records:
        status = _execution_status(row)
        exit_code = row.get("exit_code", row.get("return_code"))
        speed = _metric(row, "gen_tps", "generation_tokens_per_second")
        if status == "success" and exit_code == 0 and _reliability_status(row) == "PASS" and speed is not None:
            eligible.append(row)
    fastest = max(eligible, key=lambda r: _metric(r, "gen_tps", "generation_tokens_per_second"), default=None)

    return {
        "total_runs": len(records),
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
        "successful_processes": len(successful),
        "timed_out_runs": len(timed_out),
        "process_failures": len(process_failures),
        "reliability_passes": passes,
        "reliability_warnings": warnings,
        "reliability_failures": failures,
        "runs_without_assessment": sum(s is None for s in reliability),
        "elapsed_seconds": _stats(elapsed_values),
        "model_load_ms": _stats(model_load),
        "prompt_tokens_per_second": _stats(prompt_speeds),
        "generation_tokens_per_second": _stats(generation_speeds),
        "fastest_eligible_run": fastest,
        "selection_note": (
            "Fastest eligible run selected only from successful processes with PASS reliability and explicit generation speed."
            if fastest is not None else
            "No eligible configuration: require a completed process, a PASS reliability assessment, and an explicit generation-speed measurement."
        ),
        "limitations": [
            "Heuristic checks do not establish semantic correctness.",
            "Token throughput is included only when explicitly measured by the backend.",
            "CPU and memory usage are not measured by this runner.",
            "Missing metrics remain unavailable rather than being estimated.",
            "Legacy records are supported; absent fields remain unavailable.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize reliability-aware BitNet benchmark JSONL.")
    parser.add_argument("--input", default="results/inference_runs.jsonl")
    parser.add_argument("--output", default="results/benchmark_summary.json")
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
