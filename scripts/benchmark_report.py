
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load valid JSON objects from a JSON Lines benchmark log."""
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
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number}"
                )
            records.append(record)

    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize process outcomes, heuristic checks, and measured speeds."""
    successful = [
        row for row in records
        if row.get("exit_code") == 0 and not row.get("timed_out", False)
    ]
    timed_out = [row for row in records if row.get("timed_out", False)]
    process_failures = [
        row for row in records
        if not row.get("timed_out", False)
        and row.get("exit_code") not in (0, None)
    ]

    assessments = [
        row.get("assessment", {}).get("status")
        for row in records
        if isinstance(row.get("assessment"), dict)
    ]
    reliability_passes = assessments.count("pass_heuristic")
    reliability_failures = assessments.count("fail")

    eligible = []
    for row in records:
        assessment = row.get("assessment", {})
        speed = row.get("generation_tokens_per_second")
        if (
            row.get("exit_code") == 0
            and not row.get("timed_out", False)
            and isinstance(assessment, dict)
            and assessment.get("status") == "pass_heuristic"
            and isinstance(speed, (int, float))
        ):
            eligible.append(row)

    fastest = max(
        eligible,
        key=lambda row: row["generation_tokens_per_second"],
        default=None,
    )

    prompt_speeds = [
        row["prompt_tokens_per_second"]
        for row in records
        if isinstance(row.get("prompt_tokens_per_second"), (int, float))
    ]
    generation_speeds = [
        row["generation_tokens_per_second"]
        for row in records
        if isinstance(row.get("generation_tokens_per_second"), (int, float))
    ]

    return {
        "total_runs": len(records),
        "successful_processes": len(successful),
        "timed_out_runs": len(timed_out),
        "process_failures": len(process_failures),
        "reliability_passes": reliability_passes,
        "reliability_failures": reliability_failures,
        "runs_without_assessment": len(records) - len(assessments),
        "prompt_speed_measurements": len(prompt_speeds),
        "generation_speed_measurements": len(generation_speeds),
        "fastest_eligible_run": fastest,
        "selection_note": (
            "Fastest eligible run selected using the heuristic gate."
            if fastest is not None
            else "No eligible configuration: no run passed the heuristic gate "
                 "with a successful process and a measured generation speed."
        ),
        "limitation": (
            "Heuristic checks do not establish semantic correctness. "
            "Captured output and speed measurements require interpretation."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a BitNet JSON Lines benchmark log."
    )
    parser.add_argument(
        "--input", default="results/benchmark_log.jsonl",
        help="Path to the JSONL benchmark log",
    )
    parser.add_argument(
        "--output", default="results/benchmark_summary.json",
        help="Path to write the summary JSON",
    )
    args = parser.parse_args()

    try:
        records = load_records(Path(args.input))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    summary = summarize(records)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
