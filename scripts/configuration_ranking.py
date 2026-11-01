"""Reliability-aware ranking of BitNet benchmark configurations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _number(value):
    """Return True for valid numeric measurements, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _rank_records(records):
    """Rank individual Schema v1.1 run records."""

    groups = defaultdict(list)
    excluded_runs = []
    ignored_legacy_runs = 0

    for record in records:
        if not isinstance(record, dict):
            excluded_runs.append({"reasons": ["Invalid record"]})
            continue

        benchmark = record.get("benchmark") or {}
        config = record.get("config") or {}

        threads = benchmark.get("threads", config.get("threads"))
        ctx_size = benchmark.get("ctx_size", config.get("ctx_size"))
        key = (threads, ctx_size)

        reasons = []
        
        if record.get("schema_version") != "1.1":
            ignored_legacy_runs += 1
            continue 

        if benchmark.get("is_warmup") is not False:
            reasons.append("Warmup or unspecified warmup status")

        if record.get("execution_status") != "success":
            reasons.append("Execution was not successful")

        if record.get("exit_code") != 0:
            reasons.append("Exit code was not zero")

        reliability = record.get("reliability")
        if not isinstance(reliability, dict) or reliability.get("status") != "PASS":
            reasons.append("Reliability status was not PASS")

        if reasons:
            excluded_runs.append({
                "threads": threads,
                "ctx_size": ctx_size,
                "reasons": reasons,
            })
            continue

        groups[key].append(record)

    ranked = []
    excluded_configurations = []

    for (threads, ctx_size), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0] if item[0][0] is not None else 0,
            item[0][1] if item[0][1] is not None else 0,
        ),
    ):
        throughputs = [
            row["performance"]["gen_tps"]
            for row in group
            if isinstance(row.get("performance"), dict)
            and _number(row["performance"].get("gen_tps"))
            and row["performance"]["gen_tps"] >= 0
        ]

        latencies = [
            row["timing"]["total_seconds"]
            for row in group
            if isinstance(row.get("timing"), dict)
            and _number(row["timing"].get("total_seconds"))
            and row["timing"]["total_seconds"] >= 0
        ]

        synthetic = any(
            row.get("data_source") == "synthetic"
            for row in group
        )

        result = {
            "threads": threads,
            "ctx_size": ctx_size,
            "eligible_runs": len(group),
            "data_source": "synthetic" if synthetic else "real",
            "label": "[SYNTHETIC / STUB]" if synthetic else "REAL",
            "mean_generation_tps": (
                round(mean(throughputs), 4) if throughputs else None
            ),
            "mean_total_seconds": (
                round(mean(latencies), 4) if latencies else None
            ),
        }

        if throughputs:
            result["ranking_metric"] = "mean_generation_tps"
            ranked.append(result)
        elif latencies:
            result["ranking_metric"] = "mean_total_seconds"
            ranked.append(result)
        else:
            excluded_configurations.append({
                "threads": threads,
                "ctx_size": ctx_size,
                "reasons": [
                    "Neither generation throughput nor elapsed time is available"
                ],
            })

    # Throughput is the primary metric. Latency is the fallback
    # only when no eligible configuration has measured throughput.
    if ranked and any(
        item["ranking_metric"] == "mean_generation_tps"
        for item in ranked
    ):
        ranked = [
            item for item in ranked
            if item["ranking_metric"] == "mean_generation_tps"
        ]
        ranked.sort(
            key=lambda item: item["mean_generation_tps"],
            reverse=True,
        )
        ranking_metric = "mean_generation_tps"
    else:
        ranked.sort(
            key=lambda item: item["mean_total_seconds"]
        )
        ranking_metric = "mean_total_seconds"

    for position, item in enumerate(ranked, start=1):
        item["rank"] = position

    return {
        "report_type": "schema_v1.1_run_ranking",
        "total_runs": len(records) - ignored_legacy_runs,
        "ignored_legacy_runs": ignored_legacy_runs,
        "eligible_runs": sum(len(group) for group in groups.values()),
        "excluded_runs": len(excluded_runs),
        "eligible_configurations": len(ranked),
        "excluded_configurations": len(excluded_configurations),
        "ranking_metric": ranking_metric if ranked else None,
        "winner": ranked[0] if ranked else None,
        "ranked_configurations": ranked,
        "excluded_runs_detail": excluded_runs,
        "excluded_configurations_detail": excluded_configurations,
        "selection_note": (
            "Only successful, zero-exit-code, reliability-PASS, "
            "non-warmup runs are eligible. Synthetic results are labelled."
            if ranked else
            "No eligible winner: no qualifying configuration has a "
            "measured throughput or elapsed-time value."
        ),
    }


def _rank_legacy_summary(summary):
    """Preserve compatibility with existing aggregate summary tests."""

    configurations = summary.get("configurations")

    if configurations is None:
        successful = summary.get("successful_processes", 0)
        passes = summary.get("reliability_passes", 0)
        throughput = summary.get("generation_tokens_per_second", {})
        measured_speed = (
            throughput.get("mean")
            if isinstance(throughput, dict)
            else None
        )

        eligible = (
            successful > 0
            and passes > 0
            and _number(measured_speed)
        )

        return {
            "report_type": "single_run_summary",
            "total_runs": summary.get("total_runs", 0),
            "successful_runs": successful,
            "reliability_passes": passes,
            "mean_generation_tps": measured_speed,
            "winner": None,
            "eligible_configurations": 1 if eligible else 0,
            "selection_note": (
                "A configuration cannot be identified from this "
                "single-run summary."
                if eligible else
                "No eligible winner: successful execution, reliability "
                "passes and measured generation throughput are required."
            ),
        }

    eligible = []
    excluded = []

    for config in configurations:
        reasons = []
        successful = config.get("successful_runs", 0)
        passes = config.get("reliability_passes", 0)
        throughput = config.get("mean_generation_tps")

        if successful <= 0:
            reasons.append("No successful inference runs")
        if passes <= 0:
            reasons.append("No reliability-passing runs")
        if not _number(throughput):
            reasons.append("Generation throughput unavailable")

        if reasons:
            excluded.append({
                "threads": config.get("threads"),
                "ctx_size": config.get("ctx_size"),
                "reasons": reasons,
            })
        else:
            eligible.append(dict(config))

    ranked = sorted(
        eligible,
        key=lambda item: item["mean_generation_tps"],
        reverse=True,
    )

    for position, config in enumerate(ranked, start=1):
        config["rank"] = position

    return {
        "report_type": "sweep_summary",
        "total_configurations": len(configurations),
        "eligible_configurations": len(ranked),
        "excluded_configurations": len(excluded),
        "ranking_metric": "mean_generation_tps",
        "winner": ranked[0] if ranked else None,
        "ranked_configurations": ranked,
        "excluded_configurations_detail": excluded,
        "selection_note": (
            "Legacy aggregate summary: ranking uses its existing "
            "successful-run and reliability-pass counts."
            if ranked else
            "No winner can be selected from the available aggregate data."
        ),
    }


def rank_configurations(summary: dict[str, Any]) -> dict[str, Any]:
    """Rank Schema v1.1 records or support legacy JSON summaries."""

    if isinstance(summary, list):
        return _rank_records(summary)

    if not isinstance(summary, dict):
        raise ValueError("Input must be a JSON object or a list of records.")

    if "records" in summary and isinstance(summary["records"], list):
        return _rank_records(summary["records"])

    return _rank_legacy_summary(summary)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank reliable BitNet benchmark configurations."
    )
    parser.add_argument(
        "--input",
        default="results/sweep_runs.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/configuration_ranking.json",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    try:
        content = input_path.read_text(encoding="utf-8")

        if input_path.suffix.lower() == ".jsonl":
            records = [
                json.loads(line)
                for line in content.splitlines()
                if line.strip()
            ]
            ranking = rank_configurations(records)
        else:
            summary = json.loads(content)
            ranking = rank_configurations(summary)

    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(f"Could not process input: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(ranking, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(ranking, indent=2, ensure_ascii=False))
    print(f"\nRanking saved to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())