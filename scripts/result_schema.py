"""Shared experiment-record schema for the BitNet inference lab.

execution_status answers: did the process run?
reliability.status answers: does the output look sane?
These are deliberately separate.
"""
import json
import os
from datetime import datetime, timezone

from scripts.reliability_checker import check_output, FAIL

SCHEMA_VERSION = "1.0"
EXECUTION_STATUSES = {"success", "timeout", "error", "missing_file"}
RELIABILITY_STATUSES = {"PASS", "WARNING", "FAIL"}


def build_record(run_id, config, execution_status, exit_code, stdout, stderr,
                 total_seconds, prompt=None, performance=None, error_message=None):
    """Create one experiment record, including the reliability verdict."""
    perf = {"prompt_tps": None, "gen_tps": None, "peak_memory_mb": None}
    perf.update(performance or {})

    reliability = check_output(
        stdout,
        exit_code=exit_code,
        timed_out=(execution_status == "timeout"),
        prompt=prompt,
    )
    if execution_status == "missing_file":
        reliability["status"] = FAIL
        reliability["reasons"].append("FAIL:missing_file: executable or model file not found")

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,                      # model, threads, ctx, prompt, n_predict...
        "execution_status": execution_status,  # success|timeout|error|missing_file
        "exit_code": exit_code,
        "error_message": error_message,
        "timing": {"total_seconds": total_seconds},
        "performance": perf,
        "output": {"stdout": stdout, "stderr": stderr},
        "reliability": reliability,
    }


def validate_record(rec):
    """Return a list of problems (empty list = valid)."""
    problems = []
    required = ["schema_version", "run_id", "timestamp_utc", "config",
                "execution_status", "exit_code", "timing", "performance",
                "output", "reliability"]
    for key in required:
        if key not in rec:
            problems.append(f"missing key: {key}")
    if problems:
        return problems
    if rec["execution_status"] not in EXECUTION_STATUSES:
        problems.append(f"bad execution_status: {rec['execution_status']}")
    if rec["reliability"].get("status") not in RELIABILITY_STATUSES:
        problems.append("bad reliability.status")
    if not isinstance(rec["reliability"].get("reasons"), list):
        problems.append("reliability.reasons must be a list")
    if "total_seconds" not in rec["timing"]:
        problems.append("timing.total_seconds missing")
    return problems


def append_jsonl(path, record):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")