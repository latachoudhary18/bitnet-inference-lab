import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "raw" / "experiments.jsonl"
REPORT = ROOT / "results" / "validated" / "validation_report.json"

REQUIRED = [
    "schema_version", "run_id", "experiment_id", "timestamp_utc",
    "model_id", "model_path", "backend", "threads", "prompt_id",
    "prompt_text", "max_new_tokens", "temperature", "seed",
    "timeout_seconds", "command", "elapsed_seconds", "return_code",
    "status", "timed_out", "output_nonempty", "output_char_count",
    "stdout_file", "stderr_file"
]

issues = []
records = []

if not LOG.exists():
    issues.append({"code": "MISSING_LOG_FILE", "message": "Experiment log not found"})
else:
    for line_no, line in enumerate(LOG.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Record must be a JSON object")
            records.append((line_no, row))
        except (json.JSONDecodeError, ValueError) as exc:
            issues.append({"code": "INVALID_JSON", "line": line_no, "message": str(exc)})

seen = Counter(row.get("run_id") for _, row in records if row.get("run_id"))

for line_no, row in records:
    for field in REQUIRED:
        if field not in row:
            issues.append({"code": "MISSING_FIELD", "line": line_no, "field": field})

    if not isinstance(row.get("run_id"), str) or not row.get("run_id"):
        issues.append({"code": "INVALID_FIELD_TYPE", "line": line_no, "field": "run_id"})

    if isinstance(row.get("threads"), bool) or not isinstance(row.get("threads"), int) or row.get("threads", 0) < 1:
        issues.append({"code": "INVALID_CONFIGURATION", "line": line_no, "field": "threads"})

    elapsed = row.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed <= 0:
        issues.append({"code": "INVALID_LATENCY", "line": line_no})

    if row.get("status") == "success" and row.get("return_code") != 0:
        issues.append({"code": "PROCESS_FAILED", "line": line_no})

    if row.get("timed_out") is True or row.get("status") == "timeout":
        issues.append({"code": "PROCESS_TIMEOUT", "line": line_no})

    if row.get("status") == "launch_error":
        issues.append({"code": "LAUNCH_ERROR", "line": line_no})

    if row.get("status") == "success" and row.get("output_nonempty") is not True:
        issues.append({"code": "EMPTY_OUTPUT", "line": line_no})

    speed = row.get("generation_tokens_per_second")
    if row.get("status") == "success" and (
        isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed <= 0
    ):
        issues.append({"code": "INVALID_LATENCY", "line": line_no, "field": "generation_tokens_per_second"})

for run_id, count in seen.items():
    if count > 1:
        issues.append({"code": "DUPLICATE_RUN_ID", "run_id": run_id})

fail_codes = {"INVALID_JSON", "MISSING_FIELD", "INVALID_FIELD_TYPE",
              "DUPLICATE_RUN_ID", "INVALID_LATENCY", "PROCESS_FAILED",
              "PROCESS_TIMEOUT", "LAUNCH_ERROR", "EMPTY_OUTPUT",
              "MISSING_LOG_FILE", "INVALID_CONFIGURATION"}

status = "FAIL" if any(x["code"] in fail_codes for x in issues) else (
    "WARNING" if issues else "PASS"
)

report = {
    "schema_version": "1.0",
    "overall_status": status,
    "records_checked": len(records),
    "successful_runs": sum(r.get("status") == "success" for _, r in records),
    "issues_count": len(issues),
    "issues": issues
}

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"\nReport saved: {REPORT}")
