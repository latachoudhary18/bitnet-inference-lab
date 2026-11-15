"""Validate BitNet experiment records in flat v1.0 and nested v1.1 formats.

This module validates experiment-record structure and execution metadata.
Generated-text quality analysis remains in scripts/reliability_checker.py.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

SUPPORTED_VERSIONS = {"1.0", "1.1"}
EXECUTION_STATUSES = {"success", "timeout", "error", "missing_file"}
RELIABILITY_STATUSES = {"PASS", "WARNING", "FAIL"}
DATA_SOURCES = {"real", "synthetic"}


def _issue(severity: str, code: str, message: str, run_id: str | None = None) -> dict[str, Any]:
    result = {"severity": severity, "code": code, "message": message}
    if run_id is not None:
        result["run_id"] = run_id
    return result


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _timestamp_ok(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.tzinfo is not None and dt.utcoffset() is not None
    except (ValueError, TypeError, OverflowError):
        return False


def _validate_v10(record: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    required = (
        "experiment_id", "model_id", "backend", "threads", "prompt_id",
        "prompt_text", "max_new_tokens", "temperature", "timeout_seconds",
        "command", "elapsed_seconds", "return_code", "status", "timed_out",
        "output_nonempty", "output_char_count",
    )
    for field in required:
        if field not in record:
            issues.append(_issue("FAIL", "MISSING_REQUIRED_FIELD", f"Missing required field: {field}"))

    for field in ("threads", "max_new_tokens", "timeout_seconds", "output_char_count"):
        value = record.get(field)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            issues.append(_issue("FAIL", "INVALID_INTEGER", f"{field} must be a non-negative integer"))
    if isinstance(record.get("threads"), int) and not isinstance(record.get("threads"), bool) and record["threads"] <= 0:
        issues.append(_issue("FAIL", "INVALID_THREADS", "threads must be greater than zero"))

    elapsed = record.get("elapsed_seconds")
    if not _is_number(elapsed) or elapsed <= 0:
        issues.append(_issue("FAIL", "INVALID_ELAPSED_SECONDS", "elapsed_seconds must be finite and greater than zero"))

    backend = record.get("backend")
    if backend not in {"cpu", "cuda", "metal", "vulkan", "opencl", "llama.cpp"}:
        issues.append(_issue("FAIL", "INVALID_BACKEND", f"Unsupported backend: {backend!r}"))

    status = record.get("status")
    if status not in {"success", "completed", "timeout", "execution_error", "configuration_error", "launch_error", "nonzero_exit"}:
        issues.append(_issue("FAIL", "INVALID_STATUS", f"Unsupported status: {status!r}"))
    # For the legacy v1.0 experiment contract, any non-successful execution
    # is a failed experiment record, even if the timeout flag is consistent.
    if status not in {"success", "completed"}:
        issues.append(_issue("FAIL", "EXECUTION_NOT_SUCCESSFUL", f"Execution status is {status!r}"))

    return_code = record.get("return_code")
    if return_code is not None and (not isinstance(return_code, int) or isinstance(return_code, bool)):
        issues.append(_issue("FAIL", "INVALID_RETURN_CODE", "return_code must be an integer or null"))
    if status in {"success", "completed"} and return_code != 0:
        issues.append(_issue("FAIL", "SUCCESS_NONZERO_EXIT", "successful status requires return_code 0"))

    if not isinstance(record.get("timed_out"), bool):
        issues.append(_issue("FAIL", "INVALID_TIMED_OUT", "timed_out must be boolean"))
    if not isinstance(record.get("output_nonempty"), bool):
        issues.append(_issue("FAIL", "INVALID_OUTPUT_NONEMPTY", "output_nonempty must be boolean"))
    if record.get("output_nonempty") is False:
        issues.append(_issue("FAIL", "EMPTY_OUTPUT", "output_nonempty is false"))
    char_count = record.get("output_char_count")
    if isinstance(char_count, int) and not isinstance(char_count, bool):
        if char_count < 0:
            issues.append(_issue("FAIL", "INVALID_OUTPUT_CHAR_COUNT", "output_char_count cannot be negative"))
        if char_count == 0 and record.get("output_nonempty") is True:
            issues.append(_issue("FAIL", "INCONSISTENT_OUTPUT_METADATA", "output_nonempty is true but output_char_count is zero"))
        if char_count > 0 and record.get("output_nonempty") is False:
            issues.append(_issue("FAIL", "INCONSISTENT_OUTPUT_METADATA", "output_nonempty is false but output_char_count is positive"))

    temp = record.get("temperature")
    if temp is not None and not _is_number(temp):
        issues.append(_issue("FAIL", "INVALID_TEMPERATURE", "temperature must be a finite number"))
    return issues


def _validate_v11(record: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    required = ("data_source", "benchmark", "config", "execution_status", "exit_code",
                "timing", "performance", "output", "reliability")
    for field in required:
        if field not in record:
            issues.append(_issue("FAIL", "MISSING_V1_1_FIELD", f"Missing v1.1 field: {field}"))

    if record.get("data_source") not in DATA_SOURCES:
        issues.append(_issue("FAIL", "INVALID_DATA_SOURCE", "data_source must be 'real' or 'synthetic'"))

    config = record.get("config")
    if not isinstance(config, dict):
        issues.append(_issue("FAIL", "INVALID_CONFIG", "config must be an object"))
    else:
        for field in ("threads", "ctx_size", "requested_tokens", "timeout_seconds", "seed"):
            value = config.get(field)
            if field in config and value is not None:
                if not isinstance(value, int) or isinstance(value, bool):
                    issues.append(_issue("FAIL", "INVALID_CONFIG_VALUE", f"config.{field} must be an integer or null"))
                elif field != "seed" and value < 1:
                    issues.append(_issue("FAIL", "INVALID_CONFIG_VALUE", f"config.{field} must be positive"))
        for field in ("cli", "model", "prompt"):
            if field in config and config[field] is not None and not isinstance(config[field], str):
                issues.append(_issue("FAIL", "INVALID_CONFIG_VALUE", f"config.{field} must be a string or null"))
        if config.get("temp") is not None and not _is_number(config["temp"]):
            issues.append(_issue("FAIL", "INVALID_CONFIG_VALUE", "config.temp must be finite or null"))

    benchmark = record.get("benchmark")
    if not isinstance(benchmark, dict):
        issues.append(_issue("FAIL", "INVALID_BENCHMARK", "benchmark must be an object"))
    else:
        validators = {
            "is_warmup": lambda x: isinstance(x, bool),
            "repeat_index": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "seed": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "temperature": _is_number,
            "config_id": lambda x: isinstance(x, str),
        }
        for key, predicate in validators.items():
            value = benchmark.get(key)
            if value is not None and not predicate(value):
                issues.append(_issue("FAIL", "INVALID_BENCHMARK_VALUE", f"benchmark.{key} has an invalid type"))

    execution = record.get("execution_status")
    if execution not in EXECUTION_STATUSES:
        issues.append(_issue("FAIL", "INVALID_EXECUTION_STATUS", f"Unsupported execution_status: {execution!r}"))
    exit_code = record.get("exit_code")
    if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)):
        issues.append(_issue("FAIL", "INVALID_EXIT_CODE", "exit_code must be an integer or null"))
    if execution == "success" and exit_code != 0:
        issues.append(_issue("FAIL", "SUCCESS_EXIT_CODE_MISMATCH", "success requires exit_code 0"))
    if execution in {"timeout", "missing_file"} and exit_code is not None:
        issues.append(_issue("FAIL", "NULL_EXIT_CODE_EXPECTED", f"{execution} records should have null exit_code"))
    if execution == "error" and exit_code == 0:
        issues.append(_issue("FAIL", "ERROR_EXIT_CODE_MISMATCH", "error must not have exit_code 0"))

    if record.get("error_message") is not None and not isinstance(record.get("error_message"), str):
        issues.append(_issue("FAIL", "INVALID_ERROR_MESSAGE", "error_message must be a string or null"))

    timing = record.get("timing")
    if not isinstance(timing, dict) or "total_seconds" not in timing:
        issues.append(_issue("FAIL", "MISSING_TOTAL_SECONDS", "timing.total_seconds is required"))
    else:
        elapsed = timing["total_seconds"]
        if not _is_number(elapsed) or elapsed < 0:
            issues.append(_issue("FAIL", "INVALID_TOTAL_SECONDS", "timing.total_seconds must be finite and non-negative"))
        elif elapsed == 0 and execution != "missing_file":
            issues.append(_issue("FAIL", "ZERO_TOTAL_SECONDS", "zero total_seconds is only allowed for missing_file"))

    performance = record.get("performance")
    if not isinstance(performance, dict):
        issues.append(_issue("FAIL", "INVALID_PERFORMANCE", "performance must be an object"))
    else:
        for field in ("prompt_tps", "gen_tps", "peak_memory_mb"):
            value = performance.get(field)
            if field in performance and value is not None and (not _is_number(value) or value < 0):
                issues.append(_issue("FAIL", "INVALID_PERFORMANCE_VALUE", f"performance.{field} must be null or finite and non-negative"))

    output = record.get("output")
    if not isinstance(output, dict):
        issues.append(_issue("FAIL", "INVALID_OUTPUT", "output must be an object"))
    else:
        for field in ("stdout", "stderr"):
            if field not in output or not isinstance(output[field], str):
                issues.append(_issue("FAIL", "INVALID_OUTPUT_FIELD", f"output.{field} must be a string"))

    reliability = record.get("reliability")
    if not isinstance(reliability, dict):
        issues.append(_issue("FAIL", "INVALID_RELIABILITY", "reliability must be an object"))
    else:
        if reliability.get("status") not in RELIABILITY_STATUSES:
            issues.append(_issue("FAIL", "INVALID_RELIABILITY_STATUS", "reliability.status must be PASS, WARNING, or FAIL"))
        if not isinstance(reliability.get("reasons"), list):
            issues.append(_issue("FAIL", "INVALID_RELIABILITY_REASONS", "reliability.reasons must be a list"))
        if "codes" in reliability and not isinstance(reliability["codes"], list):
            issues.append(_issue("FAIL", "INVALID_RELIABILITY_CODES", "reliability.codes must be a list"))
        if "metrics" in reliability and not isinstance(reliability["metrics"], dict):
            issues.append(_issue("FAIL", "INVALID_RELIABILITY_METRICS", "reliability.metrics must be an object"))
        if reliability.get("rules_version") is not None and not isinstance(reliability.get("rules_version"), str):
            issues.append(_issue("FAIL", "INVALID_RULES_VERSION", "reliability.rules_version must be a string or null"))
    return issues


def validate_record(record: Any) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return [_issue("FAIL", "INVALID_RECORD", "record must be a JSON object")]
    run_id = record.get("run_id") if isinstance(record.get("run_id"), str) else None
    issues = []
    for field in ("schema_version", "run_id", "timestamp_utc"):
        if field not in record:
            issues.append(_issue("FAIL", "MISSING_COMMON_FIELD", f"Missing common field: {field}", run_id))
    if not isinstance(record.get("run_id"), str) or not record.get("run_id", "").strip():
        issues.append(_issue("FAIL", "INVALID_RUN_ID", "run_id must be a non-empty string", run_id))
    if not _timestamp_ok(record.get("timestamp_utc")):
        issues.append(_issue("FAIL", "INVALID_TIMESTAMP", "timestamp_utc must be ISO-8601 with timezone", run_id))
    version = record.get("schema_version")
    if version not in SUPPORTED_VERSIONS:
        issues.append(_issue("FAIL", "UNSUPPORTED_SCHEMA_VERSION", f"Unsupported schema_version: {version!r}", run_id))
    elif version == "1.0":
        issues.extend(_validate_v10(record))
    elif version == "1.1":
        issues.extend(_validate_v11(record))
    for item in issues:
        item.setdefault("run_id", run_id)
    return issues


def load_records(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, parse_issues = [], []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("JSONL line must contain an object")
                records.append(item)
            except (json.JSONDecodeError, ValueError) as exc:
                parse_issues.append(_issue("FAIL", "INVALID_JSONL", f"Line {line_no}: {exc}"))
    return records, parse_issues


def _record_result(record: dict[str, Any], issues: list[dict[str, Any]]) -> dict[str, Any]:
    severity = "FAIL" if any(x["severity"] == "FAIL" for x in issues) else (
        "WARNING" if issues else "PASS"
    )
    return {
        "run_id": record.get("run_id"),
        "validation_status": severity,
        "valid": severity == "PASS",
        "issues": issues,
    }


def build_report(records: Any, parse_issues: list[dict[str, Any]] | None = None,
                 check_files: bool = False) -> dict[str, Any]:
    """Build report from records, or from a JSONL path for legacy callers."""
    if isinstance(records, (str, Path)):
        path = Path(records)
        loaded, parse_errors = load_records(path)
        # Legacy build_report(path) callers expect INVALID_JSON; load_records()
        # itself retains the more explicit INVALID_JSONL code.
        for parse_error in parse_errors:
            if parse_error.get("code") == "INVALID_JSONL":
                parse_error["code"] = "INVALID_JSON"
        records = loaded
        parse_issues = list(parse_issues or []) + parse_errors
    if not isinstance(records, list):
        raise TypeError("records must be a list of dictionaries or a JSONL path")
    # The original v1.0 tests pass raw JSONL lines directly as a list of strings.
    if records and all(isinstance(item, str) for item in records):
        parsed_records = []
        line_errors = []
        for line_no, line in enumerate(records, 1):
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("JSON line must contain an object")
                parsed_records.append(parsed)
            except (json.JSONDecodeError, ValueError) as exc:
                line_errors.append(_issue("FAIL", "INVALID_JSON", f"Line {line_no}: {exc}"))
        records = parsed_records
        parse_issues = list(parse_issues or []) + line_errors

    seen = set()
    results = []
    all_issues = list(parse_issues or [])
    for record in records:
        issues = validate_record(record)
        if isinstance(record, dict):
            if record.get("schema_version") == "1.0" and "seed" not in record:
                issues.append(_issue("WARNING", "MISSING_OPTIONAL_METADATA", "Optional metadata field 'seed' is missing", record.get("run_id")))
            rid = record.get("run_id")
            if isinstance(rid, str):
                if rid in seen:
                    issues.append(_issue("FAIL", "DUPLICATE_RUN_ID", "run_id is duplicated", rid))
                seen.add(rid)
            if check_files and record.get("schema_version") == "1.0":
                for field in ("stdout_file", "stderr_file"):
                    file_name = record.get(field)
                    if file_name:
                        candidate = Path(file_name)
                        if not candidate.is_absolute():
                            candidate = Path.cwd() / candidate
                        if not candidate.exists():
                            issues.append(_issue("FAIL", "MISSING_LOG_FILE", f"Log file does not exist: {file_name}", rid))
        else:
            rid = None
        result = _record_result(record if isinstance(record, dict) else {}, issues)
        results.append(result)
        all_issues.extend(issues)

    for parse_issue in parse_issues or []:
        results.append({
            "run_id": parse_issue.get("run_id"),
            "validation_status": "FAIL",
            "valid": False,
            "issues": [parse_issue],
        })

    pass_count = sum(r["validation_status"] == "PASS" for r in results)
    warning_count = sum(r["validation_status"] == "WARNING" for r in results)
    fail_count = sum(r["validation_status"] == "FAIL" for r in results)
    summary = {
        "total_runs": len(results),
        "pass_count": pass_count,
        "warning_count": warning_count,
        "fail_count": fail_count,
    }
    # Preserve both the legacy public report contract and the newer v1.1
    # report keys so callers/tests can migrate without breaking.
    return {
        "summary": summary,
        "results": results,
        "report_version": "1.1",
        "records_checked": len(records),
        "valid_records": pass_count,
        "invalid_records": fail_count,
        "parse_errors": len(parse_issues or []),
        "issues": all_issues,
        "record_results": [
            {"run_id": r["run_id"], "valid": r["valid"], "issues": r["issues"]}
            for r in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BitNet experiment JSONL records (v1.0 and v1.1).")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    report = build_report(args.input, check_files=args.check_files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
