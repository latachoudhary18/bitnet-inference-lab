"""Shared experiment-record schema for the BitNet inference lab.

execution_status answers: did the process run?
reliability.status answers: does the output look sane?
These are deliberately separate.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from scripts.reliability_checker import check_output, FAIL

SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1.1"}
EXECUTION_STATUSES = {"success", "timeout", "error", "missing_file"}
RELIABILITY_STATUSES = {"PASS", "WARNING", "FAIL"}
DATA_SOURCES = {"real", "synthetic"}


def build_record(
    run_id: str,
    config: dict[str, Any],
    execution_status: str,
    exit_code: int | None,
    stdout: str,
    stderr: str,
    total_seconds: float,
    prompt: str | None = None,
    performance: dict[str, Any] | None = None,
    error_message: str | None = None,
    benchmark: dict[str, Any] | None = None,
    data_source: str = "real",
    requested_tokens: int | None = None,
) -> dict[str, Any]:
    """Create one experiment record (v1.1), including the reliability verdict."""
    if data_source not in DATA_SOURCES:
        raise ValueError(f"Invalid data_source: {data_source!r}. Must be one of {DATA_SOURCES}")

    perf = {"prompt_tps": None, "gen_tps": None, "peak_memory_mb": None}
    perf.update(performance or {})

    # Auto-resolve requested_tokens from config if omitted
    if requested_tokens is None and isinstance(config, dict):
        requested_tokens = config.get("requested_tokens") or config.get("n_predict")

    reliability = check_output(
        stdout,
        exit_code=exit_code,
        timed_out=(execution_status == "timeout"),
        prompt=prompt,
        requested_tokens=requested_tokens,
    )
    if execution_status == "missing_file":
        reliability["status"] = FAIL
        msg = "FAIL:missing_file: executable or model file not found"
        if msg not in reliability["reasons"]:
            reliability["reasons"].append(msg)
        if "codes" in reliability and "missing_file" not in reliability["codes"]:
            reliability["codes"].append("missing_file")

    bm: dict[str, Any] = {
        "is_warmup": None,
        "repeat_index": None,
        "seed": None,
        "temperature": None,
        "config_id": None,
    }
    if benchmark:
        bm.update(benchmark)

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "config": config,                      # model, threads, ctx, prompt, n_predict...
        "benchmark": bm,                       # warmup, repeat, seed, temperature, id
        "execution_status": execution_status,  # success|timeout|error|missing_file
        "exit_code": exit_code,
        "error_message": error_message,
        "timing": {"total_seconds": total_seconds},
        "performance": perf,
        "output": {"stdout": stdout, "stderr": stderr},
        "reliability": reliability,
    }


def validate_record(rec: dict[str, Any]) -> list[str]:
    """Return a list of problems (empty list = valid).

    Validates records against SCHEMA_VERSION 1.0 or 1.1 contracts.
    """
    problems: list[str] = []
    if not isinstance(rec, dict):
        return ["record must be a dict"]

    required_common = [
        "schema_version", "run_id", "timestamp_utc", "config",
        "execution_status", "exit_code", "timing", "performance",
        "output", "reliability"
    ]
    for key in required_common:
        if key not in rec:
            problems.append(f"missing key: {key}")
    if problems:
        return problems

    ver = str(rec["schema_version"])
    if ver not in SUPPORTED_SCHEMA_VERSIONS:
        problems.append(f"unsupported schema_version: {ver}")

    if rec["execution_status"] not in EXECUTION_STATUSES:
        problems.append(f"bad execution_status: {rec['execution_status']}")

    if not isinstance(rec.get("timing"), dict) or "total_seconds" not in rec["timing"]:
        problems.append("timing.total_seconds missing")

    rel = rec.get("reliability")
    if not isinstance(rel, dict):
        problems.append("reliability must be a dict")
    else:
        if rel.get("status") not in RELIABILITY_STATUSES:
            problems.append("bad reliability.status")
        if not isinstance(rel.get("reasons"), list):
            problems.append("reliability.reasons must be a list")
        if "codes" in rel and not isinstance(rel["codes"], list):
            problems.append("reliability.codes must be a list")
        if "metrics" in rel and not isinstance(rel["metrics"], dict):
            problems.append("reliability.metrics must be a dict")
        if "rules_version" in rel and not isinstance(rel["rules_version"], str):
            problems.append("reliability.rules_version must be a str")

    # Fields specific to v1.1 (or when present in v1.0)
    if ver == "1.1" or "data_source" in rec:
        ds = rec.get("data_source")
        if ds not in DATA_SOURCES:
            problems.append(f"bad data_source: {ds!r}")

    if ver == "1.1" or "benchmark" in rec:
        bm = rec.get("benchmark")
        if bm is not None and not isinstance(bm, dict):
            problems.append("benchmark must be a dict or None")
        elif isinstance(bm, dict):
            if bm.get("is_warmup") is not None and not isinstance(bm["is_warmup"], bool):
                problems.append("benchmark.is_warmup must be a bool or None")
            if bm.get("repeat_index") is not None and not isinstance(bm["repeat_index"], int):
                problems.append("benchmark.repeat_index must be an int or None")
            if bm.get("seed") is not None and not isinstance(bm["seed"], int):
                problems.append("benchmark.seed must be an int or None")
            if bm.get("temperature") is not None and not isinstance(bm["temperature"], (int, float)):
                problems.append("benchmark.temperature must be a float/int or None")
            if bm.get("config_id") is not None and not isinstance(bm["config_id"], str):
                problems.append("benchmark.config_id must be a str or None")

    return problems


def append_jsonl(path: str, record: dict[str, Any]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
