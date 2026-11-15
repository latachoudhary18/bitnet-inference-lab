import copy
import json

import pytest

from tools.reliability_checker import build_report, load_records, validate_record


GOOD_V11 = {
    "schema_version": "1.1",
    "run_id": "example-run-id",
    "timestamp_utc": "2026-11-15T10:17:32.866916+00:00",
    "data_source": "synthetic",
    "config": {
        "cli": "path/to/llama-cli.exe",
        "model": "path/to/model.gguf",
        "threads": 4,
        "ctx_size": 512,
        "prompt": "Explain what a 1-bit language model is in simple terms.",
        "requested_tokens": 32,
        "timeout_seconds": 180,
        "temp": 0.0,
        "seed": 42,
    },
    "benchmark": {
        "is_warmup": None,
        "repeat_index": None,
        "seed": None,
        "temperature": None,
        "config_id": None,
    },
    "execution_status": "success",
    "exit_code": 0,
    "error_message": None,
    "timing": {"total_seconds": 0.1274},
    "performance": {"prompt_tps": 1000.0, "gen_tps": 100.0, "peak_memory_mb": None},
    "output": {"stdout": "The capital of France is Paris.", "stderr": "diagnostic output"},
    "reliability": {
        "status": "PASS",
        "reasons": [],
        "codes": [],
        "metrics": {"char_count": 32},
        "rules_version": "1.1",
    },
}


GOOD_V10 = {
    "schema_version": "1.0",
    "run_id": "legacy-run",
    "timestamp_utc": "2026-11-15T10:17:32Z",
    "experiment_id": "exp-01",
    "model_id": "bitnet-test",
    "backend": "llama.cpp",
    "threads": 4,
    "prompt_id": "p01",
    "prompt_text": "Explain BitNet.",
    "max_new_tokens": 32,
    "temperature": 0.0,
    "timeout_seconds": 60,
    "command": "llama-cli ...",
    "elapsed_seconds": 1.25,
    "return_code": 0,
    "status": "completed",
    "timed_out": False,
    "output_nonempty": True,
    "output_char_count": 30,
}


def test_nested_v11_sample_is_valid():
    assert validate_record(GOOD_V11) == []


def test_legacy_flat_v10_record_remains_valid():
    assert validate_record(GOOD_V10) == []


def test_v11_does_not_require_legacy_flat_aliases():
    record = copy.deepcopy(GOOD_V11)
    assert not any(issue["code"] == "MISSING_V1_FIELD" for issue in validate_record(record))


@pytest.mark.parametrize("field", ["prompt_tps", "gen_tps", "peak_memory_mb"])
def test_v11_performance_metrics_may_be_null(field):
    record = copy.deepcopy(GOOD_V11)
    record["performance"][field] = None
    assert validate_record(record) == []


def test_v11_missing_file_allows_zero_elapsed_and_null_exit_code():
    record = copy.deepcopy(GOOD_V11)
    record.update(execution_status="missing_file", exit_code=None)
    record["timing"]["total_seconds"] = 0
    record["reliability"]["status"] = "FAIL"
    assert validate_record(record) == []


def test_v11_timeout_allows_null_exit_code_and_zero_or_positive_elapsed():
    record = copy.deepcopy(GOOD_V11)
    record.update(execution_status="timeout", exit_code=None)
    record["reliability"]["status"] = "FAIL"
    assert validate_record(record) == []


def test_v11_rejects_unknown_data_source():
    record = copy.deepcopy(GOOD_V11)
    record["data_source"] = "mock"
    assert "INVALID_DATA_SOURCE" in {i["code"] for i in validate_record(record)}


def test_v11_rejects_bad_execution_status():
    record = copy.deepcopy(GOOD_V11)
    record["execution_status"] = "completed"
    assert "INVALID_EXECUTION_STATUS" in {i["code"] for i in validate_record(record)}


def test_v11_success_requires_zero_exit_code():
    record = copy.deepcopy(GOOD_V11)
    record["exit_code"] = 1
    assert "SUCCESS_EXIT_CODE_MISMATCH" in {i["code"] for i in validate_record(record)}


def test_v11_allows_optional_config_fields_to_be_absent():
    record = copy.deepcopy(GOOD_V11)
    record["config"] = {"model": "model.gguf"}
    assert validate_record(record) == []


def test_v11_rejects_nonfinite_performance_metric():
    record = copy.deepcopy(GOOD_V11)
    record["performance"]["gen_tps"] = float("nan")
    assert "INVALID_PERFORMANCE_VALUE" in {i["code"] for i in validate_record(record)}


def test_v11_rejects_non_string_inline_output():
    record = copy.deepcopy(GOOD_V11)
    record["output"]["stdout"] = None
    assert "INVALID_OUTPUT_FIELD" in {i["code"] for i in validate_record(record)}


def test_v11_rejects_bad_benchmark_types():
    record = copy.deepcopy(GOOD_V11)
    record["benchmark"]["is_warmup"] = "false"
    assert "INVALID_BENCHMARK_VALUE" in {i["code"] for i in validate_record(record)}


def test_v11_rejects_timezone_naive_timestamp():
    record = copy.deepcopy(GOOD_V11)
    record["timestamp_utc"] = "2026-11-15T10:17:32"
    assert "INVALID_TIMESTAMP" in {i["code"] for i in validate_record(record)}


def test_v11_reliability_fields_are_structural_only():
    # Text quality verdict and metrics are produced by scripts/result_schema.py /
    # scripts/reliability_checker.py. This validator checks only their structure.
    record = copy.deepcopy(GOOD_V11)
    record["reliability"]["status"] = "WARNING"
    record["reliability"]["reasons"] = ["some upstream rule"]
    assert validate_record(record) == []


def test_v11_zero_elapsed_is_rejected_except_missing_file():
    record = copy.deepcopy(GOOD_V11)
    record["timing"]["total_seconds"] = 0
    assert "ZERO_TOTAL_SECONDS" in {i["code"] for i in validate_record(record)}


def test_build_report_counts_invalid_and_duplicate_ids():
    records = [copy.deepcopy(GOOD_V11), copy.deepcopy(GOOD_V11)]
    report = build_report(records)
    assert report["records_checked"] == 2
    assert report["valid_records"] == 1
    assert report["invalid_records"] == 1
    assert "DUPLICATE_RUN_ID" in {i["code"] for i in report["issues"]}


def test_load_records_reports_invalid_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(GOOD_V11) + "\n{bad json}\n", encoding="utf-8")
    records, errors = load_records(path)
    assert len(records) == 1
    assert len(errors) == 1
    assert errors[0]["code"] == "INVALID_JSONL"


def test_validation_does_not_mutate_input():
    record = copy.deepcopy(GOOD_V11)
    before = copy.deepcopy(record)
    validate_record(record)
    assert record == before
