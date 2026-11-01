from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.inference_runner import (
    build_command_args,
    parse_performance_metrics,
    run_inference,
)
from scripts.result_schema import validate_record

STUB_SCRIPT = Path(__file__).parent / "fake_llama_cli.py"


def _stub_prefix(mode: str) -> list[str]:
    """Helper to inject the Python test stub into run_inference."""
    return [sys.executable, str(STUB_SCRIPT), "--stub-mode", mode]


# ---------- 1. Command Builder & Dry-Run Tests ----------

def test_command_builder_flags():
    args = build_command_args(
        model="models/test.gguf",
        threads=8,
        ctx_size=256,
        predict=16,
        temp=0.0,
        seed=123,
        prompt="Test prompt",
    )
    # Must use verified flags
    assert "--single-turn" in args
    assert "--simple-io" in args
    assert "--no-display-prompt" in args
    assert "--color" in args and "off" in args
    assert "--n-gpu-layers" in args and "0" in args

    # Must NOT contain rejected legacy flag
    assert "--no-conversation" not in args

    # Verify configurable parameter mapping
    assert args[args.index("--threads") + 1] == "8"
    assert args[args.index("--ctx-size") + 1] == "256"
    assert args[args.index("--n-predict") + 1] == "16"
    assert args[args.index("--seed") + 1] == "123"
    assert args[args.index("--temp") + 1] == "0.0"
    assert args[args.index("--prompt") + 1] == "Test prompt"


def test_dry_run_does_not_execute(tmp_path):
    out_file = tmp_path / "dry_run.jsonl"
    result = run_inference(
        cli="fake_cli.exe",
        model="fake_model.gguf",
        prompt="Dry run test",
        dry_run=True,
        output_file=out_file,
    )
    assert result["dry_run"] is True
    assert "--single-turn" in result["command"]
    assert not out_file.exists()


# ---------- 2. Pre-Flight File Checks ----------

def test_missing_cli_path(tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()

    rec = run_inference(
        cli=tmp_path / "definitely_non_existent_cli.exe",
        model=model,
        data_source="synthetic",
    )
    assert rec["execution_status"] == "missing_file"
    assert rec["exit_code"] is None
    assert "CLI executable not found" in rec["error_message"]
    assert rec["reliability"]["status"] == "FAIL"
    assert validate_record(rec) == []


def test_missing_model_path(tmp_path):
    cli = tmp_path / "fake_cli.exe"
    cli.touch()

    rec = run_inference(
        cli=cli,
        model=tmp_path / "definitely_non_existent_model.gguf",
        data_source="synthetic",
    )
    assert rec["execution_status"] == "missing_file"
    assert rec["exit_code"] is None
    assert "Model file not found" in rec["error_message"]
    assert rec["reliability"]["status"] == "FAIL"
    assert validate_record(rec) == []


# ---------- 3. Subprocess Execution & Stream Separation (Synthetic Stubs) ----------

def test_stub_normal_execution(tmp_path):
    out_file = tmp_path / "normal_run.jsonl"
    rec = run_inference(
        command_prefix=_stub_prefix("normal"),
        output_file=out_file,
        data_source="synthetic",
    )
    assert rec["execution_status"] == "success"
    assert rec["exit_code"] == 0
    assert rec["reliability"]["status"] == "PASS"
    assert "The capital of France" in rec["output"]["stdout"]
    assert "llama_perf_context_print" in rec["output"]["stderr"]
    assert rec["performance"]["prompt_tps"] == 1000.0
    assert rec["performance"]["gen_tps"] == 100.0
    assert rec["data_source"] == "synthetic"
    assert validate_record(rec) == []
    assert out_file.exists()


def test_stub_question_marks_fails_reliability(tmp_path):
    rec = run_inference(
        command_prefix=_stub_prefix("question_marks"),
        data_source="synthetic",
    )
    # Execution succeeded (process exited 0), but reliability gate caught the failure!
    assert rec["execution_status"] == "success"
    assert rec["exit_code"] == 0
    assert rec["reliability"]["status"] == "FAIL"
    assert "only_question_marks" in rec["reliability"]["codes"]
    assert validate_record(rec) == []


def test_stub_empty_output_fails_reliability(tmp_path):
    rec = run_inference(
        command_prefix=_stub_prefix("empty"),
        data_source="synthetic",
    )
    assert rec["execution_status"] == "success"
    assert rec["exit_code"] == 0
    assert rec["reliability"]["status"] == "FAIL"
    assert "empty_output" in rec["reliability"]["codes"]
    assert validate_record(rec) == []


def test_stub_garbled_output_fails_reliability(tmp_path):
    rec = run_inference(
        command_prefix=_stub_prefix("garbled"),
        data_source="synthetic",
    )
    assert rec["execution_status"] == "success"
    assert rec["exit_code"] == 0
    assert rec["reliability"]["status"] == "FAIL"
    assert "replacement_chars" in rec["reliability"]["codes"]
    assert validate_record(rec) == []


def test_stub_nonzero_exit_formatting(tmp_path):
    rec = run_inference(
        command_prefix=_stub_prefix("nonzero_exit"),
        data_source="synthetic",
    )
    assert rec["execution_status"] == "error"
    assert rec["exit_code"] == 1
    assert "CLI exited with code 1" in rec["error_message"]
    assert "Fatal error in inference runtime" in rec["error_message"]
    assert rec["reliability"]["status"] == "FAIL"
    assert validate_record(rec) == []


def test_stub_unsupported_arg_captured(tmp_path):
    rec = run_inference(
        command_prefix=_stub_prefix("unsupported_arg"),
        data_source="synthetic",
    )
    assert rec["execution_status"] == "error"
    assert "--no-conversation is not supported" in rec["output"]["stderr"]
    assert validate_record(rec) == []


def test_stub_non_ascii_and_invalid_bytes(tmp_path):
    rec = run_inference(
        command_prefix=_stub_prefix("non_ascii_bytes"),
        data_source="synthetic",
    )
    # Errors="replace" ensures no uncaught UnicodeDecodeError
    assert rec["execution_status"] == "success"
    assert "Café" in rec["output"]["stdout"]
    assert validate_record(rec) == []


def test_stub_large_output_no_pipe_deadlock(tmp_path):
    # Generates 100 KB on stdout and 100 KB on stderr
    rec = run_inference(
        command_prefix=_stub_prefix("large_output"),
        data_source="synthetic",
    )
    assert rec["execution_status"] == "success"
    assert len(rec["output"]["stdout"]) > 50000
    assert len(rec["output"]["stderr"]) > 50000
    assert validate_record(rec) == []


def test_stub_timeout_kills_process(tmp_path):
    # Stub sleeps 15s; runner timeout is 1s
    rec = run_inference(
        command_prefix=_stub_prefix("hang"),
        timeout=1,
        data_source="synthetic",
    )
    assert rec["execution_status"] == "timeout"
    assert rec["exit_code"] is None
    assert "Inference exceeded timeout ceiling of 1 seconds" in rec["error_message"]
    assert rec["reliability"]["status"] == "FAIL"
    assert validate_record(rec) == []


# ---------- 4. Performance Parsing Unit Tests ----------

def test_performance_parsing_formats():
    # Format 1: bracketed summary
    p1 = parse_performance_metrics(
        stdout="The answer is 42.",
        stderr="[ Prompt: 24.1 t/s | Generation: 5.2 t/s ]",
    )
    assert p1["prompt_tps"] == 24.1
    assert p1["gen_tps"] == 5.2

    # Format 2: llama_perf_context_print lines
    llama_log = (
        "llama_perf_context_print: prompt eval time = 50.00 ms / 5 tokens ( 10.00 ms per token, 100.00 tokens per second)\n"
        "llama_perf_context_print:        eval time = 100.00 ms / 10 tokens ( 10.00 ms per token, 100.00 tokens per second)\n"
    )
    p2 = parse_performance_metrics(stdout="", stderr=llama_log)
    assert p2["prompt_tps"] == 100.0
    assert p2["gen_tps"] == 100.0

    # Format 3: unmeasured / garbage output
    p3 = parse_performance_metrics(stdout="Garbage output", stderr="Some unparseable log")
    assert p3["prompt_tps"] is None
    assert p3["gen_tps"] is None


# ---------- 5. Schema v1.1 Benchmark Block & Round-Trip Persistence ----------

def test_v1_1_benchmark_fields_and_persistence(tmp_path):
    out_file = tmp_path / "runs.jsonl"
    bm_meta = {
        "is_warmup": True,
        "repeat_index": 2,
        "seed": 42,
        "temperature": 0.0,
        "config_id": "cfg-sweep-01",
    }
    rec = run_inference(
        command_prefix=_stub_prefix("normal"),
        output_file=out_file,
        benchmark=bm_meta,
        data_source="synthetic",
    )
    assert rec["data_source"] == "synthetic"
    assert rec["benchmark"]["is_warmup"] is True
    assert rec["benchmark"]["repeat_index"] == 2
    assert rec["benchmark"]["config_id"] == "cfg-sweep-01"

    # Read back from file
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    loaded = json.loads(lines[0])
    assert loaded["benchmark"]["config_id"] == "cfg-sweep-01"
    assert validate_record(loaded) == []

