import json
import pytest
from scripts.result_schema import build_record, validate_record, append_jsonl

CFG = {"model": "test.gguf", "threads": 4, "ctx": 512}
GOOD = "The capital of France is Paris, known for the Eiffel Tower and museums."


def make(**kw):
    base = dict(run_id="t1", config=CFG, execution_status="success", exit_code=0,
                stdout=GOOD, stderr="", total_seconds=1.2)
    base.update(kw)
    return build_record(**base)


def test_good_record_is_valid_and_passes():
    r = make()
    assert validate_record(r) == []
    assert r["reliability"]["status"] == "PASS"


def test_question_marks_record_fails_reliability_but_execution_succeeds():
    r = make(stdout="??????????")
    assert r["execution_status"] == "success"
    assert r["reliability"]["status"] == "FAIL"


def test_timeout_record():
    r = make(execution_status="timeout", exit_code=None, stdout="")
    assert r["reliability"]["status"] == "FAIL"
    assert any("timeout" in x for x in r["reliability"]["reasons"])


def test_missing_file_record():
    r = make(execution_status="missing_file", exit_code=None, stdout="",
             error_message="llama-cli.exe not found")
    assert r["reliability"]["status"] == "FAIL"
    assert any("missing_file" in x for x in r["reliability"]["reasons"])
    assert validate_record(r) == []


def test_validate_catches_bad_record():
    assert validate_record({"run_id": "x"}) != []


def test_jsonl_roundtrip(tmp_path):
    p = tmp_path / "out" / "runs.jsonl"
    append_jsonl(str(p), make())
    append_jsonl(str(p), make(run_id="t2"))
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["run_id"] == "t2"


# ---------- Stage A Phase 3 Schema Evolution Tests ----------

def test_v1_0_record_validates():
    # Legacy v1.0 record without benchmark block or data_source
    legacy_rec = {
        "schema_version": "1.0",
        "run_id": "legacy-001",
        "timestamp_utc": "2026-10-25T00:00:00Z",
        "config": CFG,
        "execution_status": "success",
        "exit_code": 0,
        "error_message": None,
        "timing": {"total_seconds": 2.5},
        "performance": {"prompt_tps": 20.0, "gen_tps": 5.0, "peak_memory_mb": None},
        "output": {"stdout": GOOD, "stderr": ""},
        "reliability": {"status": "PASS", "reasons": []},
    }
    problems = validate_record(legacy_rec)
    assert problems == []


def test_v1_1_record_validates():
    rec = build_record(
        run_id="v11-001",
        config=CFG,
        execution_status="success",
        exit_code=0,
        stdout=GOOD,
        stderr="",
        total_seconds=1.5,
        benchmark={"is_warmup": False, "repeat_index": 1, "seed": 42, "temperature": 0.7, "config_id": "cfg-a"},
        data_source="real",
    )
    assert rec["schema_version"] == "1.1"
    assert rec["data_source"] == "real"
    assert rec["benchmark"]["is_warmup"] is False
    assert rec["benchmark"]["seed"] == 42
    assert validate_record(rec) == []


def test_v1_1_bad_data_source_rejected():
    rec = make()
    rec["data_source"] = "fabricated_source"
    problems = validate_record(rec)
    assert any("bad data_source" in p for p in problems)

    with pytest.raises(ValueError, match="Invalid data_source"):
        build_record(
            run_id="bad-ds",
            config=CFG,
            execution_status="success",
            exit_code=0,
            stdout=GOOD,
            stderr="",
            total_seconds=1.0,
            data_source="invalid_source",
        )


def test_v1_1_bad_benchmark_field_rejected():
    rec = make()
    rec["benchmark"]["is_warmup"] = "not_a_bool"
    rec["benchmark"]["seed"] = "not_an_int"
    problems = validate_record(rec)
    assert any("benchmark.is_warmup" in p for p in problems)
    assert any("benchmark.seed" in p for p in problems)


def test_synthetic_record_integration_roundtrip(tmp_path):
    # Tests build_record -> validate_record -> append_jsonl -> reading back
    synth_path = tmp_path / "synthetic_runs.jsonl"
    record = build_record(
        run_id="synth-integration-01",
        config=CFG,
        execution_status="success",
        exit_code=0,
        stdout=GOOD,
        stderr="",
        total_seconds=0.85,
        data_source="synthetic",
        benchmark={"is_warmup": True, "repeat_index": 0, "config_id": "synth-cfg"},
    )
    assert record["data_source"] == "synthetic"
    assert validate_record(record) == []

    append_jsonl(str(synth_path), record)

    loaded_lines = synth_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(loaded_lines) == 1
    loaded = json.loads(loaded_lines[0])

    assert loaded["run_id"] == "synth-integration-01"
    assert loaded["data_source"] == "synthetic"
    assert loaded["benchmark"]["is_warmup"] is True
    assert loaded["benchmark"]["config_id"] == "synth-cfg"
    assert loaded["reliability"]["status"] == "PASS"
    assert loaded["reliability"]["rules_version"] == "1.1"
    assert isinstance(loaded["reliability"]["metrics"], dict)
    assert isinstance(loaded["reliability"]["codes"], list)
