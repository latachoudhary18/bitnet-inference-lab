import json
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