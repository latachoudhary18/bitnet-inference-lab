import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

source = Path(r"D:\bitnet-inference-lab\tools\reliability_checker.py")

with tempfile.TemporaryDirectory() as temp:
    root = Path(temp)
    (root / "tools").mkdir()
    (root / "results" / "raw").mkdir(parents=True)

    shutil.copy2(source, root / "tools" / "reliability_checker.py")

    record = {
        "schema_version": "1.0",
        "run_id": "negative_test_r01",
        "experiment_id": "negative_test",
        "timestamp_utc": "2026-11-14T00:00:00Z",
        "model_id": "test_model",
        "model_path": "test.gguf",
        "backend": "cpu",
        "threads": 4,
        "prompt_id": "p01",
        "prompt_text": "Test prompt",
        "max_new_tokens": 32,
        "temperature": 0,
        "seed": 42,
        "timeout_seconds": 120,
        "command": ["test"],
        "elapsed_seconds": 1.0,
        "return_code": 1,
        "status": "success",
        "timed_out": False,
        "output_nonempty": True,
        "output_char_count": 10,
        "stdout_file": "test.stdout",
        "stderr_file": "test.stderr",
        "generation_tokens_per_second": 10.0
    }

    log = root / "results" / "raw" / "experiments.jsonl"
    log.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(root / "tools" / "reliability_checker.py")],
        capture_output=True, text=True
    )

    report = json.loads(
        (root / "results" / "validated" / "validation_report.json").read_text()
    )

    detected = any(x["code"] == "PROCESS_FAILED" for x in report["issues"])

    print("Negative test:", "PASS" if report["overall_status"] == "FAIL" and detected else "FAIL")
    print("Fault detected:", detected)
    print("Checker exit code:", result.returncode)
