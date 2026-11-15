
import json
import tempfile
import unittest
from pathlib import Path

from tools.reliability_checker import build_report


def valid_record():
    return {
        "schema_version": "1.0",
        "run_id": "baseline_cpu_t4_p01_r01",
        "experiment_id": "baseline",
        "timestamp_utc": "2026-11-14T17:00:00Z",
        "model_id": "test-model",
        "model_path": "models/test.gguf",
        "backend": "cpu",
        "threads": 4,
        "prompt_id": "p01",
        "prompt_text": "Explain a neural network.",
        "max_new_tokens": 32,
        "temperature": 0.0,
        "seed": 42,
        "timeout_seconds": 120,
        "command": ["python", "-c", "print('hello')"],
        "elapsed_seconds": 1.5,
        "return_code": 0,
        "status": "success",
        "timed_out": False,
        "output_nonempty": True,
        "output_char_count": 5,
        "stdout_file": "test_stdout.txt",
        "stderr_file": "test_stderr.txt",
    }


class TestExperimentValidation(unittest.TestCase):

    def run_records(self, lines):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "experiments.jsonl"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return build_report(path)

    def run_record(self, record):
        return self.run_records([json.dumps(record)])

    def test_complete_record_passes(self):
        report = self.run_record(valid_record())
        self.assertEqual(report["results"][0]["validation_status"], "PASS")

    def test_missing_required_field_fails(self):
        record = valid_record()
        del record["model_id"]
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_invalid_json_fails(self):
        report = self.run_records(["{invalid json"])
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")
        self.assertEqual(
            report["results"][0]["issues"][0]["code"], "INVALID_JSON"
        )

    def test_duplicate_run_id_fails(self):
        record = valid_record()
        report = self.run_records([
            json.dumps(record),
            json.dumps(record),
        ])
        self.assertEqual(report["summary"]["fail_count"], 1)
        self.assertIn(
            "DUPLICATE_RUN_ID",
            [i["code"] for i in report["results"][1]["issues"]],
        )

    def test_zero_latency_fails(self):
        record = valid_record()
        record["elapsed_seconds"] = 0
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_negative_latency_fails(self):
        record = valid_record()
        record["elapsed_seconds"] = -1
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_nonzero_exit_fails(self):
        record = valid_record()
        record["status"] = "nonzero_exit"
        record["return_code"] = 1
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_timeout_fails(self):
        record = valid_record()
        record["status"] = "timeout"
        record["timed_out"] = True
        record["return_code"] = None
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_empty_output_fails(self):
        record = valid_record()
        record["output_nonempty"] = False
        record["output_char_count"] = 0
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_missing_optional_metadata_warns(self):
        record = valid_record()
        del record["seed"]
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "WARNING")

    def test_inconsistent_output_metadata_fails(self):
        record = valid_record()
        record["output_char_count"] = 0
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_summary_counts(self):
        good = valid_record()
        bad = valid_record()
        bad["run_id"] = "baseline_cpu_t4_p02_r01"
        bad["elapsed_seconds"] = 0

        report = self.run_records([
            json.dumps(good),
            json.dumps(bad),
        ])

        self.assertEqual(report["summary"]["total_runs"], 2)
        self.assertEqual(report["summary"]["pass_count"], 1)
        self.assertEqual(report["summary"]["fail_count"], 1)



    def test_invalid_backend_fails(self):
        record = valid_record()
        record["backend"] = "quantum"
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_zero_threads_fails(self):
        record = valid_record()
        record["threads"] = 0
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_negative_threads_fails(self):
        record = valid_record()
        record["threads"] = -2
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_success_with_nonzero_return_code_fails(self):
        record = valid_record()
        record["return_code"] = 1
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_launch_error_fails(self):
        record = valid_record()
        record["status"] = "launch_error"
        record["return_code"] = None
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_nan_latency_fails(self):
        record = valid_record()
        record["elapsed_seconds"] = float("nan")
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_infinite_latency_fails(self):
        record = valid_record()
        record["elapsed_seconds"] = float("inf")
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_invalid_timestamp_fails(self):
        record = valid_record()
        record["timestamp_utc"] = "not-a-timestamp"
        report = self.run_record(record)
        self.assertEqual(report["results"][0]["validation_status"], "FAIL")

    def test_raw_input_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "experiments.jsonl"
            original = json.dumps(valid_record()) + "\n"
            path.write_text(original, encoding="utf-8")

            build_report(path)

            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_missing_log_files_are_reported(self):
        record = valid_record()

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "experiments.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            report = build_report(path, check_files=True)
            result = report["results"][0]

            self.assertEqual(result["validation_status"], "FAIL")
            codes = [item["code"] for item in result["issues"]]
            self.assertIn("MISSING_LOG_FILE", codes)

if __name__ == "__main__":
    unittest.main()
