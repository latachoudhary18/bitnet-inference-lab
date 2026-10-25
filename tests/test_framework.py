import unittest

from scripts.inference_check import assess_output
from scripts.config_selector import select_configuration


class ReliabilityTests(unittest.TestCase):
    def test_empty_output_fails(self):
        self.assertEqual(assess_output(" ")["status"], "fail")
        self.assertIn("empty_output", assess_output(" ")["flags"])

    def test_question_mark_heavy_output_is_flagged(self):
        result = assess_output("????????????????????????")
        self.assertEqual(result["status"], "fail")
        self.assertIn("question_mark_heavy_output", result["flags"])

    def test_normal_short_text_passes_heuristic(self):
        result = assess_output("The capital of France is Paris.")
        self.assertEqual(result["status"], "pass_heuristic")

    def test_repeated_text_is_flagged(self):
        result = assess_output("word word word word word word word word word word")
        self.assertIn("high_repetition", result["flags"])


class SelectorTests(unittest.TestCase):
    def test_selects_fastest_passing_run(self):
        results = [
            {"exit_code": 0, "timed_out": False,
             "assessment": {"status": "pass_heuristic"},
             "generation_tokens_per_second": 2.0},
            {"exit_code": 0, "timed_out": False,
             "assessment": {"status": "pass_heuristic"},
             "generation_tokens_per_second": 5.0},
        ]
        self.assertEqual(select_configuration(results)["generation_tokens_per_second"], 5.0)

    def test_does_not_select_failed_run_even_if_fast(self):
        results = [
            {"exit_code": 0, "timed_out": False,
             "assessment": {"status": "fail"},
             "generation_tokens_per_second": 100.0},
        ]
        self.assertIsNone(select_configuration(results))


from unittest.mock import patch, MagicMock
from scripts.inference_check import run_inference


class InferenceRunnerTests(unittest.TestCase):
    @patch("scripts.inference_check.subprocess.run")
    def test_stderr_diagnostics_do_not_fail_clean_stdout(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="The capital of France is Paris.",
            stderr="Loading model...\nPrompt: 10.0 t/s\nGeneration: 5.0 t/s",
            returncode=0,
        )

        result = run_inference(
            cli="fake-cli",
            model="fake-model.gguf",
            prompt="Test prompt",
            timeout=2,
        )

        self.assertEqual(result["assessment"]["status"], "pass_heuristic")
        self.assertIn("Loading model", result["stderr"])
        self.assertEqual(result["generation_tokens_per_second"], 5.0)
        mock_run.assert_called_once()

    @patch("scripts.inference_check.subprocess.run")
    def test_empty_stdout_fails_even_when_stderr_has_logs(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="Model loaded successfully.",
            returncode=0,
        )

        result = run_inference(
            cli="fake-cli",
            model="fake-model.gguf",
            prompt="Test prompt",
            timeout=2,
        )

        self.assertEqual(result["assessment"]["status"], "fail")
        self.assertIn("empty_output", result["assessment"]["flags"])


from scripts.benchmark_report import summarize


class BenchmarkReportTests(unittest.TestCase):
    def test_timeout_is_reported_and_not_selected(self):
        records = [
            {
                "exit_code": None,
                "timed_out": True,
                "generation_tokens_per_second": None,
                "assessment": {
                    "status": "fail",
                    "flags": ["high_repetition"],
                },
            }
        ]

        summary = summarize(records)

        self.assertEqual(summary["total_runs"], 1)
        self.assertEqual(summary["timed_out_runs"], 1)
        self.assertEqual(summary["successful_processes"], 0)
        self.assertIsNone(summary["fastest_eligible_run"])

    def test_fastest_eligible_run_is_selected(self):
        records = [
            {
                "exit_code": 0,
                "timed_out": False,
                "generation_tokens_per_second": 2.0,
                "assessment": {"status": "pass_heuristic"},
            },
            {
                "exit_code": 0,
                "timed_out": False,
                "generation_tokens_per_second": 5.0,
                "assessment": {"status": "pass_heuristic"},
            },
        ]

        summary = summarize(records)

        self.assertEqual(
            summary["fastest_eligible_run"]["generation_tokens_per_second"],
            5.0,
        )

    def test_fast_but_failed_run_is_not_selected(self):
        records = [
            {
                "exit_code": 0,
                "timed_out": False,
                "generation_tokens_per_second": 100.0,
                "assessment": {"status": "fail"},
            }
        ]

        summary = summarize(records)

        self.assertIsNone(summary["fastest_eligible_run"])


if __name__ == "__main__":
    unittest.main()
