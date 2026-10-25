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


if __name__ == "__main__":
    unittest.main()
