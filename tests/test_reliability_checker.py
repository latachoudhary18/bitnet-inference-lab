import pytest
from scripts.reliability_checker import check_output, PASS, WARNING, FAIL

GOOD = ("The capital of France is Paris. It is known for the Eiffel Tower, "
        "museums, and a long history of art and culture.")


def codes(result):
    return " ".join(result["reasons"])


# ---------- main cases ----------

def test_good_output_passes():
    r = check_output(GOOD)
    assert r["status"] == PASS and r["reasons"] == []


@pytest.mark.parametrize("text", ["", "   \n\t ", None])
def test_empty_output_fails(text):
    r = check_output(text)
    assert r["status"] == FAIL and "empty_output" in codes(r)


def test_question_marks_only_fails():
    r = check_output("? ? ? ? ? ? ? ? ? ? ? ? ?\n???????")
    assert r["status"] == FAIL and "only_question_marks" in codes(r)


def test_mostly_question_marks_fails():
    r = check_output("??????????????????? hi ?????????")
    assert r["status"] == FAIL


def test_single_char_repeat_fails():
    r = check_output("aaaaaaaaaaaaaaaaaaaaaaaaa")
    assert r["status"] == FAIL and "single_char_repeat" in codes(r)


def test_repeated_words_fail():
    r = check_output("the cat the cat the cat the cat the cat the cat the cat the cat")
    assert r["status"] == FAIL


def test_repeated_lines_fail():
    r = check_output("I am a model.\n" * 6)
    assert r["status"] == FAIL and "repeated_lines" in codes(r)


def test_short_output_warns():
    r = check_output("Paris.")
    assert r["status"] == WARNING and "too_short" in codes(r)


def test_timeout_fails_even_with_good_text():
    r = check_output(GOOD, timed_out=True)
    assert r["status"] == FAIL and "timeout" in codes(r)


def test_nonzero_exit_fails():
    r = check_output(GOOD, exit_code=1)
    assert r["status"] == FAIL and "nonzero_exit" in codes(r)


def test_readable_but_wrong_still_passes():
    # Documents a LIMITATION: fluent but wrong text passes sanity checks.
    r = check_output("The capital of France is Berlin, a city famous for its beaches and mountains.")
    assert r["status"] == PASS


# ---------- edge cases ----------

def test_prompt_echo_does_not_hide_garbage():
    prompt = "Explain the theory of relativity in simple terms please."
    r = check_output(prompt + " ??????????????", prompt=prompt)
    assert r["status"] == FAIL and "only_question_marks" in codes(r)


def test_prompt_only_echo_is_empty():
    prompt = "Hello there, how are you today?"
    r = check_output(prompt, prompt=prompt)
    assert r["status"] == FAIL and "empty_output" in codes(r)


def test_replacement_chars_fail():
    r = check_output("\ufffd" * 30)
    assert r["status"] == FAIL and "replacement_chars" in codes(r)


def test_non_string_input_fails():
    r = check_output(12345)
    assert r["status"] == FAIL and "invalid_type" in codes(r)


def test_multiple_problems_all_reported():
    r = check_output("", exit_code=1, timed_out=True)
    text = codes(r)
    assert "timeout" in text and "nonzero_exit" in text and "empty_output" in text


def test_exit_code_none_is_ok():
    assert check_output(GOOD, exit_code=None)["status"] == PASS


def test_mixed_question_marks_warns():
    r = check_output("What is this ??? I do not know ??? maybe ???? something")
    assert r["status"] in (WARNING, PASS)


def test_symbols_only_warns():
    r = check_output("#### **** ---- #### **** ---- ####")
    assert r["status"] in (WARNING, FAIL)


# ---------- Stage A Phase 3 Upgraded Rules & Guard Tests ----------

def test_repeated_4gram_fails():
    phrase = "in the dark forest " * 6 + "we found an ancient castle."
    r = check_output(phrase)
    assert r["status"] == FAIL
    assert "repeated_4gram" in r["codes"] or "repeated_phrase" in r["codes"]


def test_tail_loop_fails():
    text = (
        "The model was trained on a large dataset and then "
        "and then and then and then and then and then and then."
    )
    r = check_output(text)
    assert r["status"] == FAIL
    assert "tail_loop" in r["codes"]


def test_unspaced_char_ngram_loop_fails():
    text = "abcdefgabcdefgabcdefgabcdefgabcdefgabcdefg"
    r = check_output(text)
    assert r["status"] == FAIL
    assert "char_ngram_loop" in r["codes"]


def test_low_shannon_entropy_fails():
    # Length > 50 characters, only two alternating characters -> entropy ~1.0 < 2.0
    text = "abababababababababababababababababababababababababababababababababab."
    r = check_output(text)
    assert r["status"] == FAIL
    assert "low_entropy" in r["codes"] or "low_char_diversity" in r["codes"]


def test_nul_byte_fails():
    r = check_output("Valid sentence with a hidden \x00 NUL byte.")
    assert r["status"] == FAIL
    assert "nul_byte" in r["codes"]


def test_ansi_escape_warns():
    r = check_output("Processing output: \x1b[32mSuccess\x1b[0m result completed.")
    assert r["status"] in (WARNING, FAIL)
    assert "ansi_escape" in r["codes"]


def test_mojibake_warns():
    r = check_output("The cafÃ© sold crÃªpes and â€œfineâ€\x9d coffee to the Ã©lÃ¨ve.")
    assert r["status"] in (WARNING, FAIL)
    assert "mojibake" in r["codes"]


def test_clean_utf8_passes():
    r = check_output("Café résumé façade naïve Straße 日本語 text with valid characters.")
    assert r["status"] == PASS
    assert "mojibake" not in r["codes"]
    assert "non_printable_chars" not in r["codes"]


def test_truncated_output_warns():
    r = check_output("The quick brown fox jumps over the lazy dog and runs into the")
    assert r["status"] == WARNING
    assert "truncated_output" in r["codes"]


def test_token_cap_reached_downgrades_to_info():
    # 14 words / tokens without terminal punctuation, requested_tokens=14
    text = "The quick brown fox jumps over the lazy dog and runs into the wild"
    r = check_output(text, requested_tokens=14)
    assert r["status"] == PASS
    assert "token_cap_reached" in r["codes"]
    assert "truncated_output" not in r["codes"]


def test_poem_with_refrain_does_not_fail():
    poem = (
        "The autumn leaves begin to fall,\n"
        "The winter whispers through the hall.\n"
        "Listen to the gentle breeze.\n"
        "The snow will cover every tree,\n"
        "A frozen world for you and me.\n"
        "Listen to the gentle breeze.\n"
    )
    r = check_output(poem)
    assert r["status"] != FAIL


def test_numbered_list_does_not_fail():
    guide = (
        "1. Parse the incoming command line options.\n"
        "2. Validate the requested file paths.\n"
        "3. Load the model weights into memory.\n"
        "4. Execute the forward inference loop.\n"
    )
    r = check_output(guide)
    assert r["status"] != FAIL


def test_code_block_does_not_fail():
    code = (
        "def compute_total(items: list[int]) -> int:\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item\n"
        "    return total\n"
    )
    r = check_output(code)
    assert r["status"] != FAIL


def test_fenced_code_passes():
    code = (
        "```python\n"
        "def compute_total(items: list[int]) -> int:\n"
        "    total = 0\n"
        "    for item in items:\n"
        "        total += item\n"
        "    return total\n"
        "```"
    )
    r = check_output(code)
    assert r["status"] == PASS


def test_short_valid_answer_no_entropy_fail():
    r = check_output("The answer is 42.")
    # Under 20 chars triggers too_short (tested elsewhere), but must NEVER trigger entropy or diversity failures
    assert "low_entropy" not in r["codes"]
    assert "low_char_diversity" not in r["codes"]


def test_metrics_and_codes_structure():
    r = check_output(GOOD)
    assert isinstance(r["codes"], list)
    assert isinstance(r["metrics"], dict)
    assert r["rules_version"] == "1.1"
    assert "shannon_entropy" in r["metrics"]
    assert "word_count" in r["metrics"]
