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