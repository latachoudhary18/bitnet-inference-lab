"""Reliability checker for BitNet inference output.

Sanity checks only. They detect obviously broken output (empty, '????',
repetition loops, process failures). They do NOT prove the text is correct.
Pass only the GENERATED text (stdout), not llama.cpp's stderr log lines.
"""
import re
from collections import Counter

PASS, WARNING, FAIL = "PASS", "WARNING", "FAIL"
_SEVERITY = {PASS: 0, WARNING: 1, FAIL: 2}


def check_output(output_text, exit_code=0, timed_out=False, min_chars=20, prompt=None):
    """Classify model output.

    Returns: {"status": "PASS"|"WARNING"|"FAIL", "reasons": [str, ...]}
    """
    state = {"status": PASS}
    reasons = []

    def flag(level, code, msg):
        reasons.append(f"{level}:{code}: {msg}")
        if _SEVERITY[level] > _SEVERITY[state["status"]]:
            state["status"] = level

    def result():
        return {"status": state["status"], "reasons": reasons}

    # 1. Process-level problems
    if timed_out:
        flag(FAIL, "timeout", "process timed out")
    if exit_code not in (0, None):
        flag(FAIL, "nonzero_exit", f"process exited with code {exit_code}")

    # 2. Normalise input
    if output_text is not None and not isinstance(output_text, str):
        flag(FAIL, "invalid_type", f"output is {type(output_text).__name__}, not text")
        return result()
    text = output_text or ""

    # Remove the echoed prompt so it can't hide garbage output
    if prompt and text.strip().startswith(prompt.strip()):
        text = text.strip()[len(prompt.strip()):]

    # 3. Empty output
    compact = re.sub(r"\s+", "", text)
    if not compact:
        flag(FAIL, "empty_output", "output is empty or whitespace only")
        return result()

    # 4. Replacement characters (broken decoding)
    bad = compact.count("\ufffd") / len(compact)
    if bad >= 0.5:
        flag(FAIL, "replacement_chars", f"{bad:.0%} of characters are U+FFFD")
    elif bad >= 0.1:
        flag(WARNING, "replacement_chars", f"{bad:.0%} of characters are U+FFFD")

    # 5. Question-mark garbage (the I2_S symptom)
    q_ratio = compact.count("?") / len(compact)
    if q_ratio == 1.0:
        flag(FAIL, "only_question_marks", "output contains only '?' characters")
    elif q_ratio >= 0.8:
        flag(FAIL, "mostly_question_marks", f"{q_ratio:.0%} of characters are '?'")
    elif q_ratio >= 0.3:
        flag(WARNING, "many_question_marks", f"{q_ratio:.0%} of characters are '?'")
    else:
        alnum_ratio = sum(c.isalnum() for c in compact) / len(compact)
        if alnum_ratio < 0.5:
            flag(WARNING, "low_text_content",
                 f"only {alnum_ratio:.0%} of characters are letters/digits")

    # 6. Single repeated character
    if len(compact) >= 5 and len(set(compact)) == 1 and compact[0] != "?":
        flag(FAIL, "single_char_repeat", f"output is just '{compact[0]}' repeated")

    # 7. Suspiciously short
    if len(compact) < min_chars:
        flag(WARNING, "too_short", f"only {len(compact)} non-space characters")

    # 8. Repetition
    words = re.findall(r"\w+", text.lower())
    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            flag(FAIL, "repetitive_words", f"unique-word ratio {unique_ratio:.2f}")
        elif unique_ratio < 0.5:
            flag(WARNING, "repetitive_words", f"unique-word ratio {unique_ratio:.2f}")

    if len(words) >= 12:
        top = max(Counter(zip(words, words[1:], words[2:])).values())
        if top >= 5:
            flag(FAIL, "repeated_phrase", f"a 3-word phrase repeats {top} times")
        elif top >= 3:
            flag(WARNING, "repeated_phrase", f"a 3-word phrase repeats {top} times")

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 4:
        top_line = Counter(lines).most_common(1)[0][1]
        if top_line / len(lines) >= 0.5:
            flag(FAIL, "repeated_lines", f"{top_line} of {len(lines)} lines are identical")

    return result()


if __name__ == "__main__":
    print(check_output("?????????????"))