"""Reliability checker for BitNet inference output.

Sanity checks only. They detect obviously broken output (empty, '????',
repetition loops, process failures, garbled text, extreme repetition).
They do NOT prove the text is factually or semantically correct.
Pass only the GENERATED text (stdout), not llama.cpp's stderr log lines.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any

RULES_VERSION = "1.1"

# Severity levels
PASS, WARNING, FAIL = "PASS", "WARNING", "FAIL"
_SEVERITY = {PASS: 0, WARNING: 1, FAIL: 2}

# --- Module-level Threshold Constants ---
MIN_CHARS_DEFAULT = 20
MIN_CHARS_FOR_ENTROPY = 50

# Entropy & Character Diversity
SHANNON_ENTROPY_FAIL = 2.0
SHANNON_ENTROPY_WARN = 2.8
CHAR_DIVERSITY_FAIL = 0.10
CHAR_DIVERSITY_WARN = 0.18

# Question Marks & Replacement Chars
REPLACEMENT_CHARS_FAIL = 0.50
REPLACEMENT_CHARS_WARN = 0.10
QMARK_RATIO_FAIL = 0.80
QMARK_RATIO_WARN = 0.30
ALNUM_RATIO_WARN = 0.50

# Single Char Repetition
SINGLE_CHAR_REPEAT_MIN = 5

# Repetition (Words & N-Grams)
WORD_REPETITION_MIN_WORDS = 8
WORD_UNIQUE_RATIO_FAIL = 0.30
WORD_UNIQUE_RATIO_WARN = 0.50

NGRAM_REPEAT_MIN_WORDS = 10
NGRAM_REPEAT_FAIL = 5
NGRAM_REPEAT_WARN = 3
NGRAM_COVERAGE_FAIL = 0.40
NGRAM_COVERAGE_WARN = 0.25

# Tail Loop Detection
TAIL_LOOP_MIN_WORDS = 8
TAIL_FRACTION = 0.30
TAIL_CYCLE_MIN_REPEATS_FAIL = 4
TAIL_CYCLE_MIN_REPEATS_WARN = 2

# Character N-Grams (Unspaced text)
UNSPACED_MIN_CHARS = 30
UNSPACED_SPACE_RATIO = 0.05
CHAR_NGRAM_COVERAGE_FAIL = 0.50
CHAR_NGRAM_COVERAGE_WARN = 0.30

# Garbled & Non-printable
NON_PRINTABLE_RATIO_FAIL = 0.10
NON_PRINTABLE_RATIO_WARN = 0.02
MOJIBAKE_COUNT_WARN = 3

# Truncation & Token Margin
TRUNCATION_TOKEN_MARGIN = 0.90
TERMINAL_PUNCTUATION = set(".!?:;\"'“”’)]}*`\n")

# Regex for common double-encoded UTF-8 mojibake sequences
MOJIBAKE_PATTERN = re.compile(
    r"Ã[\x80-\xbf]|â€[\x80-\xbf]|Â[\xa0-\xbf]|Ã©|Ã¨|Ã |Ã¡|Ã³|Ãº|Ã¼|Ã±|Ã§|â€™|â€œ|â€”|â€"
)
# ANSI escape sequence regex
ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def compute_shannon_entropy(text: str) -> float:
    """Compute Shannon entropy in bits per character over given text."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def check_output(
    output_text: Any,
    exit_code: int | None = 0,
    timed_out: bool = False,
    min_chars: int = MIN_CHARS_DEFAULT,
    prompt: str | None = None,
    requested_tokens: int | None = None,
) -> dict[str, Any]:
    """Classify model output sanity and extract reliability metrics.

    Parameters
    ----------
    output_text : Any
        Generated stdout from model process.
    exit_code : int | None
        Process return code (0 or None is treated as normal exit).
    timed_out : bool
        Whether the execution exceeded its timeout ceiling.
    min_chars : int
        Minimum expected non-space characters before flagging too_short.
    prompt : str | None
        Input prompt to strip from output start if echoed by the runner.
    requested_tokens : int | None
        Expected token generation budget; if reached, truncation warning is downgraded.

    Returns
    -------
    dict
        {
            "status": "PASS" | "WARNING" | "FAIL",
            "reasons": ["LEVEL:code: msg", ...],
            "codes": ["code1", ...],
            "metrics": { ... },
            "rules_version": "1.1"
        }
    """
    state = {"status": PASS}
    reasons: list[str] = []
    codes: list[str] = []

    def flag(level: str, code: str, msg: str) -> None:
        reasons.append(f"{level}:{code}: {msg}")
        codes.append(code)
        if _SEVERITY[level] > _SEVERITY[state["status"]]:
            state["status"] = level

    def info(code: str, msg: str) -> None:
        reasons.append(f"INFO:{code}: {msg}")
        codes.append(code)

    # Initial metrics scaffold
    metrics: dict[str, Any] = {
        "char_count": 0,
        "non_space_chars": 0,
        "word_count": 0,
        "unique_word_ratio": None,
        "qmark_ratio": 0.0,
        "replacement_char_ratio": 0.0,
        "alnum_ratio": 0.0,
        "shannon_entropy": None,
        "unique_char_ratio": None,
        "non_printable_ratio": 0.0,
        "mojibake_count": 0,
        "top_ngram_repeats": {},
        "tail_loop_detected": False,
        "estimated_tokens": 0,
        "requested_tokens": requested_tokens,
    }

    def result() -> dict[str, Any]:
        return {
            "status": state["status"],
            "reasons": reasons,
            "codes": codes,
            "metrics": metrics,
            "rules_version": RULES_VERSION,
        }

    # 1. Process-level checks
    if timed_out:
        flag(FAIL, "timeout", "process timed out")
    if exit_code not in (0, None):
        flag(FAIL, "nonzero_exit", f"process exited with code {exit_code}")

    # 2. Input type validation
    if output_text is not None and not isinstance(output_text, str):
        flag(FAIL, "invalid_type", f"output is {type(output_text).__name__}, not text")
        return result()

    raw_text = output_text or ""
    text = raw_text

    # Prompt stripping: remove echoed prompt if present
    if prompt:
        p_stripped = prompt.strip()
        t_stripped = text.strip()
        if p_stripped and t_stripped.startswith(p_stripped):
            idx = text.find(p_stripped)
            if idx != -1:
                text = text[idx + len(p_stripped):]
            else:
                text = t_stripped[len(p_stripped):]

    compact = re.sub(r"\s+", "", text)
    metrics["char_count"] = len(text)
    metrics["non_space_chars"] = len(compact)

    # 3. Empty output
    if not compact:
        flag(FAIL, "empty_output", "output is empty or whitespace only")
        return result()

    # 4. Replacement characters (U+FFFD - broken utf-8 decoding)
    bad_ratio = compact.count("\ufffd") / len(compact)
    metrics["replacement_char_ratio"] = round(bad_ratio, 4)
    if bad_ratio >= REPLACEMENT_CHARS_FAIL:
        flag(FAIL, "replacement_chars", f"{bad_ratio:.0%} of characters are U+FFFD")
    elif bad_ratio >= REPLACEMENT_CHARS_WARN:
        flag(WARNING, "replacement_chars", f"{bad_ratio:.0%} of characters are U+FFFD")

    # 5. Question-mark degeneration (the known BitNet I2_S defect)
    q_ratio = compact.count("?") / len(compact)
    metrics["qmark_ratio"] = round(q_ratio, 4)
    if q_ratio == 1.0:
        flag(FAIL, "only_question_marks", "output contains only '?' characters")
    elif q_ratio >= QMARK_RATIO_FAIL:
        flag(FAIL, "mostly_question_marks", f"{q_ratio:.0%} of characters are '?'")
    elif q_ratio >= QMARK_RATIO_WARN:
        flag(WARNING, "many_question_marks", f"{q_ratio:.0%} of characters are '?'")
    else:
        alnum_ratio = sum(c.isalnum() for c in compact) / len(compact)
        metrics["alnum_ratio"] = round(alnum_ratio, 4)
        if alnum_ratio < ALNUM_RATIO_WARN:
            flag(WARNING, "low_text_content",
                 f"only {alnum_ratio:.0%} of characters are letters/digits")

    # 6. Single character repetition across entire output
    if len(compact) >= SINGLE_CHAR_REPEAT_MIN and len(set(compact)) == 1 and compact[0] != "?":
        flag(FAIL, "single_char_repeat", f"output is just '{compact[0]}' repeated")

    # 7. Suspiciously short output
    if len(compact) < min_chars:
        flag(WARNING, "too_short", f"only {len(compact)} non-space characters")

    # 8. Garbled, Non-printable, ANSI escapes, NUL bytes, and Mojibake
    if "\x00" in text:
        flag(FAIL, "nul_byte", "output contains NUL byte")

    ansi_matches = ANSI_ESCAPE_PATTERN.findall(text)
    if ansi_matches:
        flag(WARNING, "ansi_escape", f"output contains {len(ansi_matches)} ANSI escape sequences")

    non_printable = sum(
        1 for c in text
        if c not in ("\n", "\r", "\t") and unicodedata.category(c) in ("Cc", "Cf", "Co", "Cn")
    )
    np_ratio = non_printable / len(text) if text else 0.0
    metrics["non_printable_ratio"] = round(np_ratio, 4)
    if np_ratio >= NON_PRINTABLE_RATIO_FAIL:
        flag(FAIL, "non_printable_chars", f"{np_ratio:.1%} of characters are control/unassigned Unicode")
    elif np_ratio >= NON_PRINTABLE_RATIO_WARN:
        flag(WARNING, "non_printable_chars", f"{np_ratio:.1%} of characters are control/unassigned Unicode")

    mojibake_matches = len(MOJIBAKE_PATTERN.findall(text))
    metrics["mojibake_count"] = mojibake_matches
    if mojibake_matches >= MOJIBAKE_COUNT_WARN:
        flag(WARNING, "mojibake", f"detected {mojibake_matches} probable mojibake character sequences")

    # 9. Shannon Entropy and Character Diversity (guarded by minimum length)
    if len(compact) >= MIN_CHARS_FOR_ENTROPY:
        entropy = compute_shannon_entropy(compact)
        unique_char_ratio = len(set(compact)) / len(compact)
        metrics["shannon_entropy"] = round(entropy, 4)
        metrics["unique_char_ratio"] = round(unique_char_ratio, 4)

        if entropy < SHANNON_ENTROPY_FAIL:
            flag(FAIL, "low_entropy", f"Shannon entropy {entropy:.2f} bits/char is extremely low")
        elif entropy < SHANNON_ENTROPY_WARN:
            flag(WARNING, "low_entropy", f"Shannon entropy {entropy:.2f} bits/char is low")

        if unique_char_ratio < CHAR_DIVERSITY_FAIL:
            flag(FAIL, "low_char_diversity", f"unique character ratio {unique_char_ratio:.2f} is extremely low")
        elif unique_char_ratio < CHAR_DIVERSITY_WARN:
            flag(WARNING, "low_char_diversity", f"unique character ratio {unique_char_ratio:.2f} is low")

    # 10. Unspaced text character n-gram loops
    space_count = sum(c.isspace() for c in text)
    if len(text) >= UNSPACED_MIN_CHARS and (space_count / len(text)) < UNSPACED_SPACE_RATIO:
        worst_char_loop: tuple[str, str, int, float] | None = None
        for k in range(2, 9):
            if len(compact) >= k * 3:
                subs = [compact[i : i + k] for i in range(len(compact) - k + 1)]
                if subs:
                    top_sub, count = Counter(subs).most_common(1)[0]
                    coverage = (count * k) / len(compact)
                    if count >= 4 and coverage >= CHAR_NGRAM_COVERAGE_FAIL:
                        worst_char_loop = (FAIL, top_sub, count, coverage)
                        break
                    elif count >= 3 and coverage >= CHAR_NGRAM_COVERAGE_WARN:
                        if worst_char_loop is None or worst_char_loop[0] != FAIL:
                            worst_char_loop = (WARNING, top_sub, count, coverage)
        if worst_char_loop:
            lvl, sub, count, cov = worst_char_loop
            flag(lvl, "char_ngram_loop",
                 f"unspaced text repeats '{sub}' {count} times (coverage {cov:.0%})")

    # 11. Word Tokenization & Repetition (General N-Grams n=2..6, Vocab, Lines)
    words = re.findall(r"\w+", text.lower())
    metrics["word_count"] = len(words)

    if len(words) >= WORD_REPETITION_MIN_WORDS:
        unique_word_ratio = len(set(words)) / len(words)
        metrics["unique_word_ratio"] = round(unique_word_ratio, 4)
        if unique_word_ratio < WORD_UNIQUE_RATIO_FAIL:
            flag(FAIL, "repetitive_words", f"unique-word ratio {unique_word_ratio:.2f}")
        elif unique_word_ratio < WORD_UNIQUE_RATIO_WARN:
            flag(WARNING, "repetitive_words", f"unique-word ratio {unique_word_ratio:.2f}")

    # N-gram repetitions (n=2..6)
    top_ngrams: dict[str, int] = {}
    if len(words) >= NGRAM_REPEAT_MIN_WORDS:
        for n in range(2, 7):
            if len(words) >= n * 2:
                ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
                if ngrams:
                    top_phrase, count = Counter(ngrams).most_common(1)[0]
                    top_ngrams[f"{n}gram"] = count
                    coverage = (count * n) / len(words)
                    code_name = "repeated_phrase" if n == 3 else f"repeated_{n}gram"

                    if count >= NGRAM_REPEAT_FAIL and coverage >= NGRAM_COVERAGE_FAIL:
                        flag(FAIL, code_name,
                             f"a {n}-word phrase repeats {count} times (coverage {coverage:.0%})")
                    elif count >= NGRAM_REPEAT_WARN and coverage >= NGRAM_COVERAGE_WARN:
                        flag(WARNING, code_name,
                             f"a {n}-word phrase repeats {count} times (coverage {coverage:.0%})")
                    elif n == 3 and count >= 5:
                        flag(FAIL, "repeated_phrase", f"a 3-word phrase repeats {count} times")
                    elif n == 3 and count >= 3:
                        flag(WARNING, "repeated_phrase", f"a 3-word phrase repeats {count} times")
    metrics["top_ngram_repeats"] = top_ngrams

    # Tail Loop Detection (cycle in the final ~30% of text)
    if len(words) >= TAIL_LOOP_MIN_WORDS:
        tail_len = max(6, int(len(words) * TAIL_FRACTION))
        tail = words[-tail_len:]
        tail_cycle_found = False

        for p in range(1, (tail_len // 2) + 1):
            repeats = 0
            for i in range(p, tail_len):
                if tail[i] == tail[i - p]:
                    repeats += 1
            match_ratio = repeats / (tail_len - p)
            if match_ratio >= 0.85:
                cycle_count = tail_len // p
                if cycle_count >= TAIL_CYCLE_MIN_REPEATS_FAIL:
                    flag(FAIL, "tail_loop", f"final {tail_len} words repeat a {p}-word cycle {cycle_count} times")
                    tail_cycle_found = True
                    break
                elif cycle_count >= TAIL_CYCLE_MIN_REPEATS_WARN:
                    flag(WARNING, "tail_loop", f"final {tail_len} words repeat a {p}-word cycle {cycle_count} times")
                    tail_cycle_found = True
                    break
        metrics["tail_loop_detected"] = tail_cycle_found

    # Repeated lines
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 4:
        top_line_count = Counter(lines).most_common(1)[0][1]
        if top_line_count / len(lines) >= 0.5:
            flag(FAIL, "repeated_lines", f"{top_line_count} of {len(lines)} lines are identical")

    # 12. Truncation & Token-Cap Handling
    tokens = re.findall(r"\w+|[^\w\s]", text)
    metrics["estimated_tokens"] = len(tokens)

    stripped = text.rstrip()
    if len(stripped) >= 5 and stripped[-1] not in TERMINAL_PUNCTUATION:
        # Text ends without terminal punctuation
        if requested_tokens is not None and len(tokens) >= int(requested_tokens * TRUNCATION_TOKEN_MARGIN):
            info("token_cap_reached",
                 f"output ends without terminal punctuation at requested token cap (~{len(tokens)}/{requested_tokens} tokens)")
        else:
            flag(WARNING, "truncated_output", "output ends mid-sentence without terminal punctuation")

    return result()


if __name__ == "__main__":
    print(check_output("?????????????"))
