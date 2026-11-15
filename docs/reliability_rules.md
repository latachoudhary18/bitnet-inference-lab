# Reliability Check Rules

`scripts/reliability_checker.py` exposes:

```python
check_output(
    output_text,
    exit_code=0,
    timed_out=False,
    min_chars=20,
    prompt=None,
    requested_tokens=None,
)
```

It returns:
```json
{
  "status": "PASS | WARNING | FAIL",
  "reasons": ["LEVEL:code: message", "..."],
  "codes": ["code1", "..."],
  "metrics": { ... },
  "rules_version": "1.1"
}
```

The overall status is the worst status triggered, and every triggered reason and code is recorded.

## Rules Summary (v1.1)

| Rule Category | Condition | Status | Reason Code |
|---|---|---|---|
| **Process** | Timeout or non-zero exit code | FAIL | `timeout`, `nonzero_exit` |
| **Input Type** | Output is not text (e.g. a number) | FAIL | `invalid_type` |
| **Completeness** | Empty or whitespace-only output (after prompt stripping) | FAIL | `empty_output` |
| **Degenerate Output (Question Marks)** | Only `?` characters, or $\ge 80\%$ `?` | FAIL | `only_question_marks`, `mostly_question_marks` |
| **Degenerate Output (Question Marks)** | $30\% - 79\%$ `?` characters | WARNING | `many_question_marks` |
| **Encoding** | $\ge 50\%$ U+FFFD replacement characters | FAIL | `replacement_chars` |
| **Encoding** | $10\% - 49\%$ U+FFFD replacement characters | WARNING | `replacement_chars` |
| **Repetition** | Single character repeated across output ($\ge 5$ chars) | FAIL | `single_char_repeat` |
| **Length** | Fewer than 20 non-space characters | WARNING | `too_short` |
| **Garbled / Encoding** | Output contains NUL byte (`\x00`) | FAIL | `nul_byte` |
| **Garbled / ANSI** | ANSI terminal escape sequences detected | WARNING | `ansi_escape` |
| **Garbled / Control** | $\ge 10\%$ control/unassigned Unicode categories | FAIL | `non_printable_chars` |
| **Garbled / Control** | $2\% - 9\%$ control/unassigned Unicode categories | WARNING | `non_printable_chars` |
| **Garbled / Mojibake** | $\ge 3$ probable double-encoded UTF-8 sequences | WARNING | `mojibake` |
| **Low Diversity** | Shannon entropy $< 2.0$ bits/char ($\ge 50$ chars) | FAIL | `low_entropy` |
| **Low Diversity** | Shannon entropy $2.0 - 2.8$ bits/char ($\ge 50$ chars) | WARNING | `low_entropy` |
| **Low Diversity** | Unique character ratio $< 0.10$ ($\ge 50$ chars) | FAIL | `low_char_diversity` |
| **Low Diversity** | Unique character ratio $0.10 - 0.18$ ($\ge 50$ chars) | WARNING | `low_char_diversity` |
| **Repetition (Words)** | Unique-word ratio $< 0.30$ ($\ge 8$ words) | FAIL | `repetitive_words` |
| **Repetition (Words)** | Unique-word ratio $0.30 - 0.49$ ($\ge 8$ words) | WARNING | `repetitive_words` |
| **Repetition (N-Grams)**| Word n-gram ($n=2..6$) repeats $\ge 5$ times & coverage $\ge 40\%$ | FAIL | `repeated_phrase` ($n=3$) / `repeated_{n}gram` |
| **Repetition (N-Grams)**| Word n-gram ($n=2..6$) repeats $\ge 3$ times & coverage $\ge 25\%$ | WARNING | `repeated_phrase` ($n=3$) / `repeated_{n}gram` |
| **Tail Loop** | Cycle in trailing $\sim 30\%$ repeats $\ge 4$ times | FAIL | `tail_loop` |
| **Tail Loop** | Cycle in trailing $\sim 30\%$ repeats $2 - 3$ times | WARNING | `tail_loop` |
| **Repetition (Unspaced)**| Unspaced text substring repeats $\ge 4$ times (coverage $\ge 50\%$) | FAIL | `char_ngram_loop` |
| **Repetition (Unspaced)**| Unspaced text substring repeats $\ge 3$ times (coverage $\ge 30\%$) | WARNING | `char_ngram_loop` |
| **Repetition (Lines)** | $\ge 50\%$ of lines identical ($\ge 4$ lines) | FAIL | `repeated_lines` |
| **Character Density** | Less than 50% letters/digits | WARNING | `low_text_content` |
| **Truncation** | Ends mid-word or mid-sentence without terminal punctuation | WARNING | `truncated_output` |
| **Truncation** | Ends without punctuation at token budget ($\ge 90\%$ of `requested_tokens`) | INFO (PASS) | `token_cap_reached` |

## Input Rules
- Pass **stdout only**. llama.cpp prints loading and timing logs to stderr; including them
  would make broken output look valid.
- Pass `prompt=` so an echoed prompt is stripped and cannot hide garbage output.
- Pass `requested_tokens=` so planned truncation is recorded as an informational note rather than a warning.

## Shared Record Schema (v1.1)
Defined and validated in `scripts/result_schema.py`:
- `schema_version`: "1.0" or "1.1"
- `data_source`: "real" or "synthetic"
- `benchmark`: `{is_warmup, repeat_index, seed, temperature, config_id}`
- `execution_status`: success / timeout / error / missing_file
- `performance`: prompt_tps, gen_tps, peak_memory_mb (null until a valid baseline exists)
- `reliability`: status, reasons, codes, metrics, rules_version

## Limitations
- These are sanity checks, not proof of factual correctness.
- Readable, fluent output can still be factually wrong or irrelevant to the prompt.
  A test (`test_readable_but_wrong_still_passes`) documents this on purpose.
- Execution success is not output quality. A run can exit with code 0 and still FAIL reliability.
- No performance or correctness conclusion is valid until the I2_S / F32 output problem is diagnosed.
