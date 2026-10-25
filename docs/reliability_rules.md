# Reliability Check Rules

`scripts/reliability_checker.py` exposes:

    check_output(output_text, exit_code=0, timed_out=False, min_chars=20, prompt=None)

It returns `{"status": "PASS" | "WARNING" | "FAIL", "reasons": [...]}`.
The overall status is the worst status triggered, and every triggered reason is listed.

## Rules

| Condition | Status |
|---|---|
| Timeout or non-zero exit code | FAIL |
| Output is not text (e.g. a number) | FAIL |
| Empty or whitespace-only output (after removing the echoed prompt) | FAIL |
| Only `?` characters, or >= 80% `?` | FAIL |
| 30-80% `?` characters | WARNING |
| >= 50% U+FFFD replacement characters | FAIL |
| 10-50% U+FFFD replacement characters | WARNING |
| One character repeated (e.g. `aaaa`) | FAIL |
| Fewer than 20 non-space characters | WARNING |
| Unique-word ratio < 0.3 (>= 8 words) | FAIL |
| Unique-word ratio < 0.5 (>= 8 words) | WARNING |
| Same 3-word phrase repeated >= 5 times (>= 12 words) | FAIL |
| Same 3-word phrase repeated 3-4 times | WARNING |
| >= 50% of lines identical (>= 4 lines) | FAIL |
| Less than 50% letters/digits | WARNING |

## Input rules
- Pass **stdout only**. llama.cpp prints loading and timing logs to stderr; including them
  would make broken output look valid.
- Pass `prompt=` so an echoed prompt is stripped and cannot hide garbage output.

## Shared record schema
Defined and validated in `scripts/result_schema.py`. Each record keeps three things separate:
- `execution_status`: success / timeout / error / missing_file (did the process run?)
- `performance`: prompt_tps, gen_tps, peak_memory_mb (null until a valid baseline exists)
- `reliability`: status and reasons (does the output look sane?)

## Limitations
- These are sanity checks, not proof of correctness.
- Readable, fluent output can still be factually wrong or irrelevant to the prompt.
  A test (`test_readable_but_wrong_still_passes`) documents this on purpose.
- Thresholds are initial heuristics and need tuning on real model output.
- Execution success is not output quality. A run can exit with code 0 and still FAIL reliability.
- No performance or correctness conclusion is valid until the I2_S / F32 output problem is diagnosed.