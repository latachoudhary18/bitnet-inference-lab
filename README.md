# BitNet Hardware-Aware and Reliability-Aware Inference

## Objective
Build a reproducible framework for benchmarking BitNet inference and
validating output reliability on available hardware.

## Current environment
- OS: Windows
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)
- Initial backend: CPU
- Reference implementation: https://github.com/microsoft/BitNet

## Current progress
- Set up the official BitNet repository and Python environment.
- Built the inference executables.
- Inspected the downloaded GGUF file: 332 tensors detected.
- Inspected 150 selected quantized tensor arrays.
- Recorded the initial inference failure and inspection limitations.

## Current blocker
The tested model produced repeated question marks instead of readable text.
A valid inference baseline has not yet been established.

## Next steps
1. Establish successful inference.
2. Build repeatable latency and throughput benchmarks.
3. Add automated reliability checks.
4. Compare configurations using measured results.

## Important limitation
Nonzero bytes in quantized tensor storage do not prove that decoded weights
or model inference are correct. No performance improvement is claimed yet.

## Reliability validation (branch `reliability-checker`)

**Completed**
- `scripts/reliability_checker.py`: classifies output as PASS / WARNING / FAIL with reasons
  (empty, question-mark-only, repeated text, replacement characters, too short, timeout, non-zero exit).
- `scripts/result_schema.py`: shared experiment-record schema with validation. It keeps
  execution status, performance metrics and reliability status separate.
- `tests/`: synthetic tests that do not depend on the model.
- `scripts/demo_integration.py`: safe test using harmless commands (the model is not run).
  Sample records are in `results/reliability_demo.jsonl`.
- Rules documented in `docs/reliability_rules.md`.

**Current blocker**
- I2_S output is question marks and F32 output is incoherent. The root cause is not established.
  No valid inference or performance result is claimed.

**Limitations**
- Checks are sanity heuristics, not proof of semantic correctness. Readable output can still be wrong.
- Thresholds are initial and need tuning on real model output.
- No benchmark conclusion is valid until a trustworthy baseline exists.

**Run the tests:** `python -m pytest -v`
**Run the safe demo:** `python -m scripts.demo_integration`
