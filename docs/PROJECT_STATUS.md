# Project Status and Validation Boundary

## What is implemented

- A legacy inference-checking path (`inference_check.py`) and configuration selector (`config_selector.py`).
- A reliability-aware inference runner, versioned result schema, heuristic reliability checker, and JSONL benchmark report.
- Hardware/system information capture (`system_info.py`).
- Unit tests and a synthetic integration demo that do not require a working model or GPU.

## Validation performed on this snapshot

- Full automated test suite: 38 tests passed.
- Synthetic integration scenarios cover good-looking output, suspicious question marks, process error, timeout, and missing executable/model.
- The benchmark summary was regenerated from `results/benchmark_log.jsonl` with the current `scripts/benchmark_report.py`.

These results validate software paths exercised by the tests and synthetic data. They do not prove every external CLI/model combination works.

## Observed inference status

The checked-in historical benchmark record timed out after approximately 301.6 seconds and its output failed heuristic reliability checks. A separate recorded run timed out after approximately 180.1 seconds. No successful, trustworthy inference baseline has been established from the available records.

## Claims not supported by current evidence

- No inference speedup or best-performing configuration has been demonstrated.
- Token throughput and peak memory are unavailable where not explicitly measured by the backend.
- The runner does not currently measure inference-time CPU utilization or memory utilization.
- Detecting a GPU does not prove that the CLI was compiled with GPU support or used the GPU.
- A reliability `PASS` means only that the configured heuristics did not flag the output; it does not establish semantic correctness.

## Reproduction commands

From the repository root, with Python and pytest installed:

```powershell
python -m pytest -q
python -m scripts.demo_integration
python -m scripts.benchmark_report --input results/benchmark_log.jsonl --output results/benchmark_summary.json
python -m scripts.system_info
```

The first three commands do not require running model inference. The hardware profile command reads the current host environment and writes a profile; run it only when you want to refresh the environment-specific record.

## Open-source and contribution status

This repository is a public project with an MIT license for original project code, subject to contributor rights and third-party licensing terms. It is not evidence that a change has been accepted into Microsoft's upstream BitNet repository. An upstream contribution would require a separate pull request and review.
