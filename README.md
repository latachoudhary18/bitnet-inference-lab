# BitNet Hardware-Aware and Reliability-Aware Inference Lab

A research/miniproject prototype for recording BitNet inference runs, checking
basic output reliability, and summarizing benchmark results. The framework is
intended to make failures visible rather than treating every process exit as a
successful model response.

## Current status

- The reliability checker, result schema, report, and synthetic integration demo
  are implemented.
- Unit tests cover the checker, schema, and legacy benchmark-report behavior.
- **A trustworthy successful BitNet inference baseline has not been established.**
  Experiments in `results/` include failed/unreadable output and must not be used
  to claim model correctness or a speed improvement.
- Throughput and peak-memory metrics remain unavailable unless the backend
  explicitly reports them. The runner does not currently measure CPU or memory
  utilization during inference.
- Reliability checks are heuristics (for example, empty output, repeated text,
  or suspicious question marks). A `PASS` is not a guarantee of semantic or
  factual correctness.

## Requirements

- Python 3.10 or newer is recommended.
- `pytest` to run the test suite.
- A built BitNet/llama CLI executable and a compatible GGUF model are required
  only for real inference runs. The tests and synthetic demo do not require a
  working model or GPU.

## Setup

From the repository root, create and activate a virtual environment, then
install pytest:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pytest
```

The BitNet CLI and model are external prerequisites. Build/download them using
the official [Microsoft BitNet repository](https://github.com/microsoft/BitNet)
and follow its current instructions. Paths and supported model formats depend
on that upstream runtime; this project does not bundle model weights or the CLI.

## Run tests and synthetic demo

Run the full test suite from the repository root:

```powershell
python -m pytest -q
```

Run the synthetic integration demo (does not run model inference):

```powershell
python -m scripts.demo_integration
```

The demo covers representative successful output, suspicious question marks,
process error, timeout, and missing-file cases. These are synthetic records, not
benchmark evidence.

## Primary workflow and legacy scripts

The recommended workflow for new runs is `scripts/inference_runner.py` followed
by `scripts/benchmark_report.py`. These use the reliability-aware schema and
retain legacy aliases for existing consumers.

The repository also retains `scripts/inference_check.py` and
`scripts/config_selector.py` for the earlier benchmark format. They write/read
records with fields such as `assessment.status`, `timed_out`, and
`generation_tokens_per_second`. Use these only when working with older logs;
avoid mixing their output files with new-run logs unless you intentionally want
the report's legacy compatibility path to normalize them.

## Run an inference record

Use module invocation from the repository root so Python can resolve the
`scripts` package:

```powershell
python -m scripts.inference_runner --cli "C:\path\to\llama-cli.exe" --model "C:\path\to\model.gguf" --prompt "Explain a 1-bit language model simply." --threads 4 --ctx-size 512 --predict 32 --timeout 180 --output results/inference_runs.jsonl
```

This command starts real inference and may take time. Replace both paths with
valid local paths. It is not required for running tests or the synthetic demo.
Each run is appended as one JSON object per line (JSONL).

## Generate a benchmark summary

```powershell
python -m scripts.benchmark_report --input results/inference_runs.jsonl --output results/benchmark_summary.json
```

The report includes execution-status counts, reliability counts, elapsed-time
statistics, and throughput statistics only when numeric measurements are
present. A run is eligible for fastest-run selection only if execution succeeded,
exit code is zero, reliability is `PASS`, and generation throughput was
explicitly measured.

## Result formats

Each new runner record contains:

- `schema_version`, `run_id`, and `timestamp_utc`;
- `config` describing CLI/model paths and requested settings;
- `execution_status`, `exit_code`, and `error_message`;
- `timing.total_seconds`;
- `performance.prompt_tps`, `performance.gen_tps`, and
  `performance.peak_memory_mb` (null when unavailable);
- `output.stdout` and `output.stderr`;
- `reliability.status` (`PASS`, `WARNING`, or `FAIL`) and `reliability.reasons`.

Legacy aliases are retained in runner records for compatibility. The report
also recognizes older fields such as `timed_out`, `generation_tokens_per_second`,
and `assessment.status` where present. Missing values are not estimated.

## Reproduce the recorded summary

The checked-in `results/benchmark_summary.json` is generated from the historical
`results/benchmark_log.jsonl` record using the current report code:

```powershell
python -m scripts.benchmark_report --input results/benchmark_log.jsonl --output results/benchmark_summary.json
```

That historical record is a timeout with failed heuristic assessment. The
summary is evidence of a failed run, not a performance result. Other summary
files under `results/` may refer to different input logs or earlier stages;
check their source logs before comparing counts.

## Hardware profile

Capture a system profile with:

```powershell
python -m scripts.system_info
```

The profile records detected system information and whether the configured CLI
exists. It is not a measurement of inference-time resource utilization and does
not demonstrate GPU acceleration.

## Known limitations

1. No valid successful inference baseline or optimization speedup has been
   demonstrated yet.
2. Output reliability is heuristic and does not assess semantic correctness.
3. Token throughput is not populated unless the backend provides a trustworthy
   explicit measurement; CPU and peak memory utilization are not measured by the
   runner at present.
4. GPU detection alone does not mean the inference executable was built with GPU
   support or used the GPU.
5. Legacy-format compatibility is tested, but real-world formats from all CLI
   versions may differ.

## License

This repository's original project code is licensed under the MIT License; see
[LICENSE](LICENSE). The license does not relicense Microsoft BitNet, model
weights, third-party dependencies, or other separately licensed material.
Review each upstream component's terms before redistributing it.

## Project background
See [project background](docs/project_background.md) for the goal and scope, and [setup guide](docs/setup_guide.md) for setup.
