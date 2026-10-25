# BitNet Hardware- and Reliability-Aware Inference Lab

## Goal
This project builds a repeatable harness for running BitNet inference experiments, recording performance, applying lightweight output-reliability heuristics, and selecting a configuration only when it passes the heuristic gate.

## Current experimental boundary
The current local environment has CPU-only inference enabled (`n_gpu_layers=0`). Preliminary runs with the supplied F32 and I2_S GGUF files produced incoherent output. These are recorded as an unresolved inference/model/runtime limitation, not as a successful model-quality result. Speed measurements from these preliminary runs are exploratory and should not be treated as validated benchmarks.

## Framework components
- `scripts/inference_check.py`: launches the existing `llama-cli.exe`, captures output and timing, applies simple output checks, and appends a JSONL record.
- `scripts/config_selector.py`: selects the fastest run only among runs that pass the current heuristic gate; returns no selection if none pass.
- `tests/test_framework.py`: unit tests that run without loading a language model.
- `results/benchmark_log.jsonl`: append-only experiment log created by the runner.

## Reliability checks and limitations
The checker flags empty output, output dominated by question marks, very short output, and highly repetitive text. These are screening heuristics, not a semantic correctness benchmark. A fluent but factually incorrect response may pass; valid unusual output may be flagged. A stronger evaluation should add fixed prompts, expected-answer checks or an established benchmark.

## Run unit tests
From the project root, with Python 3.10+:
```powershell
python -m unittest discover -s tests -v
```

## Run an inference experiment
Example from PowerShell, adapting paths to your machine:
```powershell
python scripts/inference_check.py `
  --cli "$env:USERPROFILE\source\repos\BitNet\build\bin\Release\llama-cli.exe" `
  --model "$env:USERPROFILE\source\repos\BitNet\models\BitNet-b1.58-2B-4T\BitNet-b1.58-2B-4T\ggml-model-f32.gguf" `
  --prompt "The capital of France is" `
  --threads 4 --ctx-size 128 --n-predict 8
```
The runner writes one JSON object per line. It stores raw CLI output to support auditability. Do not interpret a successful process exit code as proof that the generated text is correct.

## Current status
Implemented scaffold; must be tested on the target Windows environment. No configuration should be described as optimized until repeated runs and a reliability gate support that claim. GPU-aware experiments are future work until a working GPU inference path is established.
