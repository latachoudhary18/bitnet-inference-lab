# Collaboration Guide for Neeraja & Lata (Phases 1, 5, 6, 7)

**Repository:** `bitnet-inference-lab`  
**Latest Push to `main`:** Commit `ef29ebc` (Phase 4 complete, 75/75 tests passing)  
**Deadline:** 1:00 PM Tomorrow  

---

## 1. Lineup Verification & Current Status

| Phase | Task | Owner | Status | Details |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Inference diagnosis | Neeraja | **UP NEXT** | Ready for execution. Verified flags and runner available. |
| **2** | Benchmark harness | Lata | **COMPLETED & INTEGRATED** | Implemented in `scripts/benchmark_sweep.py` (commit `fb383ff`). Tested in `tests/test_benchmark_sweep.py`. |
| **3** | Reliability checker upgrade | Shreya | **COMPLETED & MERGED** | Upgraded to v1.1 in `scripts/reliability_checker.py` and `scripts/result_schema.py` (commit `3a89078`). |
| **4** | Runner update | Shreya | **COMPLETED & MERGED** | Implemented in `scripts/inference_runner.py` (commit `ef29ebc`). 15 new tests in `tests/test_inference_runner.py`. |
| **5** | Ranking and results reporting | Lata | **UP NEXT** | Update selector / ranking logic for Schema v1.1 and generate table. |
| **6** | Repo cleanup & fresh clone test | Neeraja | **PENDING** | Clean stray scripts, fix hardcoded paths, add `.gitignore`, fresh clone test. |
| **7** | Final documentation & status report | All Three | **PENDING** | Each owner documents their area (Inference diagnosis, Benchmark notes, README). |

---

## 2. Guide for Neeraja (Phase 1 & Phase 6)

### Phase 1: Inference Diagnosis
**Goal:** Determine why BitNet inference is failing (I2_S producing `?` characters and F32 producing incoherent text), compare F32 vs I2_S under identical settings, and log all findings.

#### Crucial Findings from Phase 4 (Save Time!)
1. **Do NOT use `--no-conversation`**: `llama-cli.exe` rejects this flag (`error: unknown argument: --no-conversation`).
2. **Verified Non-Interactive Flag**: Use `--single-turn` (or shorthand `-st`).
3. **Prevent Prompt Hangs**: Always ensure `stdin=subprocess.DEVNULL` is passed if writing custom scripts (already built into `scripts/inference_runner.py`).
4. **Clean Flags Implemented**:
   - `--simple-io`: Disables fancy terminal escape sequences.
   - `--no-display-prompt`: Omits echoing the input prompt into stdout.
   - `--color off`: Suppresses ANSI color codes.
   - `--n-gpu-layers 0` (`-ngl 0`): Forces CPU inference.

#### How to Run Diagnostic Comparisons
Use the newly updated runner directly from the terminal or PowerShell:

```powershell
# 1. Test I2_S GGUF model with fixed seed, zero temperature, short predict length
python -m scripts.inference_runner `
  --cli "path/to/llama-cli.exe" `
  --model "models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf" `
  --prompt "The capital of France is" `
  --predict 16 `
  --temp 0 `
  --seed 42 `
  --timeout 60

# 2. Test F32 GGUF model with identical parameters
python -m scripts.inference_runner `
  --cli "path/to/llama-cli.exe" `
  --model "models/BitNet-b1.58-2B-4T-gguf/ggml-model-f32.gguf" `
  --prompt "The capital of France is" `
  --predict 16 `
  --temp 0 `
  --seed 42 `
  --timeout 60
```

Every run automatically appends a Schema v1.1 record to `results/inference_runs.jsonl`.

#### Diagnosis Checklist
- [ ] Check if the `?` issue in I2_S is caused by the BitNet quantizer mismatch / kernel issue (refer to upstream issue [microsoft/BitNet#608](https://github.com/microsoft/BitNet/issues/608)).
- [ ] Check if `llama-cli` prints warning/error lines in stderr (e.g. tokenizer vocab size mismatch, missing tensor).
- [ ] Compare F32 output vs I2_S output for coherence on standard prompts.
- [ ] Log findings in `results/inference_diagnosis_report.md` (or update `results/baseline_status.txt` and `results/f32_experiment.txt`).
- [ ] **Crucial Rule:** If inference fails, report it truthfully. Never fabricate a performance baseline or claim speedups on garbled text.

---

### Phase 6: Repo Cleanup & Fresh-Checkout Test
**Goal:** Clean stray artifacts, eliminate machine-specific paths, ensure strict `.gitignore`, and verify a clean-slate checkout passes tests.

#### Concrete Items to Fix
1. **Remove Hardcoded Machine Paths**:
   - `scripts/system_info.py` line 16 has a hardcoded path:
     ```python
     DEFAULT_BITNET_CLI = Path(r"C:\Users\Lata\source\repos\BitNet\build\bin\Release\llama-cli.exe")
     ```
     Change this to use `os.environ.get("BITNET_CLI", ...)` with a generic repository-relative fallback.
2. **Clean Stray Files**:
   - `results/inspect_model.py`: Move this script to `scripts/inspect_model.py` or `tools/inspect_model.py`.
3. **Update `.gitignore`**:
   Add the following patterns to ensure no binary or heavy files are committed:
   ```gitignore
   # Models and weights
   *.gguf
   *.bin
   models/

   # Build outputs and executables
   build/
   bin/
   Release/
   Debug/
   *.exe
   *.dll
   *.lib

   # Temporary test logs
   results/sweep_runs.jsonl
   results/*_test.jsonl
   ```
4. **Fresh-Clone Verification**:
   Clone the repository into a clean temporary directory on your machine:
   ```powershell
   git clone https://github.com/latachoudhary18/bitnet-inference-lab.git test-clone
   cd test-clone
   python -m pytest -v
   ```
   Confirm all 75 tests pass without requiring any `.gguf` weights or `llama-cli.exe`.

---

## 3. Guide for Lata (Phase 2 & Phase 5)

### Phase 2 State & Alignment
- **Status:** Phase 2 was completed in `scripts/benchmark_sweep.py` and tested in `tests/test_benchmark_sweep.py`.
- **Schema v1.1 Alignment Note:**
  In `scripts/benchmark_sweep.py`:
  - `run_inference(...)` from `scripts.inference_runner` now directly accepts `benchmark={...}` and `data_source="real"` or `"synthetic"`.
  - In Schema v1.1, `data_source` must be strictly `"real"` or `"synthetic"` (avoid custom strings like `"real_cli_attempt"`, as Schema v1.1 validation expects one of the two standard values).

---

### Phase 5: Ranking and Results Reporting
**Goal:** Write a configuration ranking function that only considers reliable runs, and generate a clean results table from the sweep.

#### 1. What Needs Updating in Ranking / Selection
The existing selector in `scripts/config_selector.py` is based on legacy pre-v1.1 fields:
```python
# Old legacy check in config_selector.py:
assessment.get("status") == "pass_heuristic"
```
Under **Schema v1.1**, the reliability gate is:
```python
record["reliability"]["status"] == "PASS"
```

#### 2. Strict Ranking Gate Requirements
A run is only eligible for ranking if:
1. `record.get("execution_status") == "success"`
2. `record.get("exit_code") == 0`
3. `record.get("reliability", {}).get("status") == "PASS"` (runs with `"WARNING"` or `"FAIL"` must **never** be ranked as optimal)
4. `record.get("benchmark", {}).get("is_warmup") is False` (warmups are discarded)

#### 3. Selection Metric
- Sort eligible configurations by highest generation throughput:
  `record.get("performance", {}).get("gen_tps")` (or lowest `record.get("timing", {}).get("total_seconds")` if TPS was unmeasured).
- If no configurations pass reliability, the ranking function must return `None` (or indicate that no configuration is verified reliable).

#### 4. Synthetic Results Labeling
- If `record.get("data_source") == "synthetic"` or if fake stubs were used:
  The summary table and report MUST explicitly display a `[SYNTHETIC / STUB]` badge/label.
- Never present synthetic test numbers as real hardware benchmarks.

#### 5. Using the Fake CLI Stub for Phase 5 Testing
You do not need real weights or native binaries to build and test Phase 5!
You can use `tests/fake_llama_cli.py`:
```powershell
# Run a synthetic sweep with the fake CLI:
python -m scripts.benchmark_sweep `
  --cli "fake_cli" `
  --model "fake_model.gguf" `
  --repetitions 2
```
Or run `tests/test_inference_runner.py` to see how `fake_llama_cli.py` is invoked with `command_prefix=[sys.executable, str(stub_path)]`.

---

## 4. Shared Data Contract (Schema v1.1 & Reliability Rules)

Every record produced in this repository must conform to Schema v1.1 ([`scripts/result_schema.py`](file:///d:/bitnet/bitnet-inference-lab/scripts/result_schema.py)).

### Record Structure
```json
{
  "schema_version": "1.1",
  "run_id": "uuid-v4",
  "timestamp_utc": "ISO-8601",
  "data_source": "real | synthetic",
  "config": {
    "cli": "path/to/llama-cli.exe",
    "model": "path/to/model.gguf",
    "threads": 4,
    "ctx_size": 512,
    "prompt": "...",
    "requested_tokens": 16,
    "timeout_seconds": 30,
    "temp": 0.0,
    "seed": 42
  },
  "benchmark": {
    "is_warmup": false,
    "repeat_index": 0,
    "seed": 42,
    "temperature": 0.0,
    "config_id": "threads_4_ctx_512"
  },
  "execution_status": "success | timeout | error | missing_file",
  "exit_code": 0,
  "error_message": null,
  "timing": {
    "total_seconds": 1.234
  },
  "performance": {
    "prompt_tps": 24.5,
    "gen_tps": 12.1,
    "peak_memory_mb": null
  },
  "output": {
    "stdout": "generated text",
    "stderr": "diagnostic logs"
  },
  "reliability": {
    "status": "PASS | WARNING | FAIL",
    "reasons": [],
    "codes": [],
    "metrics": { ... }
  }
}
```

### Reliability Rules Reference (v1.1)
- `repeated_ngram`: Flagged when $n$-gram loops ($n=2..6$) exceed repetition thresholds (`FAIL`).
- `tail_loop`: Detected when generated text ends in repetitive trailing cycles (`FAIL`).
- `unspaced_char_loop`: Flagged when non-space characters repeat with high coverage (`FAIL` or `WARNING`).
- `low_entropy`: Shannon entropy below $2.0$ on text length $\ge 50$ (`FAIL`), or below $2.8$ (`WARNING`).
- `garbled_output`: NUL bytes (`FAIL`), replacement characters `\ufffd` (`FAIL`), ANSI escapes (`WARNING`), mojibake (`WARNING`).
- `empty_output`: No text or whitespace only (`FAIL`).
- `truncated_output`: Mid-sentence truncation or token cap reached (`WARNING` or `INFO`).

---

## 5. Verification Commands

Run these at any time to verify the codebase:

```powershell
# Run full automated test suite (all 75 tests)
python -m pytest -v

# Run synthetic integration demo
python -m scripts.demo_integration

# Run single dry-run inference command check
python -m scripts.inference_runner --dry-run
```

All team members are encouraged to pull the latest `main` branch before starting their respective phases!
