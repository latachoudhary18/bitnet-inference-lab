# BitNet b1.58 – Experiment Summary

## Project
**Title:** Hardware-Aware and Reliability-Aware Inference Optimization for BitNet b1.58  
**Subtitle:** An Experiment-Logging and Reliability-Validation Framework

## 1. Experimental Environment
- OS: Windows 11
- CPU: Intel Core i5-1334U
- RAM: 15.69 GB
- Execution backend: CPU
- Model: 1bitLLM/bitnet_b1_58-large (F32 GGUF)
- Inference executable: llama-completion
- Maximum generated tokens: 32
- Temperature: 0
- Seed: 42

## 2. Methodology
Three CPU thread configurations were evaluated: 2, 4 and 8 threads.
Each configuration was tested three times using the same model, prompt and generation settings.

Prompt: "Explain what a neural network is in one sentence."

Generation throughput was used as the primary performance metric.

## 3. Performance Results

| Threads | Runs | Average Generation Speed (tokens/sec) | Standard Deviation | Change vs. 4 Threads |
|---|---:|---:|---:|---:|
| 2 | 3 | 9.853 | 0.076 | -4.34% |
| 4 | 3 | 10.300 | 0.118 | 0.00% |
| 8 | 3 | 11.070 | 0.010 | +7.48% |

The 8-thread configuration achieved the highest average throughput among the configurations tested.

## 4. Reliability Validation
- Total experiment records: 9
- Successful runs: 9
- Validation issues: 0
- Overall validation status: PASS
- Negative test: PASS

The negative test confirmed that the reliability checker detects a failed process even when its record incorrectly reports success.

## 5. Limitations
- Experiments were conducted on CPU; no usable GPU was available.
- The TL2 model configuration encountered an access violation during inference.
- The initial 2B model was incompatible with the available llama.cpp architecture support.
- F32 model generation was repetitive and incomplete; semantic output quality was not validated.
- Only three repetitions were performed per configuration.
- The optimization evaluated CPU thread configurations, not custom BitNet inference kernels.
- Results may vary with system load and hardware.

## 6. Conclusion
The experiment established a reproducible CPU benchmarking and reliability-validation workflow for BitNet b1.58. Among the tested configurations, 8 threads achieved an average generation speed of 11.07 tokens/sec, representing a 7.48% improvement over the 4-thread baseline.

All nine experiment records passed reliability validation. The results demonstrate the usefulness of systematic experiment logging, performance comparison and automated reliability checks, while the documented limitations identify areas for further investigation.

## 7. Key Deliverables
- tools/benchmark_runner.py
- tools/optimization_runner.py
- tools/optimization_runner_8t.py
- tools/reliability_checker.py
- tools/compare_results.py
- tests/test_reliability_checker.py
- results/raw/experiments.jsonl
- results/validated/validation_report.json
- results/validated/performance_comparison.json
- results/outputs/
