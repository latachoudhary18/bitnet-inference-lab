\# Inference Diagnosis Log (Phase 1)



Build: BitNet 0b341e5, llama.cpp 390c30775, Clang 22.1.3, static libs (-DBUILD\_SHARED\_LIBS=OFF).

Model: official BitNet-b1.58-2B-4T I2\_S. Machine: i5-1334U, CPU only.

Common settings: -t 4 -c 128 --temp 0 --seed 42 -no-cnv (via llama-completion.exe)



| # | Prompt | n | Exit | Output | Result |

|---|---|---|---|---|---|

| 1 | The capital of France is | 16 | 0 | "...is a small city, and the capital of France..." | readable, wrong |

| 2 | The capital of France is (repeat) | 64 | 0 | same clause repeated | deterministic (identical hashes), loops |

| 3 | Question: What is the capital of France? Answer: | 32 | 0 | "The capital of France is Paris. It is..." then repeats | correct, then repeats |

| 4 | The capital of France is, temp 0.7, repeat-penalty 1.1 | 32 | 0 | "...not the same as its capital..." | no loop, incoherent |



Errors seen earlier: llama-cli rejected --no-conversation (use llama-completion); my first run omitted -no-cnv, entered chat mode and crashed (exit -1073740791) on Lata's machine.

Stderr warning in all runs: "missing pre-tokenizer type ... GENERATION QUALITY WILL BE DEGRADED" (unexplained).

Not done: F32 comparison (no F32 file available). Speeds from single runs are not benchmarks.

