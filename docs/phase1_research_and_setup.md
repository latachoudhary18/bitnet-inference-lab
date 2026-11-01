\# Phase 1: Research and Project Setup



\## Project goal

Measure BitNet b1.58 inference speed and also check that the output is sane,

so we only compare speeds when the output is trustworthy.



\## Why BitNet b1.58

\- BitNet b1.58 was chosen because its ternary weights (-1, 0, +1) use far less

&#x20; memory than standard models and can run efficiently on CPUs, which suits a

&#x20; hardware-aware inference study on a laptop.

\- Microsoft provides an official CPU inference build (bitnet.cpp).



\## Research questions

1\. Can the official build run a BitNet b1.58 model and give readable output?

2\. How do threads and context size change speed?

3\. How can we detect broken output automatically?

4\. Which setting is best when speed and reliability are both considered?



\## Scope and limits

\- CPU only: 13th Gen Intel Core i5-1334U (10 cores, 12 logical processors),

&#x20; 15.7 GB RAM, Windows.

\- The reliability checks are sanity checks, not proof of correctness.

\- No speed result is valid until the output problem is fixed.



\## Tools

\- Model family: BitNet b1.58

\- Inference code: Microsoft BitNet repo, https://github.com/microsoft/BitNet,

&#x20; commit: not recorded (to be added by Lata)

\- Project repo: https://github.com/latachoudhary18/bitnet-inference-lab

\- Python 3.13.7, PowerShell



\## Phase plan

| Phase | Topic | Status |

|---|---|---|

| 1 | Research and setup | Done |

| 2 | Environment and model inspection | Pushed |

| 3 | Inference investigation | Pushed, no valid baseline yet |

| 4 | Runner and logging | In progress |

| 5 | Reliability and ranking | Pushed |

| 6 | Benchmarking | Waiting for valid output |

| 7 | Optimization | Future work |



\## Current status

\- No successful, trustworthy inference baseline has been established.

\- Recorded runs timed out (about 301.6 s and about 180.1 s), and the output failed the reliability checks.

\- No speedup or best configuration has been demonstrated.

\- A reliability PASS means only that the heuristics did not flag the output, not that it is correct.



\## References

\- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits" (2024). Citation to be verified.

\- Microsoft BitNet repository: https://github.com/microsoft/BitNet

