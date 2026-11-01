"""Fake llama-cli stub for emulating BitNet / llama.cpp executions in automated tests.

Does NOT load model weights or require C++ binaries. Used exclusively for synthetic
integration and unit tests of scripts/inference_runner.py.
"""
from __future__ import annotations

import argparse
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="Test stub for llama-cli.")
    parser.add_argument("--stub-mode", default="normal", help="Execution mode for testing.")
    args, _ = parser.parse_known_args()

    mode = args.stub_mode

    if mode == "normal":
        sys.stdout.buffer.write(
            "The capital of France is Paris, famous for its architecture and art.\n".encode("utf-8")
        )
        sys.stderr.buffer.write(
            (
                "llama_perf_context_print: prompt eval time =   10.00 ms /   10 tokens (   1.00 ms per token,  1000.00 tokens per second)\n"
                "llama_perf_context_print:        eval time =  200.00 ms /   20 tokens (  10.00 ms per token,   100.00 tokens per second)\n"
                "[ Prompt: 1000.0 t/s | Generation: 100.0 t/s ]\n"
            ).encode("utf-8")
        )
        sys.exit(0)

    elif mode == "question_marks":
        sys.stdout.buffer.write("????????????????????????????????\n".encode("utf-8"))
        sys.stderr.buffer.write("[ Prompt: 1.0 t/s | Generation: 0.8 t/s ]\n".encode("utf-8"))
        sys.exit(0)

    elif mode == "empty":
        sys.stdout.buffer.write(b"")
        sys.stderr.buffer.write("Loading model...\nModel loaded successfully.\n".encode("utf-8"))
        sys.exit(0)

    elif mode == "garbled":
        sys.stdout.buffer.write(
            ("\ufffd" * 30 + "\n").encode("utf-8")
        )
        sys.stderr.buffer.write("error: corrupted token buffer\n".encode("utf-8"))
        sys.exit(0)

    elif mode == "hang":
        time.sleep(15)
        sys.exit(0)

    elif mode == "nonzero_exit":
        sys.stderr.buffer.write(
            (
                "ggml_compute_forward: tensor shape mismatch\n"
                "Fatal error in inference runtime\n"
                "Process terminated unexpectedly\n"
            ).encode("utf-8")
        )
        sys.exit(1)

    elif mode == "unsupported_arg":
        sys.stderr.buffer.write(
            (
                "--no-conversation is not supported by llama-cli\n"
                "please use llama-completion instead\n"
            ).encode("utf-8")
        )
        sys.exit(1)

    elif mode == "non_ascii_bytes":
        sys.stdout.buffer.write(b"Valid UTF-8: Caf\xc3\xa9 and invalid: \xff\xfe bytes\n")
        sys.stderr.buffer.write(b"Log: \xc3\xa9 invalid \x80 sequence\n")
        sys.exit(0)

    elif mode == "large_output":
        sys.stdout.buffer.write(("Line of generated text for buffer test.\n" * 2500).encode("utf-8"))
        sys.stderr.buffer.write(("Diagnostic stderr line for buffer test.\n" * 2500).encode("utf-8"))
        sys.exit(0)

    else:
        sys.stderr.buffer.write(f"Unknown stub-mode: {mode}\n".encode("utf-8"))
        sys.exit(2)


if __name__ == "__main__":
    main()
