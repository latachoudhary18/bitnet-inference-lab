"""Safe integration demo: harmless commands -> schema record -> JSONL.

Does NOT run the BitNet model. It proves that stdout/stderr capture, timeouts,
missing files and process errors all produce valid, understandable records.
"""
import subprocess
import sys
import time

from scripts.result_schema import build_record, validate_record, append_jsonl

OUT = "results/reliability_demo.jsonl"


def _txt(x):
    if x is None:
        return ""
    return x.decode("utf-8", errors="replace") if isinstance(x, bytes) else x


def run_case(run_id, cmd, timeout, prompt=None):
    cfg = {"command": cmd, "timeout_s": timeout}
    start = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        status = "success" if p.returncode == 0 else "error"
        rec = build_record(run_id, cfg, status, p.returncode, p.stdout, p.stderr,
                           round(time.time() - start, 3), prompt=prompt,
                           error_message=None if status == "success" else "non-zero exit")
    except subprocess.TimeoutExpired as e:
        rec = build_record(run_id, cfg, "timeout", None, _txt(e.stdout), _txt(e.stderr),
                           round(time.time() - start, 3), prompt=prompt,
                           error_message=f"timed out after {timeout}s")
    except FileNotFoundError as e:
        rec = build_record(run_id, cfg, "missing_file", None, "", "",
                           round(time.time() - start, 3), prompt=prompt,
                           error_message=str(e))
    problems = validate_record(rec)
    assert not problems, problems
    append_jsonl(OUT, rec)
    print(f"{run_id:14s} exec={rec['execution_status']:13s} "
          f"reliability={rec['reliability']['status']}")


if __name__ == "__main__":
    py = sys.executable
    run_case("demo_good", [py, "-c", "print('The capital of France is Paris, known for the Eiffel Tower.')"], 10)
    run_case("demo_qmarks", [py, "-c", "print('?????????????')"], 10)
    run_case("demo_error", [py, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"], 10)
    run_case("demo_timeout", [py, "-c", "import time; time.sleep(5)"], 1)
    run_case("demo_missing", ["definitely_not_a_real_program.exe"], 5)