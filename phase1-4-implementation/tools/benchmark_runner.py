import json, re, subprocess, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "BitNet" / "build" / "bin" / "Release" / "llama-completion.exe"
MODEL = ROOT / "models" / "bitnet_b1_58-large" / "ggml-model-f32.gguf"
OUT = ROOT / "results" / "outputs"
LOG = ROOT / "results" / "raw" / "experiments.jsonl"
PROMPT = "Explain what a neural network is in one sentence."

OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

for repeat in range(1, 4):
    run_id = f"baseline_cpu_t4_p01_r{repeat:02d}"
    command = [str(EXE), "-m", str(MODEL), "-n", "32", "-t", "4",
               "-c", "512", "-p", PROMPT, "--temp", "0", "--seed", "42"]
    start = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=120)
        elapsed = time.perf_counter() - start
        stdout, stderr = result.stdout, result.stderr
        combined = stdout + "\n" + stderr
        code = result.returncode
        status = "success" if code == 0 else "nonzero_exit"
        prompt_speed = re.search(r"prompt eval time.*?([0-9.]+) tokens per second", combined)
        gen_speed = re.search(r"^.*common_perf_print:\s+eval time\s*=.*?([0-9.]+) tokens per second", combined, re.M)
        generated = re.search(r"eval time\s*=.*?/\s*(\d+) runs", combined)
        data = {
            "schema_version": "1.0", "run_id": run_id,
            "experiment_id": "baseline", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": "bitnet_b1_58_large_f32", "model_path": str(MODEL),
            "backend": "cpu", "threads": 4, "prompt_id": "p01",
            "prompt_text": PROMPT, "max_new_tokens": 32, "temperature": 0,
            "seed": 42, "timeout_seconds": 120, "command": command,
            "elapsed_seconds": round(elapsed, 4), "return_code": code,
            "status": status, "timed_out": False,
            "output_nonempty": bool(stdout.strip()),
            "output_char_count": len(stdout),
            "stdout_file": f"results/outputs/{run_id}.stdout.txt",
            "stderr_file": f"results/outputs/{run_id}.stderr.txt",
            "prompt_tokens_per_second": float(prompt_speed.group(1)) if prompt_speed else None,
            "generation_tokens_per_second": float(gen_speed.group(1)) if gen_speed else None,
            "generated_token_runs": int(generated.group(1)) if generated else None
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        data = {
            "schema_version": "1.0", "run_id": run_id, "experiment_id": "baseline",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_id": "bitnet_b1_58_large_f32", "model_path": str(MODEL),
            "backend": "cpu", "threads": 4, "prompt_id": "p01",
            "prompt_text": PROMPT, "max_new_tokens": 32, "temperature": 0,
            "seed": 42, "timeout_seconds": 120, "command": command,
            "elapsed_seconds": round(elapsed, 4), "return_code": None,
            "status": "timeout", "timed_out": True,
            "output_nonempty": bool(stdout.strip()), "output_char_count": len(stdout),
            "stdout_file": f"results/outputs/{run_id}.stdout.txt",
            "stderr_file": f"results/outputs/{run_id}.stderr.txt",
            "prompt_tokens_per_second": None, "generation_tokens_per_second": None,
            "generated_token_runs": None
        }

    (OUT / f"{run_id}.stdout.txt").write_text(stdout, encoding="utf-8")
    (OUT / f"{run_id}.stderr.txt").write_text(stderr, encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")
    print(f"{run_id}: {data['status']} | elapsed={data['elapsed_seconds']}s | generation={data.get('generation_tokens_per_second')} tok/s")
