
import json

from scripts import benchmark_sweep


def test_sweep_with_fake_runner(tmp_path, monkeypatch):
    calls = []

    def fake_run_inference(
        cli, model, prompt, threads, ctx_size,
        predict, timeout, output_file
    ):
        calls.append((threads, ctx_size))
        elapsed = float(len(calls) * 2)

        return {
            "execution_status": "success",
            "exit_code": 0,
            "timing": {"total_seconds": elapsed},
            "reliability": {"status": "PASS"},
            "performance": {
                "prompt_tps": None,
                "gen_tps": None,
                "peak_memory_mb": None,
            },
        }

    monkeypatch.setattr(
        benchmark_sweep,
        "run_inference",
        fake_run_inference,
    )

    output_file = tmp_path / "sweep_runs.jsonl"

    warmup = benchmark_sweep.run_one(
        "fake-cli",
        "fake-model.gguf",
        "test prompt",
        2, 128, 8, 10, True, output_file,
        data_source="synthetic",
    )

    measured = [
        benchmark_sweep.run_one(
            "fake-cli",
            "fake-model.gguf",
            "test prompt",
            2, 128, 8, 10, False, output_file,
            data_source="synthetic",
        )
        for _ in range(2)
    ]

    records = [
        json.loads(line)
        for line in output_file.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    summary = benchmark_sweep.build_summary(
        [warmup] + measured
    )

    assert len(calls) == 3
    assert len(records) == 3
    assert records[0]["benchmark"]["is_warmup"] is True
    assert all(
        not record["benchmark"]["is_warmup"]
        for record in records[1:]
    )

    config = summary["configurations"][0]

    assert config["total_measured_runs"] == 2
    assert config["successful_runs"] == 2
    assert config["reliability_passes"] == 2
    assert config["latency"]["mean_seconds"] == 5.0
    assert config["latency"]["median_seconds"] == 5.0
    assert config["mean_generation_tps"] is None

    assert all(
        record["benchmark"]["data_source"] == "synthetic"
        and record["benchmark"]["is_synthetic"] is True
        for record in records
    )

    print("Fake-runner benchmark sweep test PASSED")
