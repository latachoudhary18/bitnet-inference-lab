
from scripts.configuration_ranking import rank_configurations


def test_ranks_reliable_configurations_by_throughput():
    summary = {
        "configurations": [
            {
                "threads": 2,
                "ctx_size": 128,
                "successful_runs": 3,
                "reliability_passes": 3,
                "mean_generation_tps": 12.0,
            },
            {
                "threads": 4,
                "ctx_size": 128,
                "successful_runs": 3,
                "reliability_passes": 3,
                "mean_generation_tps": 18.0,
            },
        ]
    }

    result = rank_configurations(summary)

    assert result["eligible_configurations"] == 2
    assert result["winner"]["threads"] == 4
    assert result["winner"]["mean_generation_tps"] == 18.0
    assert result["ranked_configurations"][0]["rank"] == 1


def test_excludes_unreliable_configuration():
    summary = {
        "configurations": [
            {
                "threads": 2,
                "ctx_size": 128,
                "successful_runs": 3,
                "reliability_passes": 0,
                "mean_generation_tps": 20.0,
            }
        ]
    }

    result = rank_configurations(summary)

    assert result["winner"] is None
    assert result["eligible_configurations"] == 0
    assert result["excluded_configurations"] == 1


def test_no_winner_when_throughput_is_unavailable():
    summary = {
        "configurations": [
            {
                "threads": 2,
                "ctx_size": 128,
                "successful_runs": 3,
                "reliability_passes": 3,
                "mean_generation_tps": None,
            }
        ]
    }

    result = rank_configurations(summary)

    assert result["winner"] is None
    assert result["eligible_configurations"] == 0
    assert "throughput unavailable" in (
        result["excluded_configurations_detail"][0]["reasons"][-1]
    )


def test_excludes_configuration_without_successful_runs():
    summary = {
        "configurations": [
            {
                "threads": 8,
                "ctx_size": 256,
                "successful_runs": 0,
                "reliability_passes": 0,
                "mean_generation_tps": None,
            }
        ]
    }

    result = rank_configurations(summary)

    assert result["winner"] is None
    assert result["excluded_configurations"] == 1

def test_handles_existing_single_run_summary():
    summary = {
        "total_runs": 1,
        "successful_processes": 0,
        "reliability_passes": 0,
        "generation_tokens_per_second": {
            "measurement_count": 0,
            "mean": None,
        },
    }

    result = rank_configurations(summary)

    assert result["report_type"] == "single_run_summary"
    assert result["winner"] is None
    assert result["eligible_configurations"] == 0
    assert result["successful_runs"] == 0
    assert result["mean_generation_tps"] is None

def make_v11_record(
    threads=4,
    ctx_size=128,
    warmup=False,
    execution_status="success",
    exit_code=0,
    reliability_status="PASS",
    gen_tps=15.0,
    seconds=2.0,
    data_source="real",
):
    return {
        "schema_version": "1.1",
        "data_source": data_source,
        "execution_status": execution_status,
        "exit_code": exit_code,
        "benchmark": {
            "threads": threads,
            "ctx_size": ctx_size,
            "is_warmup": warmup,
        },
        "performance": {
            "gen_tps": gen_tps,
        },
        "timing": {
            "total_seconds": seconds,
        },
        "reliability": {
            "status": reliability_status,
        },
    }


def test_v11_ranks_reliable_records_by_throughput():
    records = [
        make_v11_record(threads=2, gen_tps=12.0),
        make_v11_record(threads=4, gen_tps=18.0),
    ]

    result = rank_configurations(records)

    assert result["winner"]["threads"] == 4
    assert result["ranking_metric"] == "mean_generation_tps"


def test_v11_excludes_warmup_and_unreliable_records():
    records = [
        make_v11_record(warmup=True, gen_tps=100.0),
        make_v11_record(reliability_status="WARNING"),
        make_v11_record(exit_code=1),
        make_v11_record(execution_status="timeout"),
    ]

    result = rank_configurations(records)

    assert result["winner"] is None
    assert result["eligible_configurations"] == 0


def test_v11_uses_latency_when_throughput_unavailable():
    records = [
        make_v11_record(threads=2, gen_tps=None, seconds=3.0),
        make_v11_record(threads=4, gen_tps=None, seconds=1.5),
    ]

    result = rank_configurations(records)

    assert result["ranking_metric"] == "mean_total_seconds"
    assert result["winner"]["threads"] == 4


def test_v11_groups_repetitions_by_configuration():
    records = [
        make_v11_record(gen_tps=10.0),
        make_v11_record(gen_tps=20.0),
    ]

    result = rank_configurations(records)

    assert result["eligible_configurations"] == 1
    assert result["winner"]["eligible_runs"] == 2
    assert result["winner"]["mean_generation_tps"] == 15.0


def test_v11_labels_synthetic_results():
    records = [
        make_v11_record(data_source="synthetic"),
    ]

    result = rank_configurations(records)

    assert result["winner"]["label"] == "[SYNTHETIC / STUB]"
