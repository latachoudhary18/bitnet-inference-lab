import json
import statistics
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "raw" / "experiments.jsonl"
REPORT = ROOT / "results" / "validated" / "performance_comparison.json"

groups = defaultdict(list)

for line in LOG.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("status") == "success" and row.get("generation_tokens_per_second"):
        groups[row["threads"]].append(row["generation_tokens_per_second"])

if 4 not in groups:
    raise SystemExit("Baseline (4-thread) results missing")

baseline = statistics.mean(groups[4])
results = []

for threads in sorted(groups):
    values = groups[threads]
    avg = statistics.mean(values)
    results.append({
        "threads": threads,
        "runs": len(values),
        "average_tokens_per_second": round(avg, 3),
        "std_dev": round(statistics.stdev(values), 3) if len(values) > 1 else 0,
        "improvement_over_baseline_percent": round((avg / baseline - 1) * 100, 2)
    })

report = {
    "experiment": "CPU thread configuration comparison",
    "model": "bitnet_b1_58_large_f32",
    "baseline_threads": 4,
    "results": results,
    "best_configuration_threads": max(results, key=lambda x: x["average_tokens_per_second"])["threads"]
}

REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"\nReport saved: {REPORT}")
