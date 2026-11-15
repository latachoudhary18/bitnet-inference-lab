# Experiment-record reliability validation

This repository keeps two complementary reliability checks separate:

- `scripts/reliability_checker.py` evaluates generated text quality using heuristic rules.
- `tools/reliability_checker.py` validates experiment-record structure and metadata.

The experiment-record validator supports legacy flat schema v1.0 records and
nested schema v1.1 records. It does not repeat generated-text quality checks;
for v1.1 records it checks the structure and types of the `reliability` fields
produced by the shared record builder.

## Tests

Run from the repository root:

```powershell
python -m pytest tests\test_reliability_checker.py tests\test_experiment_validation.py tests\test_experiment_validation_v11.py -v
```

The validator-specific suite comprises 22 v1.0 tests and 21 v1.1 tests.
The combined 80-test result previously recorded on the contributor's local
workspace includes 37 generated-text checker tests, 22 v1.0 validator tests,
and 21 v1.1 validator tests. Re-run the command against the merged checkout
before treating it as the submission's final test result.

## Scope and limitations

A structurally valid experiment record does not prove that the underlying
inference was successful or semantically correct. Synthetic/stub records are
for software testing only and must not be reported as real BitNet benchmarks.
