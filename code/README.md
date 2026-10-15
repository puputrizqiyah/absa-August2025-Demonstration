# code/

The pipeline of the deposit. Run everything from the repository root, for example:

```bash
pip install -r requirements.txt
python code/run_benchmark.py --report-vocabulary
python code/run_benchmark.py --schemes stratified grouped day_held_out --tasks relevance aspect polarity
python code/make_tables.py
```

## Order that matters

1. `neardup.py` and `day_partition.py` write the inputs of the grouped and temporal
   schemes into `data/derived/`, so they must run before those schemes.
2. `run_benchmark.py` writes `results/folds/<scheme>_<task>.csv`. `indobert_model.py`
   reads those files so that the encoder is trained on exactly the same folds as the
   lexical models; run the benchmark first.
3. `make_tables.py` reads the predictions and `results/indobert_results.jsonl`, so it
   runs last.

## Modules

| Module | Role |
| --- | --- |
| `config.py` | every path, hyperparameter, keyword list and expected value (the self-checks compare against it) |
| `data_io.py` | gold standard and comparator labels; `task_indices` / `task_frame` define the record set of a task |
| `folds.py` | stratified, near-duplicate grouped and day-held-out splits |
| `lexical_models.py` | model zoo and `vocabulary_report` (the feature counts of Section 3) |
| `metrics.py` | macro-F1, accuracy, per-class, fold statistics, conditional and coverage-aware scoring |
| `stats_tests.py` | paired bootstrap, exact McNemar, Nadeau-Bengio corrected resampled t-test, Holm-Bonferroni |
| `run_benchmark.py` | orchestration and per-record out-of-fold predictions |
| `make_tables.py` | the tables of the article, rebuilt from the predictions |
| `neardup.py`, `day_partition.py`, `features.py` | the derived artifacts and the Table 8 feature rates |
| `zeroshot.py` | comparator scoring, optional re-query of the API configuration |
| `indobert_model.py` | fine-tuned encoder on the identical folds |

## Conventions

* **Coverage is explicit.** Anything that can be unlabelled carries `None` rather than an
  imputed label, and `metrics.conditional_scores` / `metrics.coverage_aware_scores` are
  reported as a pair.
* **Folds are data, not code.** A split is written to `results/folds/*.csv` and consumed
  by every later step, including the encoder, so the comparability of the paired tests
  is auditable.
* **Self-checks are part of the run.** Every script prints the published value next to
  the recomputed one; `docs/reproducibility_status.md` records the outcome, including the
  two places where the deposit cannot reach the published number and why.
