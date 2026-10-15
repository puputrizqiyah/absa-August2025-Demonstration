# Reproducibility status

What the deposited code reproduces from the published article, and what it does not.
Every number below was produced by running the scripts in `code/` on the files in
`data/`; the commands are in `README.md`.

Environment: Python 3.11, scikit-learn 1.9.1, numpy 2.4.6, scipy 1.17.1, openpyxl 3.1.
Seed 42 throughout. The encoder results in `results/indobert_results.jsonl` are the
recorded fold and pooled scores of the run reported in the article.

## Reproduces exactly

| Item | Published | Recomputed |
| --- | --- | --- |
| Corpus size, relevant subset | 7,131 records, 6,075 relevant | 7,131 / 6,075 |
| Vectoriser sizes, Section 3 | 23,652 / 21,400 word, 39,593 / 36,408 character | identical (`run_benchmark.py --report-vocabulary`) |
| Table 3/4, all 15 model-task cells | macro-F1 and accuracy | identical to 3 decimals and 0.01 |
| Table 3/4, encoder row | 0.757 ± 0.021 / 89.17; 0.772 ± 0.010 / 77.48; 0.665 ± 0.013 / 78.67 | identical |
| Table 5, comparator per-class | polarity Positif F1 0.571 on 547 answered records | identical |
| Table 6, comparator | coverage 100% / 78.85% / 84.18%; coverage-aware 0.698 / 0.711 / 0.647 | identical |
| Table 8, population | 5,114 comparator-labelled records, 784 errors, 15.3% | identical |
| Table 9, all 15 contrasts | Δ macro-F1, 95% interval, p | Δ identical to 4 decimals, intervals within ±0.0005, every p identical (0.488, 0.002, 0.114, 0.032, 0.084, 0.130, 0.759, 0.015 and the < 0.001 entries) |
| Table 9, Holm-Bonferroni | 8 of 15 contrasts survive | 8 of 15 |
| Fold sizes vs the recorded encoder run | relevance 1,427 + 4 x 1,426; aspect/polarity 5 x 1,215 | identical, i.e. the encoder in `results/indobert_results.jsonl` was trained on the same five folds as the lexical models, which is what makes the paired contrasts legitimate (see *Encoder fold correspondence* below for how far that rests on construction rather than on re-running the encoder) |
| Section 3, temporal concentration | 6,334 records (88.8%) in one 24-hour window | identical (10 collection days, 29-31 August 2025 dominant) |
| Section 3, label distribution | 2,192 / 1,990 / 1,893 aspects; 4,149 / 1,252 / 674 polarities | identical |

## Reproduces as a near miss

| Item | Published | Recomputed | Why |
| --- | --- | --- | --- |
| Table 10, day-held-out row | 0.461 / 0.469 / 0.602 / 0.506 / 0.489 (relevance), 0.713 / 0.719 / 0.657 / 0.734 (aspect), 0.499 / 0.585 / 0.529 / 0.510 (polarity) | 0.460 / 0.469 / 0.608 / 0.505 / 0.491, 0.713 / 0.718 / 0.657 / 0.733, 0.500 / 0.586 / 0.531 / 0.511: within ± 0.006 | the temporal row pools a macro-F1 computed per held-out day, so the value depends on which days can serve as a test fold: relevance keeps all 10 collection days (folds of 3 to 4,362 records), while aspect and polarity keep 8, because two days have no relevant record. Two of those folds hold 1 and 2 records, and the pooled average is therefore sensitive to the exact day set; the residual of at most 0.006 is smaller than one record of the smallest fold |
| Table 10, grouped row | 0.558 / 0.720 / 0.601 / 0.629 (relevance), 0.763 aspect, 0.567 polarity | 0.553 / 0.721 / 0.588 / 0.631, 0.766, 0.576 | the grouping key is the near-duplicate clustering, which does not reproduce exactly (next table); the five folds and their sizes are otherwise deterministic, so the difference is confined to the composition of the clusters |

## Does not reproduce

| Item | Published | Recomputed | Why |
| --- | --- | --- | --- |
| Near-duplicate audit | 96 records in 76 clusters, largest cluster 7, 1.35% of the corpus | 85 records in 38 clusters, largest cluster 7, 1.19% | the original audit script is not part of the record. The deposited implementation follows the printed definition (normalised word-trigram Jaccard >= 0.7, case, punctuation, URLs, hashtags and mentions removed) and is exhaustive (prefix filter over a document-frequency token order, so no pair above the threshold can be missed). The sweep run for this deposit covers n-gram length 2-4, threshold 0.6-0.9, hashtags deleted / kept as words / kept with the marker, punctuation removed or not, and both prefix-filtered and brute-force candidate generation: the closest configurations give 107 records in 47 clusters (bigrams, 0.7) and 83 records in 37 clusters (trigrams, 0.7), and none returns 96 in 76. The largest cluster agrees at 7, so the difference lies in the number of small pairs found, not in the structure |
| Table 8, news template | 230 tweets, 48 errors, 20.9%, p = 0.022 | 327 tweets, 68 errors, 20.8%, p = 0.006 (outlet names plus report markers) or 139 tweets, 30 errors, 21.6% (outlet names only) | the original keyword list is not part of the record; the printed definition ("names a mainstream Indonesian news outlet") is re-implemented as a list of 38 Indonesian outlet names plus the report-template markers. Both variants reproduce the published error *rate* to within 0.7 points but not the base count |
| Table 8, informal markers | 2,264 / 290 / 12.8 / 17.3 | 2,263 / 290 / 12.8 / 17.3 | one record, consistent with a tokenisation boundary in the original run |
| Table 8, negation | 1,860 / 269 / 14.5 / 15.8, p = 0.207 | 1,857 / 269 / 14.5 / 15.8, p = 0.220 | three records; same tokenisation boundary |
| Table 8, English code-mixing | 67 / 5 / 7.5, p = 0.103 | 89 / 5 / 5.6, p = 0.016 | the published count implies a shorter function-word list than the one implemented |
| Table 8, sarcasm markers | 272 / 35 / 12.9, p = 0.284 | 174 / 20 / 11.5, p = 0.186 | the published count is between word-boundary matching (174) and substring matching (321) over the printed marker list, so the original matched a variant of the list |

Three features reproduce exactly (emoji/non-ASCII 237 / 50 / 21.1, question form
792 / 83 / 10.5, and the population), and the three features whose definition depends
only on arithmetic from the published table (informal, negation, code-mixing) agree on
the error counts and rates.

## Encoder fold correspondence

The recorded encoder run labels its five folds 1 to 5 (`indobert_bench.log` prints
`fold 1/5 ... acc=0.8886`); the deposited fold files label them 0 to 4. The two agree
positionally.

- The deposited fold files are exactly what the deposited builder produces. Rebuilding
  the stratified split from `code/folds.py` (`StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42)` over the same frame) reproduces the recorded fold of every record:
  0 mismatches over all 7,131 relevance records and over both 6,075-record subsets. The
  split is therefore fixed by the seed and the frame order, not redrawn per run.
- The one fold whose test size differs from its siblings identifies the offset. In the
  recorded run only fold 1 has 1,427 records (folds 2-5 have 1,426); in the deposit only
  fold 0 has 1,427 (folds 1-4 have 1,426). Recorded fold k is deposited fold k - 1.
- `code/make_tables.py` pairs them exactly that way: the encoder's per-fold scores are
  read in file order (1 to 5) and the per-fold scores of the lexical models in fold-label
  order (0 to 4), that is position against position. The nine fold-level contrasts of
  Table 9 are computed on those pairs.

Scope of the claim. The correspondence is established by construction - same splitter,
same seed, same frame, one-fold size anchor - and not by score agreement. A full re-run
of the encoder on the deposited folds, which would confirm it by reproducing the five
per-fold scores in the same order, was not carried out: at the measured throughput the
run takes about 2.5 hours. The single-fold smoke test that was run gave macro-F1 0.7551
where the recorded run reports 0.7592 for the fold at that position, a difference of
0.004. Comparing folds by how hard they are is not a substitute: per-fold SD (0.010 to
0.021) is of the same order as the fold-to-fold spread, so the ordering of folds by
difficulty is not stable enough across model families to identify the correspondence.

## Consequence for the article

Two options, both defensible:

1. **Regenerate Table 8 and the near-duplicate sentence from this deposit**, so that
   every number in the article is reproducible from the released code. The affected
   numbers are the near-duplicate audit counts and four rows of Table 8; the qualitative
   reading (emotive marking and quoted-reporting templates are the harder cases;
   register is not) is unaffected in either direction.
2. **Re-run the missing detectors from the original keyword lists**, if those are still
   available. `code/config.py` holds every list in one place, so restoring them and
   re-running `code/features.py` is a single edit; `code/neardup.py` exposes the n-gram
   length and the threshold as arguments.
