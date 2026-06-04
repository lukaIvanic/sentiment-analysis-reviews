# Final Strict Review

Review date: 2026-06-04

Reviewer stance: hostile-but-fair professor. The goal of this review is not to
be encouraging, but to decide whether the project is actually hand-in ready.

## Verdict

`PASS_WITH_MINOR_RISK`

The submission is coherent and evidence-backed. The core task is implemented:
IMDb sentiment analysis, no manual labelling, TF-IDF features, the ten required
classifiers, RandomizedSearchCV with cross-validation, required metrics,
confusion matrices, and final Voting/Stacking ensembles. The report and
presentation exist in accepted formats and have been rendered/checked.

The remaining risks are real but defensible:

- `RandomizedSearchCV` is demonstrated on tuned `MultinomialNB`
  (`n_iter=10`, `cv=3`) rather than used for many classifiers. The wording of
  the task requires RandomizedSearchCV/CV, not exhaustive tuning of every
  classifier, but a stricter grader could have preferred broader search.
- The Croatian report is ASCII-only rather than fully written with Croatian
  diacritics. It is readable and consistent, but less polished typographically.

Neither issue looks like a missing required component. They are more likely to
cost points than to fail the submission.

## Requirement Audit

| Requirement | Status | Evidence |
| --- | --- | --- |
| Topic 7, sentiment analysis of reviews | pass | `AGENTS.md`; report section 2 |
| Legit labelled datasets | pass | Kaggle IMDb 50K and Stanford ACL IMDB listed in report section 3 |
| No manual labelling | pass | report sections 3 and summary explicitly state no manual labelling |
| TF-IDF for required comparison | pass | report section 4; classical runner folders use `TfidfVectorizer` |
| 10 required classifiers | pass | report section 8 includes all required families |
| RandomizedSearchCV | pass | tuned MultinomialNB row; report section 6 |
| Cross-validation | pass | tuned MultinomialNB uses `cv=3`; report section 6 |
| ACC, BACC, precision, recall, F1 | pass | `reports/final_report/results_tables.md`; report section 8 |
| ROC-AUC, PR-AUC, MCC | pass | `reports/final_report/results_tables.md`; report section 8 |
| Log-loss | pass with explanation | reported where `predict_proba` exists; n/a explained for uncalibrated margin models |
| Confusion matrices | pass | report sections 8, 9, 10 |
| Voting/Stacking final model | pass | soft voting, hard voting, and prefit stacking in report section 9 |
| Required and optional work separated | pass | transformer work is section 10 and explicitly marked optional |
| Report accepted format | pass | `reports/final_report/analiza_sentimenta_recenzija.docx` |
| Presentation accepted format | pass | `presentation/analiza_sentimenta_recenzija.pptx` |
| Reproducibility instructions | pass | `README.md`; report section 15 and appendices |

## Artifact Checks

- Python source compilation: `python -m compileall src classifiers` passed.
- DOCX render: `reports/final_report/analiza_sentimenta_recenzija.docx`
  rendered successfully to 40 PNG pages.
- Report source length after appendix expansion and visual guide: 8,928 words.
- Generated explanatory infographics are tracked under `figures/generated/`
  and embedded in the DOCX.
- Presentation: 12 slides, within the 10-20 slide guidance. Prior layout QA
  passed with 0 errors and 0 warnings.
- Important generated results were copied into tracked files:
  `reports/final_report/results_table.csv` and
  `reports/final_report/results_tables.md`.

## Most Likely Point Loss

The most likely reason to lose points is limited hyperparameter search. The report
honestly says `RandomizedSearchCV` was run for `MultinomialNB`, not every model.
That satisfies the literal requirement but is not maximal experimental depth.

A smaller possible point loss is polish: the report is in Croatian content-wise
but uses ASCII transliteration instead of diacritics. The writing is still
specific, but it is not typographically ideal.

## AI-Slop Check

This does not read like empty AI slop. The strongest evidence:

- exact metrics are internally consistent across tables,
- confusion matrices are present,
- model-specific limitations are explained rather than hidden,
- log-loss is not faked for models without probabilities,
- transformer results are separated from the required TF-IDF/classical work,
- source paths and run commands are named concretely,
- generated infographics are labeled and tied to actual sections/results.

The biggest style weakness is that the Croatian text is ASCII-only and lacks
diacritics. That is less polished, but it is still readable and consistent with
the rest of the project notes.

## Professor-Fail Check

Question: If Luka walked into the professor's office with exactly this
submission, would the professor be able to fail him or heavily penalize him
because a required part is missing, fake, unsupported, incoherent, or obviously
AI-generated?

Answer: No. The remaining risks are minor and defensible. A strict professor
could reduce points for limited tuning breadth or typographic polish, but the
core required work is present, reported, and reproducible.
