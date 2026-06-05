# Final Submission Scratchpad

This file is the working scratchpad for taking the project from "experiments
mostly done" to "safe to hand to the professor." Keep it updated as work
progresses. Do not mark an item complete unless there is file/artifact evidence.

## Active Goal

Produce a professor-ready final submission for the `Primjena umjetne
inteligencije` IMDb sentiment-analysis course project:

- stop unnecessary GPU rental
- complete any missing required experiments/evidence
- write the final report
- create the presentation
- add reproducibility/submission instructions
- perform a strict professor-style review
- fix all real issues found

Deadline: `2026-06-05 23:59`.

## Non-Negotiable Requirements

Source of truth: `AGENTS.md` and the introductory course PDF.

- Topic: `7. Analiza sentimenta recenzija (tekst klasifikacija)`
- Dataset: labelled IMDb review sentiment data; no manual labelling
- Required feature representation: `TF-IDF`
- Required classifier comparison:
  - `MultinomialNB`
  - `ComplementNB`
  - `LogisticRegression`
  - `LinearSVC`
  - `SGDClassifier`
  - `PassiveAggressiveClassifier`
  - `RandomForestClassifier`
  - `ExtraTreesClassifier`
  - `XGBoostClassifier`
  - `LightGBMClassifier`
- Required validation/search:
  - at least 10 classifiers
  - `RandomizedSearchCV`
  - cross-validation
- Required metrics:
  - accuracy
  - balanced accuracy
  - precision
  - recall
  - F1
  - ROC-AUC
  - PR-AUC
  - MCC
  - log-loss
  - confusion matrix
- Required final model: `VotingClassifier` or `StackingClassifier`
- Required deliverables:
  - seminar/report in `.pdf`, `.docx`, or `.tex`
  - Python source code or notebook plus short run instructions
  - presentation in `.tex` or `.pptx`

## Current Evidence Snapshot

- Core Python package exists under `src/` and `classifiers/`.
- All 10 required classifier families have runner folders and README files.
- Current best classical baseline: `LinearSVC`, test accuracy `0.9150`.
- Required ensembles exist:
  - soft `VotingClassifier`
  - hard `VotingClassifier`
  - prefit `StackingClassifier`
- Best optional modern model: `microsoft/deberta-v3-small`, test accuracy
  `0.9564`.
- Transformer runs are optional extension material; they do not replace the
  required TF-IDF/classical comparison.
- `outputs/` is ignored by git; important final numbers must be copied into
  tracked report/presentation files.
- Final report renders to 45 DOCX pages and includes 20 centered/captioned
  explanatory infographics under `figures/generated/`.
- The report now places metrics after the dataset section and before
  preprocessing/TF-IDF, matching the natural evaluation-before-pipeline reading
  order.
- The optional transformer section now documents the scratch-model experiment
  progression, including the 256-token failure, 10k BPE/mean-pooling/warmup
  run, the best MLX scratch result (`0.8943` test accuracy), CUDA context checks,
  and pretrained DistilBERT/DeBERTa fine-tuning.
- The transformer result tables now separate experiment progression from final
  metrics, and explicitly state that pre-fine-tune BERT/DeBERTa accuracy was
  not measured.
- The ten required classifier cards now appear in the main body, in section 6
  next to each classifier/family explanation, rather than being buried in a
  large appendix.
- Appendices are reduced to supporting material (`Dodatak A`-`G`): artifacts,
  run commands, environment, assignment coverage, limitations, source links, and
  oral-presentation plan.

## Progress Checklist

Use these statuses:

- `[ ]` not started
- `[~]` in progress or partially evidenced
- `[x]` complete and evidenced
- `[!]` risk or requires decision

### Cost / Runtime Hygiene

- `[x]` Stop the running sentiment Vast.ai GPU instance and verify it is no
  longer running.
- `[ ]` Decide whether to destroy any stopped Vast instances/storage after
  confirming nothing useful remains only on the remote machine.

### Assignment Coverage

- `[x]` Re-read `AGENTS.md` and the course PDF before writing final claims.
- `[x]` Verify selected topic and professor-provided dataset sources are stated
  accurately in the report.
- `[x]` Verify all 10 required classifier outputs exist and contain the required
  metrics and confusion matrices.
- `[x]` Strengthen or clearly justify the `RandomizedSearchCV`/cross-validation
  requirement. Current known evidence includes tuned `MultinomialNB`
  (`n_iter=10`, `cv=3`). A stricter professor might prefer broader tuning, but
  the report now states this limitation honestly.
- `[x]` Handle log-loss carefully for non-probabilistic models such as
  `LinearSVC`, hinge-loss `SGDClassifier`, and `PassiveAggressiveClassifier`.
  Either provide calibrated/probabilistic companion results or explicitly
  document why log-loss is unavailable for those exact hard-margin-style models.
- `[x]` Confirm final `VotingClassifier`/`StackingClassifier` result is included
  and interpreted honestly, even though it does not beat `LinearSVC`.
- `[x]` Clearly separate required course work from optional transformer
  extension work.

### Report

- `[x]` Create final report source under `reports/final_report/` or equivalent.
- `[x]` Export final report to professor-acceptable format (`.pdf`, `.docx`,
  or `.tex`).
- `[x]` Include task definition, no-manual-labelling note, dataset sources, and
  train/test split.
- `[x]` Explain TF-IDF and how vocabulary/ngrams are built from training data.
- `[x]` Explain each required classifier briefly and sensibly.
- `[x]` Include one compact table comparing all required classifiers on all
  required metrics.
- `[x]` Include confusion matrix discussion for at least the best classical
  model and final ensemble; include transformer confusion matrix only as
  optional extension if space allows.
- `[x]` Include RandomizedSearchCV/cross-validation setup and best parameters.
- `[x]` Include final model section for Voting/Stacking.
- `[x]` Include optional transformer section as extra research, not as a
  substitute for assignment requirements.
- `[x]` Expand the transformer section with the actual experiment progression
  and update the tiny-transformer table/figure values.
- `[x]` Replace the transformer metric block with cleaner tables and document
  why pre-fine-tune BERT accuracy is not reported.
- `[x]` Include limitations and conclusion.
- `[x]` Ensure all claims have local file/output evidence.
- `[x]` Include generated explanatory infographics for the pipeline, TF-IDF,
  model families, metrics, ensembles, results, and transformer extension.
- `[x]` Rebuild the DOCX with real Word formatting: centered figures, italic
  captions, bordered tables, repeated table headers, and cleaner appendix page
  breaks.
- `[x]` Inspect the revised DOCX render one page at a time for the affected
  figure-heavy pages, including the TF-IDF pages, section 6 classifier cards,
  result/confusion-matrix tables, transformer pages, and final Appendix G page.
- `[x]` Ensure top-level sections start on new pages and fix the classical
  confusion-matrix table so every row renders visibly in the PDF/DOCX output.

### Presentation

- `[x]` Create presentation source under `presentation/`.
- `[x]` Export final presentation to `.pptx` or `.tex`.
- `[x]` Keep it short enough to present comfortably.
- `[x]` Include problem, dataset, methods, result table, final ensemble,
  transformer extension, conclusion.
- `[x]` Render the PPTX through the presentation artifact-tool inspector and
  inspect all 12 slide previews one by one. Slides are readable; table slides
  are plain but acceptable.

### Reproducibility / Submission Package

- `[x]` Improve README with install and run instructions.
- `[x]` Add a short "submission contents" section explaining what to hand in.
- `[x]` Decide whether generated outputs should remain ignored or whether a
  small tracked results table should be committed.
- `[x]` Ensure code can be run from repo root with `PYTHONPATH=.` or package
  install instructions.
- `[x]` Remove or avoid submitting irrelevant scratch/SRO handoff artifacts.
- `[x]` Final git status should be understood; no accidental huge files.

### Final Review

- `[x]` Run a strict self-review as the professor.
- `[x]` Fix every legitimate issue found by the strict review.
- `[x]` Perform final "would the professor fail me?" check.
- `[x]` Perform final "does this look like AI slop?" check and make the writing
  concrete, specific, and evidence-backed.

## Worker Prompt

You are Codex working inside:

`/Users/lukaivanic/projects/faks/Primjena umjetne inteligencije/sentiment_analysis_reviews`

Your task is to finish the university course project for hand-in, not to merely
make progress. The final state must be something Luka can submit to the
professor without embarrassment.

Act as a careful engineer and technical writer. Before changing files, read
`AGENTS.md`, `README.md`, existing classifier READMEs, existing metric artifacts,
and the relevant report/presentation folders. Treat `AGENTS.md` and the
introductory course PDF as the assignment source of truth.

Complete the following:

1. Stop any unnecessary active Vast.ai GPU rental and verify it is no longer
   running. Do not expose API keys or secrets.
2. Audit requirement coverage against the course task:
   - 10 required TF-IDF classifier families
   - `RandomizedSearchCV`
   - cross-validation
   - all required metrics
   - confusion matrices
   - final `VotingClassifier` or `StackingClassifier`
3. If the RandomizedSearchCV/cross-validation evidence is too thin for a human
   professor, implement or run a reasonable additional search for one or more
   fast classical models, preferably without using local Mac for intensive work.
4. Resolve the log-loss issue honestly. For classifiers without probabilities,
   either add calibrated/probabilistic companion results or document clearly why
   log-loss is not mathematically available for those exact model outputs.
5. Write the final report in a professor-readable style. It must include:
   - selected topic and assignment statement
   - dataset sources and no-manual-labelling note
   - preprocessing and TF-IDF explanation
   - classifier explanations
   - validation/search explanation
   - full results table
   - final ensemble model
   - optional transformer extension
   - limitations and conclusion
6. Create the final presentation.
7. Add or improve run instructions so the professor can understand how to run
   the code and where outputs are saved.
8. Keep this scratchpad updated after each major step.
9. Commit only appropriate source/docs/final deliverables. Do not commit large
   raw datasets, giant model weights, irrelevant `.DS_Store`, or scratch
   artifacts unless deliberately required.

Do not claim completion unless the deliverables exist and have been inspected.

## Reviewer Prompt

You are a hostile-but-fair reviewer acting as the professor for the `Primjena
umjetne inteligencije` course. Assume the worker model may be overconfident,
may confuse optional transformer work with required TF-IDF work, and may try to
convince you that partial work is done. Do not be gaslit. Require concrete file
paths, metrics, reports, and presentation artifacts.

Review the project as if you personally wrote the course requirements. Check:

1. Does the submission clearly match topic 7, sentiment analysis of reviews?
2. Are the datasets legitimate, labelled, professor-provided sources?
3. Is there any hidden manual labelling? If yes, fail it.
4. Is TF-IDF actually used for the required classical comparison?
5. Are all 10 required classifier families implemented and evaluated?
6. Is `RandomizedSearchCV` actually used, with cross-validation, and is it
   described in the report?
7. Are all required metrics present or responsibly handled?
8. Is log-loss missing where it should be available? Are unavailable cases
   explained mathematically rather than hand-waved?
9. Is a `VotingClassifier` or `StackingClassifier` actually implemented and
   reported as the final required ensemble?
10. Does the report distinguish required assignment results from optional
    transformer extension results?
11. Are the results tables internally consistent with local artifacts?
12. Can a human reader reproduce the main runs from the README?
13. Are report and presentation present in accepted formats?
14. Are there signs of AI slop: generic filler, unsupported claims, inconsistent
    numbers, fake citations, unexplained acronyms, or polished prose hiding
    missing work?

After reviewing, answer these two questions explicitly:

1. If Luka handed this in to the professor right now, what is the most likely
   reason he could lose points or fail?
2. Does this look like a serious student project with real experiments, or does
   it smell like AI-generated slop? Explain exactly why.

Mark the submission as one of:

- `PASS_READY`
- `PASS_WITH_MINOR_RISK`
- `NOT_READY`
- `FAIL_RISK`

Do not choose `PASS_READY` unless the report, presentation, code, instructions,
and requirement evidence are all present and coherent.

## Final Professor-Fail Check

Before declaring the goal complete, ask:

> If Luka walked into the professor's office with exactly this submission, would
> the professor be able to fail him or heavily penalize him because a required
> part is missing, fake, unsupported, incoherent, or obviously AI-generated?

If the honest answer is anything other than "no, the remaining risks are minor
and defensible," keep working.
