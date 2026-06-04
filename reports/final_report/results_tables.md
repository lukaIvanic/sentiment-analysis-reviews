# Final Results Tables

These tables are generated from local `outputs/**/metrics.json` and `outputs/**/confusion_matrix.json` artifacts.

## Required Classical TF-IDF Models

| model | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc | mcc | log_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MultinomialNB | 0.8841 | 0.8841 | 0.8808 | 0.8884 | 0.8846 | 0.9531 | 0.9513 | 0.7682 | 0.3135 |
| MultinomialNB tuned | 0.8854 | 0.8854 | 0.8842 | 0.8870 | 0.8856 | 0.9554 | 0.9537 | 0.7708 | 0.2934 |
| ComplementNB | 0.8841 | 0.8841 | 0.8808 | 0.8884 | 0.8846 | 0.9531 | 0.9513 | 0.7682 | 0.3135 |
| LogisticRegression | 0.9095 | 0.9095 | 0.9044 | 0.9158 | 0.9101 | 0.9710 | 0.9702 | 0.8191 | 0.2707 |
| LinearSVC | 0.9150 | 0.9150 | 0.9135 | 0.9168 | 0.9152 | 0.9720 | 0.9708 | 0.8300 |  |
| SGDClassifier | 0.9111 | 0.9111 | 0.9052 | 0.9184 | 0.9117 | 0.9714 | 0.9707 | 0.8223 |  |
| PassiveAggressiveClassifier | 0.9062 | 0.9062 | 0.9064 | 0.9060 | 0.9062 | 0.9656 | 0.9638 | 0.8124 |  |
| RandomForestClassifier | 0.8633 | 0.8633 | 0.8673 | 0.8578 | 0.8625 | 0.9376 | 0.9299 | 0.7266 | 0.4554 |
| ExtraTreesClassifier | 0.8749 | 0.8749 | 0.8820 | 0.8656 | 0.8737 | 0.9466 | 0.9398 | 0.7499 | 0.4342 |
| XGBoostClassifier | 0.8509 | 0.8509 | 0.8307 | 0.8814 | 0.8553 | 0.9355 | 0.9337 | 0.7031 | 0.3565 |
| LightGBMClassifier | 0.8918 | 0.8918 | 0.8865 | 0.8986 | 0.8925 | 0.9605 | 0.9599 | 0.7837 | 0.2617 |


## Required Classical Confusion Matrices

| model | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: |
| MultinomialNB | 4399 | 601 | 558 | 4442 |
| MultinomialNB tuned | 4419 | 581 | 565 | 4435 |
| ComplementNB | 4399 | 601 | 558 | 4442 |
| LogisticRegression | 4516 | 484 | 421 | 4579 |
| LinearSVC | 4566 | 434 | 416 | 4584 |
| SGDClassifier | 4519 | 481 | 408 | 4592 |
| PassiveAggressiveClassifier | 4532 | 468 | 470 | 4530 |
| RandomForestClassifier | 4344 | 656 | 711 | 4289 |
| ExtraTreesClassifier | 4421 | 579 | 672 | 4328 |
| XGBoostClassifier | 4102 | 898 | 593 | 4407 |
| LightGBMClassifier | 4425 | 575 | 507 | 4493 |


## Final Ensemble Models

| model | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc | mcc | log_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VotingClassifier soft | 0.9095 | 0.9095 | 0.9036 | 0.9168 | 0.9102 | 0.9707 | 0.9702 | 0.8191 | 0.2654 |
| VotingClassifier hard | 0.9107 | 0.9107 | 0.9059 | 0.9166 | 0.9112 |  |  | 0.8215 |  |
| StackingClassifier prefit | 0.9099 | 0.9099 | 0.9077 | 0.9126 | 0.9101 | 0.9686 | 0.9669 | 0.8198 | 0.4079 |


## Ensemble Confusion Matrices

| model | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: |
| VotingClassifier soft | 4511 | 489 | 416 | 4584 |
| VotingClassifier hard | 4524 | 476 | 417 | 4583 |
| StackingClassifier prefit | 4536 | 464 | 437 | 4563 |


## Optional Transformer Extension

| model | accuracy | balanced_accuracy | precision | recall | f1 | roc_auc | pr_auc | mcc | log_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tiny Transformer from scratch | 0.8779 | 0.8779 | 0.8982 | 0.8524 | 0.8747 | 0.9511 | 0.9499 | 0.7568 | 0.3000 |
| DistilBERT fine-tuned | 0.9369 | 0.9369 | 0.9296 | 0.9454 | 0.9374 | 0.9840 | 0.9831 | 0.8739 | 0.1888 |
| DeBERTa-v3-small fine-tuned | 0.9564 | 0.9564 | 0.9521 | 0.9612 | 0.9566 | 0.9895 | 0.9886 | 0.9128 | 0.1561 |


## Transformer Confusion Matrices

| model | tn | fp | fn | tp |
| --- | ---: | ---: | ---: | ---: |
| Tiny Transformer from scratch | 4517 | 483 | 738 | 4262 |
| DistilBERT fine-tuned | 4642 | 358 | 273 | 4727 |
| DeBERTa-v3-small fine-tuned | 4758 | 242 | 194 | 4806 |

