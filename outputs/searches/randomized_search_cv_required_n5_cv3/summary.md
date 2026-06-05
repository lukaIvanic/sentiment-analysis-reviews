# RandomizedSearchCV coverage summary

Run configuration: `n_iter=5`, `cv=3`, `scoring=f1`, `random_state=42`.

| Classifier | Best CV F1 | Test accuracy | Test F1 | ROC-AUC | PR-AUC | MCC | Log-loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| MultinomialNB | 0.8800 | 0.8819 | 0.8829 | 0.9511 | 0.9487 | 0.7639 | 0.3044 |
| ComplementNB | 0.8800 | 0.8819 | 0.8829 | 0.9511 | 0.9487 | 0.7639 | 0.3044 |
| LogisticRegression | 0.8946 | 0.9029 | 0.9035 | 0.9655 | 0.9651 | 0.8059 | 0.2716 |
| LinearSVC | 0.9081 | 0.9168 | 0.9172 | 0.9728 | 0.9717 | 0.8336 | nan |
| SGDClassifier | 0.8946 | 0.8975 | 0.8987 | 0.9632 | 0.9626 | 0.7952 | 0.3744 |
| PassiveAggressiveClassifier | 0.8986 | 0.9054 | 0.9054 | 0.9656 | 0.9637 | 0.8108 | nan |
| RandomForestClassifier | 0.8666 | 0.8743 | 0.8765 | 0.9463 | 0.9440 | 0.7491 | 0.5558 |
| ExtraTreesClassifier | 0.8762 | 0.8745 | 0.8776 | 0.9494 | 0.9474 | 0.7499 | 0.5944 |
| XGBoostClassifier | 0.8427 | 0.8412 | 0.8475 | 0.9274 | 0.9263 | 0.6848 | 0.3842 |
| LightGBMClassifier | 0.8836 | 0.8922 | 0.8927 | 0.9608 | 0.9597 | 0.7844 | 0.2607 |

## Best parameter samples

| Classifier | Best sampled parameters |
|---|---|
| MultinomialNB | `{"classifier__alpha": 0.1, "tfidf__max_df": 0.9, "tfidf__max_features": 30000, "tfidf__min_df": 3, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
| ComplementNB | `{"classifier__alpha": 0.1, "tfidf__max_df": 0.9, "tfidf__max_features": 30000, "tfidf__min_df": 3, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
| LogisticRegression | `{"classifier__C": 1.0, "tfidf__max_df": 0.9, "tfidf__max_features": 30000, "tfidf__min_df": 5, "tfidf__ngram_range": [1, 1], "tfidf__sublinear_tf": true}` |
| LinearSVC | `{"classifier__C": 0.3, "tfidf__max_df": 0.9, "tfidf__max_features": 30000, "tfidf__min_df": 3, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
| SGDClassifier | `{"classifier__alpha": 0.001, "classifier__loss": "modified_huber", "classifier__penalty": "l2", "tfidf__max_df": 0.95, "tfidf__max_features": 50000, "tfidf__min_df": 5, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
| PassiveAggressiveClassifier | `{"classifier__C": 2.0, "classifier__loss": "hinge", "tfidf__max_df": 0.9, "tfidf__max_features": 50000, "tfidf__min_df": 2, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": false}` |
| RandomForestClassifier | `{"classifier__criterion": "gini", "classifier__max_depth": 120, "classifier__max_features": "log2", "classifier__min_samples_leaf": 2, "classifier__n_estimators": 200, "tfidf__max_df": 0.95, "tfidf__max_features": 30000, "tfidf__min_df": 2, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
| ExtraTreesClassifier | `{"classifier__criterion": "gini", "classifier__max_depth": 120, "classifier__max_features": "log2", "classifier__min_samples_leaf": 2, "classifier__n_estimators": 200, "tfidf__max_df": 0.95, "tfidf__max_features": 30000, "tfidf__min_df": 2, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
| XGBoostClassifier | `{"classifier__colsample_bytree": 1.0, "classifier__learning_rate": 0.1, "classifier__max_depth": 3, "classifier__n_estimators": 200, "classifier__reg_lambda": 0.5, "classifier__subsample": 1.0, "tfidf__max_df": 0.9, "tfidf__max_features": 50000, "tfidf__min_df": 5, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
| LightGBMClassifier | `{"classifier__colsample_bytree": 1.0, "classifier__learning_rate": 0.1, "classifier__min_child_samples": 20, "classifier__n_estimators": 200, "classifier__num_leaves": 31, "classifier__reg_lambda": 0.5, "classifier__subsample": 0.8, "classifier__subsample_freq": 1, "tfidf__max_df": 0.95, "tfidf__max_features": 50000, "tfidf__min_df": 2, "tfidf__ngram_range": [1, 2], "tfidf__sublinear_tf": true}` |
