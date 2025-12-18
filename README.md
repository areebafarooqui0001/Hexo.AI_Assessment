# Autonomous MLE Agent

## One-Line Run Command
```bash
python3 automl.py --dataset_path ./data/<folder-name>
```

## Overview

The AutoMLEAgent is a fully autonomous system for building and running competition-ready ML pipelines. It inspects the dataset to determine its modality and key columns, then uses an LLM to generate a custom training script.
Its self-correcting loop allows the agent to debug itself: if the generated code errors out, the full traceback is fed back into the LLM for automatic fixing. If recovery fails, a deterministic fallback still produces a valid submission—no competition-specific hardcoding required.

## Submission

--------------------
| competition_id                                  | score   | gold_threshold   | silver_threshold   | bronze_threshold   | median_threshold   | any_medal   | gold_medal   | silver_medal   | bronze_medal   | above_median   | submission_exists   | valid_submission   | is_lower_better   | created_at                 | submission_path                     
                                                      |
|:------------------------------------------------|:--------|:-----------------|:-------------------|:-------------------|:-------------------|:------------|:-------------|:---------------|:---------------|:---------------|:--------------------|:-------------------|:------------------|:---------------------------|:------------------------------------------------------------------------------------------|
| tabular-playground-series-may-2022              | 0.81727 | 0.99823          | 0.99822            | 0.99818            | 0.972675           | False       | False        | False          | False          | False          | True                | True               | False             | 2025-12-19T01:10:37.359180 | ../mlebench_results/tabular-playground-series-may-2022/seed_0/submission.csv              |
| spooky-author-identification                    | 1.08474 | 0.16506          | 0.26996            | 0.29381            | 0.418785           | False       | False        | False          | False          | False          | True                | True               | True              | 2025-12-19T00:52:48.822728 | ../mlebench_results/spooky-author-identification/seed_0/submission.csv                    |
| the-icml-2013-whale-challenge-right-whale-redux | 0.5     | 0.98961          | 0.95017            | 0.90521            | 0.86521            | False       | False        | False          | False          | False          | True                | True               | False             | 2025-12-19T01:12:39.235519 | ../mlebench_results/the-icml-2013-whale-challenge-right-whale-redux/seed_0/submission.csv |
| text-normalization-challenge-english-language   |         | 0.99724          | 0.99135            | 0.99038            | 0.99037            | False       | False        | False          | False          | False          | True                | False              | False             | 2025-12-19T01:01:54.316739 | ../mlebench_results/text-normalization-challenge-english-language/seed_0/submission.csv   |
| siim-isic-melanoma-classification               |         |                  |                    |                    |                    |             |              |                |                |                | false               | false              |                   |                            |                                     
                                                      |
--------------------

## Evaluation (Mean ± SEM)

| Dataset | Data Modality | Evaluation Metric (used by script) | Mean Private Score ± SEM | Individual runs (Private Scores) | Target Value |
|---|---|---|---:|---|---|
| SIIM-ISIC Melanoma Classification (siim-isic-melanoma-classification) | Image | ROC AUC (binary AUC) | 0.6672 ± 0.0034 | 0.6613, 0.6730, 0.6674 | Maximize |
| Tabular Playground Series — May 2022 (tabular-playground-series-may-2022) | Tabular | ROC AUC (binary AUC) | 0.9445 ± 0.0108 | 0.92310, 0.95255, 0.95794 | Maximize |
| Spooky Author Identification (spooky-author-identification) | Text | Multiclass LogLoss (probabilistic outputs) | 0.5672 ± 0.0186 | 0.54913, 0.54801, 0.60435 | Minimize |
| Text Normalization Challenge — English Language (text-normalization-challenge-english-language) | Text (Seq2Seq) | Exact-match Accuracy | 0.9729 ± 0.0074 | 0.98063, 0.95817, 0.98002 | Maximize |
| The ICML 2013 Whale Challenge — Right Whale Redux (the-icml-2013-whale-challenge-right-whale-redux) | Image / Audio | Not Evaluated (script configured for AUC) | N/A | Competition Closed | N/A |


## Future Improvements

Planned upgrades include stronger modality-specific fallback models (simple CNN/RNN baselines) and richer execution-time logging to improve the agent’s ability to diagnose and fix failures automatically.