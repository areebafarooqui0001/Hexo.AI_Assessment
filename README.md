# Autonomous MLE Agent

## One-Line Run Command
```bash
python3 automl.py --dataset_path ./data/<folder-name>
```

## Overview

The AutoMLEAgent is a fully autonomous system for building and running competition-ready ML pipelines. It inspects the dataset to determine its modality and key columns, then uses an LLM to generate a custom training script.
Its self-correcting loop allows the agent to debug itself: if the generated code errors out, the full traceback is fed back into the LLM for automatic fixing. If recovery fails, a deterministic fallback still produces a valid submission—no competition-specific hardcoding required.

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