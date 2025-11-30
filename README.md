# Autonomous MLE Agent

## One-Line Run Command
```bash
python3 automl.py --dataset_path ./data/<folder-name>
```

## Overview

The AutoMLEAgent is a fully autonomous system for building and running competition-ready ML pipelines. It inspects the dataset to determine its modality and key columns, then uses an LLM to generate a custom training script.
Its self-correcting loop allows the agent to debug itself: if the generated code errors out, the full traceback is fed back into the LLM for automatic fixing. If recovery fails, a deterministic fallback still produces a valid submission—no competition-specific hardcoding required.

## Evaluation (Mean ± SEM)

| Dataset | Score |
| Melanoma (SIIM-ISIC) | 0.6800 ± 0.0041 |
| Tabular May 2022 | 0.9455 ± 0.0107 |
| Spooky Author ID | 0.5764 ± 0.0177 |
| Text Normalization | 0.9748 ± 0.0077 |
| Right Whale | Competition Closed |

## Future Improvements

Planned upgrades include stronger modality-specific fallback models (simple CNN/RNN baselines) and richer execution-time logging to improve the agent’s ability to diagnose and fix failures automatically.