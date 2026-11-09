# BIPIA EmailQA local detector run

Run date: 2026-11-09

Dataset: Microsoft BIPIA EmailQA, official train and test splits

Model: dependency-free word/bigram/character n-gram multinomial Naive Bayes

Window size: 200 characters

Decision threshold: 0.5

Seed: 42

## Held-out results

| Metric | Result |
| --- | ---: |
| Test windows | 550 |
| Clean windows | 100 |
| Injected windows | 450 |
| Accuracy | 84.9% |
| Always-attack accuracy baseline | 81.8% |
| Precision | 97.9% |
| Recall | 83.3% |
| F1 | 90.0% |
| ROC-AUC | 93.7% |
| True positives | 375 |
| False positives | 8 |
| True negatives | 92 |
| False negatives | 75 |

The script pairs two contexts with each attack prompt and places the prompt at
the start, middle, or end. It uses only train contexts and train attack
templates when fitting. Test contexts and test attack templates are separate.
The small, attack-heavy constructed window set explains why accuracy alone is
misleading; precision, recall, F1, AUC, and the majority baseline are included.

## Interpretation

This result is for text-window classification only. It does not show that the
detector outperforms MELON on AgentDojo. The paper reports agent task utility
and attack success, which require running the action-level defense and an agent
on AgentDojo. BIPIA does not provide MELON's masked-run tool-action traces.

The training command writes the full JSON report and model under
runs/bipia_detector; those generated files are ignored by Git. Re-run the
script to reproduce the run with the downloaded BIPIA files.
