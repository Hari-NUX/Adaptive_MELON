# MELON teacher trace generation

## Purpose

Teacher traces are needed only for a separate distillation experiment. They
must represent MELON's runtime decision from the original and masked
trajectories. BIPIA question/answer rows are not teacher labels.

## Required trajectory fields

Keep one record per inspected agent step, with:

- scenario, task, attack family, tool, model, and seed identifiers
- original user task and observed tool-output cache
- original proposed tool action
- masked-run neutral task and proposed tool actions
- action descriptions, embedding model, similarity scores, and threshold
- resulting MELON decision
- whether executing the action would complete the user objective or attack
- reviewer decision and rationale for ambiguous cases

Generate traces only from training scenarios. The runtime action comparator
can produce candidate labels, but label validation must check actual tool
effects and human-audit false positives. Partition by scenario and attack
template before training; do not create train/test splits from adjacent steps
of the same trajectory. Keep official AgentDojo test cases sealed until final
evaluation.

This repository currently provides the action-comparison core and a case
schema contract, but it does not run AgentDojo or generate a complete labeled
teacher corpus automatically.
