# Architecture

## Runtime MELON path

The host agent first produces the original tool actions. After it has tool
outputs, the integration passes the full cached tool-output history to a masked
reexecutor. The reexecutor uses a task-neutral user prompt and a no-side-effect
tool route. The ActionComparator describes the proposed calls and compares
their embeddings with the original calls. Similar actions above the configured
threshold are returned as a detection before the original side-effecting call
is allowed.

The host owns the model, conversation state, tool cache, and tool permission
checks. This repository supplies the comparison core, not a complete AgentDojo
pipeline registration.

## Auxiliary BIPIA path

The BIPIA loader creates clean and injected short windows conditioned on the
EmailQA question. A local multinomial Naive Bayes model scores each window.
TextWindowGuard scans overlapping windows at inference and returns the highest
score as an alert. This signal can support the action-level defense, but should
not independently authorize tool use.

## Qwen SFT path

The BIPIA SFT builder retains the original expected answer for clean and
poisoned EmailQA contexts. The split guard keeps official test examples out of
the trainer. LoRA/QLoRA trains only on completion tokens; prompt tokens are
masked in the loss. This is an answer-preservation fine-tuning experiment, not
MELON teacher distillation.
