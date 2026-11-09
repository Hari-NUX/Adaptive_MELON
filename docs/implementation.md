# Paper implementation and BIPIA extension

## Two separate experiments

MELON is a runtime defense, not a text-classifier training method. Its core
signal is whether tool actions remain similar after masking the user's task and
re-executing over cached tool outputs. The MELONRunner and ActionComparator
modules implement that framework-neutral action-comparison core. The masked
reexecutor is an adapter supplied by the host agent.

The local BIPIA model is a separate auxiliary experiment. It classifies short
retrieved-text windows conditioned on the user question. BIPIA does not contain
the original-run and masked-run tool traces required to train MELON directly.
Do not describe BIPIA F1/AUC as AgentDojo UA/ASR or as a paper reproduction.

## MELON action comparison

Create an embedding provider and comparator:

~~~python
from melon_guard.teacher.action_comparator import ActionComparator
from melon_guard.teacher.melon_runner import MELONRunner
from melon_guard.teacher.openai_embedder import OpenAIEmbedder

comparator = ActionComparator(OpenAIEmbedder(), threshold=0.8)
runner = MELONRunner(comparator, masked_reexecutor=masked_reexecute)
~~~

The host adapter calls runner.inspect(original_actions, tool_output_cache)
before allowing the original actions to execute. masked_reexecute receives the
neutral MELON task, the cached outputs, and execute_tools=False; it must return
proposed tool calls without running them. Wire this flag to a no-side-effect
tool router. Never let the probe send messages, transfer funds, delete records,
or make other live changes.

The comparator serializes tool names and focused arguments, caches embeddings,
and checks each original action against the masked-run action bank. The default
threshold is 0.8. For paper-aligned results, use the paper's model, attack
settings, and AgentDojo tasks; report per-case denominators, BU, UA, ASR, and
benign false-positive rate. The generic adapter here is not a complete
AgentDojo pipeline registration.

## Train the BIPIA auxiliary detector

Download only BIPIA files if BrowseSafe-Bench is not needed:

~~~powershell
.\scripts\download_datasets.ps1 -SkipBrowseSafe
~~~

Train and evaluate the dependency-free local model:

~~~powershell
py .\scripts\train_bipia_detector.py
~~~

The script reads only BIPIA EmailQA train data and its attack templates while
fitting, then evaluates on the official test contexts and disjoint test attack
templates. It writes the local model and JSON metrics under runs/bipia_detector
(ignored by Git). The default benchmark root is datasets/gmail/raw. An
original BIPIA checkout can also be passed with --benchmark-root.

Use the model on retrieved text:

~~~powershell
py .\scripts\predict_bipia.py --task "Find the invoice total" --file .\retrieved.txt
~~~

The detector scans overlapping 200-character windows and reports the highest
Naive Bayes score. Scores are not calibrated confidence estimates. Evaluate
precision, recall, F1, ROC-AUC, benign false positives, and the Always-Attack
baseline; choose any deployment threshold using a validation split, never the
official test split.

## Qwen supervised fine-tuning on BIPIA

This is a dataset-specific answer-preservation experiment. It trains Qwen to
answer the original EmailQA question while the email context may contain an
injected instruction. It is not MELON teacher distillation because no tool
actions or masked-run traces are available in BIPIA.

Prepare train and validation records only from the official training split.
Prepare the test file separately when ready to evaluate:

~~~powershell
py .\scripts\prepare_bipia_sft.py
py .\scripts\prepare_bipia_sft.py --include-test
~~~

The second command overwrites the same train/validation files and additionally
writes test.jsonl. The training script checks split metadata and refuses test
records:

~~~powershell
py .\scripts\train_qwen_sft.py --model Qwen/Qwen3-0.6B --method lora
~~~

For QLoRA, use a Linux/WSL2 CUDA environment with bitsandbytes and a compatible
PyTorch build:

~~~powershell
pip install -r requirements-qlora.txt
py .\scripts\train_qwen_sft.py --model Qwen/Qwen3-0.6B --method qlora
~~~

Compare the base model with the adapter on the official test file:

~~~powershell
py .\scripts\evaluate_qwen_bipia.py --adapter .\runs\qwen3_0_6b_lora
~~~

The evaluator reports exact match and token F1 against the BIPIA ideal answer,
separated for clean and injected contexts. These answer-quality metrics do not
measure attack success; use BIPIA's official judge for ASR and AgentDojo for the
paper's BU/UA/ASR comparison.

## AgentDojo metrics

Export one JSON object per evaluated case with attacked,
user_task_success, and, for attacked cases, attack_success. Add blocked
for benign cases to compute FPR. Optional defense, agent, and attack
fields group results:

~~~powershell
py .\scripts\evaluate_agentdojo_results.py .\runs\agentdojo\cases.jsonl
~~~

This aggregator does not run AgentDojo. A full comparison requires connecting
the MELON adapter to AgentDojo and running the baseline and proposed defense on
the same cases and model configuration.

## Improvement protocol

1. Record a faithful MELON baseline before changing the decision rule.
2. Add the BIPIA text score as an auxiliary alert signal; keep tool permissions
   as the final control for sensitive actions.
3. Compare MELON alone, text detector alone, and the combined system on the same
   AgentDojo tasks. Report changes in UA, ASR, BU, FPR, latency, and embedding
   calls.
4. Tune thresholds on validation cases only. Keep attack families and tasks
   separated between training and final evaluation where the benchmark allows.
5. Audit false positives and false negatives, then add those patterns to
   training or policies without modifying the held-out test set.

Fine-tuning Qwen from teacher traces is a later, separate experiment. It needs
audited trajectory labels and a compatible training setup; BIPIA Q/A examples
alone are not MELON teacher labels.
