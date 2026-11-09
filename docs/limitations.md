# Limitations

- The local Naive Bayes model detects benchmark-generated text windows. It is
  not a general prompt-injection detector and is not calibrated.
- BIPIA supplies text contexts and injection templates, not original and masked
  agent trajectories or tool-action labels.
- The framework-neutral MELON runner requires a host-agent adapter. The adapter
  must guarantee that masked probes cannot perform live side effects.
- The OpenAI embedding provider requires an API key and network access. The
  action-comparison core accepts alternative local embedders.
- Qwen SFT trains answer preservation on BIPIA EmailQA, not MELON teacher
  decisions. LoRA/QLoRA performance depends on hardware and the base model.
- BIPIA exact-match/token-F1 results are not comparable to AgentDojo BU, UA, or
  ASR. An improved paper-level result requires AgentDojo evaluation.
