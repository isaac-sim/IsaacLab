* Fixed the RSL-RL training benchmark to resolve ``--checkpoint`` like the training entrypoint, so distillation
  benchmarks initialize their teacher from an explicit checkpoint instead of searching the student experiment's runs.
