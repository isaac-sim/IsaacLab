* Reused the NIST success-guided sampler for conveyor reset rows, with completed transfers as the
  success signal and standard PPO. Removed the task-specific progress curriculum and checkpoint
  extension; sampling history now starts fresh when an environment is recreated.
