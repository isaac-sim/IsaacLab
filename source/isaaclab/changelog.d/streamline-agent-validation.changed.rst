* **Breaking:** ``isaaclab --format`` now runs pre-commit once and returns a nonzero exit status when hooks fail or modify files.
  Callers that relied on automatic retries must inspect and accept the edits, then explicitly rerun the command.
