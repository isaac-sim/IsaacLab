* Fixed stale particle camera captures after OvStage geometry updates by waiting for renderer loading
  on every capture in scenes with published geometry and USD particles. Other scenes retain the
  first-capture-only wait to avoid extra steady-state render passes.
