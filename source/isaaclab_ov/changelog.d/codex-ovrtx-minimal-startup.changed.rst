* Applied OVRTX's public startup mode flags to Minimal-only workloads, avoiding unused PT/RT2
  resources. Released retained graphics resources when successive simulations changed mode
  availability, and rejected late incompatible cameras before native fallback could change their output.
