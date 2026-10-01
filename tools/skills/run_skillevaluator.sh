#!/usr/bin/env bash
# Run SkillEvaluator Tier 1 / 2A / 2B / 3 against touched skills/user/ directories.
#
# Usage:
#   ./tools/skills/run_skillevaluator.sh [skill_dir ...]
#
# With no arguments, runs against all three PR skills:
#   isaaclab-building-environments
#   isaaclab-converting-direct-to-manager
#   isaaclab-migrating-from-isaac-gym
#
# Tier 2A / 2B / 3 require NVIDIA_API_KEY to be set in the environment.
# Tier 3 also requires Docker.
#
# Examples:
#   export NVIDIA_API_KEY=nvapi-...
#   ./tools/skills/run_skillevaluator.sh
#   ./tools/skills/run_skillevaluator.sh skills/user/isaaclab-building-environments

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_SKILLS=(
  skills/user/isaaclab-building-environments
  skills/user/isaaclab-converting-direct-to-manager
  skills/user/isaaclab-migrating-from-isaac-gym
)

if [ $# -gt 0 ]; then
  SKILLS=("$@")
else
  SKILLS=("${DEFAULT_SKILLS[@]}")
fi

if ! command -v skillevaluator &>/dev/null; then
  echo "ERROR: skillevaluator not found. Install with:"
  echo "  pip install 'skillevaluator[tier2,security,tier3] @ git+https://github.com/NVIDIA/SkillEvaluator.git@v0.1.0'"
  exit 1
fi

echo "skillevaluator $(skillevaluator --version 2>&1)"
echo "Skills to evaluate: ${SKILLS[*]}"
echo ""

# ── Tier 1: schema / lint ────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo " TIER 1 — schema / lint (no API key required)"
echo "════════════════════════════════════════════════════════════"
tier1_failed=0
for skill in "${SKILLS[@]}"; do
  echo ""
  echo "── Tier 1: ${skill} ──"
  if ! skillevaluator validate "${skill}" \
        -c \
        --external \
        --no-dedup \
        --checks schema,version,pii,license,code-integrity,unicode,quality,lint; then
    tier1_failed=1
  fi
done

# ── Tier 2A: intra-skill context dedup ───────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " TIER 2A — intra-skill context dedup (requires NVIDIA_API_KEY)"
echo "════════════════════════════════════════════════════════════"
if [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "SKIPPED: NVIDIA_API_KEY not set"
else
  for skill in "${SKILLS[@]}"; do
    echo ""
    echo "── Tier 2A: ${skill} ──"
    skillevaluator context-optimization-check "${skill}" \
      || echo "WARNING: Tier 2A failed for ${skill} (advisory — dedup endpoint may be unavailable)"
  done
fi

# ── Tier 2B: intra-repo similarity scan ──────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " TIER 2B — intra-repo similarity scan (requires NVIDIA_API_KEY)"
echo "════════════════════════════════════════════════════════════"
if [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "SKIPPED: NVIDIA_API_KEY not set"
else
  echo ""
  skillevaluator similarity-check skills/user/ --type skill \
    || echo "WARNING: Tier 2B failed (advisory — dedup endpoint may be unavailable)"
fi

# ── Tier 3: live evaluation (advisory) ───────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " TIER 3 — live evaluation / advisory (requires NVIDIA_API_KEY + Docker)"
echo "════════════════════════════════════════════════════════════"
if [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "SKIPPED: NVIDIA_API_KEY not set"
elif ! command -v docker &>/dev/null; then
  echo "SKIPPED: docker not found"
else
  for skill in "${SKILLS[@]}"; do
    echo ""
    echo "── Tier 3: ${skill} ──"
    skillevaluator tier3 evaluate "${skill}" \
        --agents codex \
        --env-mode docker \
        --skill-workspace-mode group \
      || echo "WARNING: Tier 3 eval failed for ${skill} (advisory)"
  done
fi

echo ""
echo "════════════════════════════════════════════════════════════"
if [ "${tier1_failed}" -eq 1 ]; then
  echo " RESULT: Tier 1 FAILED — fix schema/lint errors above"
  exit 1
else
  echo " RESULT: Tier 1 passed"
fi
