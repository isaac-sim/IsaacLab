# CI Required-Check Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the exact legacy required-check names for docs-only pull requests without rerunning demo or kit-less rendering tests.

**Architecture:** Add three static jobs to `.github/workflows/build.yaml`. Each job uses `always()` so it runs after an intentionally skipped dependency, checks the shared `changes` output, and otherwise mirrors its matrix test job's aggregate result. Add a focused static pytest file to prevent accidental removal or renaming of these compatibility contexts.

**Tech Stack:** GitHub Actions YAML, Bash, pytest, `./isaaclab.sh -p -m pytest`.

## Global Constraints

- Do not modify branch-protection settings.
- Do not duplicate or launch the existing expensive test jobs.
- Docs-only changes must pass the compatibility jobs while the underlying test jobs remain skipped.
- Relevant changes must fail the compatibility job if its underlying test job fails or is cancelled.
- Keep changes limited to CI workflow configuration and its focused test.

---

### Task 1: Add compatibility jobs

**Files:**
- Modify: `.github/workflows/build.yaml` after the test jobs and before the disabled quarantine section.

**Interfaces:**
- Consumes: `needs.changes.outputs.should_run`, `needs.changes.result`, and the aggregate result of the existing matrix jobs.
- Produces: exact check names `standalone demos (headless, Kit)`, `standalone demos (headless, non-Kit)`, and `rendering-correctness-kitless (legacy)`.

- [ ] **Step 1: Add the three static job definitions.**

  Each job must use `needs: [changes, <existing-matrix-job>]`, `if: always() && github.event_name == 'pull_request'`, and a Bash step that fails if change detection failed or if relevant tests did not succeed. When `should_run` is false, the step exits zero before inspecting the skipped test result.

- [ ] **Step 2: Inspect the resulting workflow structure.**

  Run `sed -n '820,980p' .github/workflows/build.yaml` and verify the jobs are at the top level under `jobs`, have exact `name:` values, and do not contain test execution actions.

- [ ] **Step 3: Commit the workflow change.**

  Run `git add .github/workflows/build.yaml && git commit -m "ci: publish legacy required test checks"`.

### Task 2: Add focused regression coverage

**Files:**
- Create: `.github/workflows/test_required_check_compat.py`.

**Interfaces:**
- Consumes: the checked-in `.github/workflows/build.yaml` text.
- Produces: static assertions protecting exact names, `always()` conditions, dependencies, and the docs-only success branch.

- [ ] **Step 1: Write tests for all three compatibility jobs.**

  Read `build.yaml`, locate each exact job block, and assert that it has `if: always() && github.event_name == 'pull_request'`, depends on the expected matrix job, and contains the `should_run` false success branch.

- [ ] **Step 2: Run the focused test.**

  Run `./isaaclab.sh -p -m pytest .github/workflows/test_required_check_compat.py -q`.

- [ ] **Step 3: Commit the regression test.**

  Run `git add .github/workflows/test_required_check_compat.py && git commit -m "test: cover legacy required CI checks"`.

### Task 3: Verify and prepare the pull request

**Files:**
- Verify: `.github/workflows/build.yaml`, `.github/workflows/test_required_check_compat.py`.

- [ ] **Step 1: Run the focused regression test and pre-commit.**

  Run `./isaaclab.sh -p -m pytest .github/workflows/test_required_check_compat.py -q`, then `./isaaclab.sh -f`.

- [ ] **Step 2: Review the final diff.**

  Run `git diff origin/develop...HEAD --check` and `git diff origin/develop...HEAD --stat`.

- [ ] **Step 3: Push the feature branch to the `antoine` fork.**

  Run `git push -u antoine antoiner/ci-required-check-compat`.

- [ ] **Step 4: Open the pull request.**

  Create a PR targeting `develop` that explains the stale required contexts, the exact compatibility names, and that no expensive tests are duplicated.
