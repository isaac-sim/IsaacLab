# Perf Gate — Project Sync Speaker Notes (2026-06-16)

Use this as a speaking guide. The bold lines are the main points to say out loud.
The text below each point is the simple explanation. Technical names are kept to
the minimum needed so this is easier to skim while talking.

---

## 0. Opening Summary

**The unified POC is now running the full performance gate end-to-end on the RTX
PRO 6000 runners.**

Today we got to a clean full run: all 9 benchmark jobs passed, and the final
summary job passed too.

**The biggest bottleneck was not the benchmarks anymore — it was repeatedly
building the same Docker image.**

Before today, each job could spend about 26 minutes building the same image. Now
we build the image once, store it in GitHub's container registry, and every job
pulls that image instead of rebuilding it.

**The image workaround is verified, not just theoretical.**

The image pushed successfully to GitHub's registry. The next gate run pulled that
image, skipped the build step, and passed the full matrix.

Quick numbers:

- Old path: about 26 minutes to build the image per job.
- New path: about 7.5 minutes to pull the image on a cold runner.
- Full gate today: 9/9 benchmark jobs passed.

---

## 1. Where the Unified POC Branch Stands

**The branch now combines Angelina's architecture with the runner fixes we needed
for the NVIDIA fleet.**

I started from Angelina's latest benchmark-gate work and made it work on the RTX
PRO 6000 GitHub runners. The branch is no longer just my older POC or just
Angelina's branch — it is the combined version.

**What Angelina's side gave us:**

- Cleaner separation between runtime config, baseline handling, GitHub context,
  and the oracle/verdict logic.
- Better structured benchmark result files.
- A compatibility contract so baselines are compared only when the runtime
  environment is comparable.
- More robust baseline branch management.

**What we layered on top:**

- Runner/container compatibility fixes.
- Permission fixes for JIT caches and renderer shader caches.
- Correct task IDs and task validation before spending GPU time.
- Per-task status reporting, so failed benchmarks do not get hidden by a green
  workflow.
- A sticky PR verdict comment.
- The GHCR prebuilt image path to avoid repeated cold builds.

**Branch status today:**

The branch is green on the full RTX PRO 6000 run. The remaining work is not
"make it run" anymore. It is now cleanup, production-hardening, baseline
collection, and alignment with upstream benchmark changes.

Reference only:

- Branch: `unified-POC`
- Latest successful gate run: `27648344312`
- Published image: `ghcr.io/nvidia-omniverse/isaaclab-perf-gate:sha-6f97bf4`

---

## 2. OVRTX Decision

**We dropped the OVRTX renderer backend from this POC matrix.**

One of the Shadow-Vision failures was not a gate bug. It was because the OVRTX
renderer requires an optional runtime package that is not installed in the CI
image.

**Why dropping it was the right short-term call:**

- OVRTX is optional, not required for the core gate.
- Adding it to the shared image would affect more than just this POC and would
  need broader review.
- Keeping it in the matrix made the gate fail for an environment/dependency
  reason, not because of an IsaacLab performance regression.
- Removing it reduced the matrix from 10 jobs to 9 jobs and let us focus on the
  backends that are actually available in the current image.

**Important nuance:**

This is not saying OVRTX should never be benchmarked. It just means we should not
block the first gate deployment on an optional dependency that is not currently
in the image. If the team wants OVRTX coverage later, we can add the dependency
properly and re-enable that cell.

---

## 3. The Renderer Crash Fix

**The remaining Shadow-Vision crash was a container cache permission issue, not a
benchmark logic bug.**

The renderer needed to write shader cache files inside the container. That
location was not writable by the user running the benchmark, so the renderer
failed to initialize correctly and later crashed with a CUDA error.

**Fix:**

We added a writable cache mount for the renderer, similar to what we already did
for Warp/CUDA JIT caches.

**Result:**

The Shadow-Vision jobs now run cleanly in the full matrix.

---

## 4. Docker Image / Registry Work

**The RTX PRO 6000 runners do not get a managed ECR image cache from the runners
team.**

I asked about the right cache setup. The answer was: there is no NVKS-specific
shared ECR cache. The only platform-provided cache is a lower-level DockerHub
mirror, which helps with base images but does not store our finished IsaacLab
image.

**That means any project-built image cache is our responsibility.**

We have two self-contained options:

**Option A: use our own AWS/ECR setup.**

We could provision our own AWS credentials and our own ECR repository. That would
let us keep using the existing ECR-style action. It would work, but we would own
the AWS account, IAM permissions, credential rotation, and costs.

**Option B: use GitHub's container registry.**

This is what we tried today. It already works. It uses GitHub's built-in access
model, lives next to the repo, and avoids introducing a separate AWS dependency.

**My recommendation for now:**

Keep GHCR as the practical path for the POC and likely the easiest long-term path
unless someone specifically wants AWS/ECR parity.

---

## 5. What Would Make the GHCR Path Production-Ready

**GHCR itself is not the hack. The temporary part is how manually we use it right
now.**

Right now we manually publish one image and point the gate at that image. That is
fine for proving the path, but production should be automatic.

Production checklist:

**1. Do not freeze the source code inside the image.**

The image should contain the heavy dependencies, not the exact PR code. At test
time, the fresh PR code should be mounted into the container. Otherwise we could
accidentally benchmark old code.

**2. Name images by dependency fingerprint.**

The image should change only when the dependencies change: Dockerfile, Python
requirements, Isaac Sim base image, etc. That way ordinary code changes do not
force a new image build.

**3. Build automatically if the image is missing.**

The gate should try to pull the image. If it is not there, build and publish it
once.

**4. Make access clean.**

The image is currently private and linked to the repo, which works for our
trusted runs. For broader usage, we would likely make it internal after
confirming no secrets are baked into it.

**5. Add cleanup.**

Old images should be deleted on a schedule so the registry does not fill up.

---

## 6. Antoine's Benchmark Refactor

**Antoine has four upstream benchmark refactor PRs, and they matter to us.**

The important point is not just that upstream is reorganizing benchmarks. It is
that our gate currently calls an old benchmark script that one of these PRs
removes.

Simple breakdown:

**Part 1: a shared benchmark core.**

This creates reusable benchmark building blocks and a structured output format.
It also captures versions, hardware, resources, and run identity. That overlaps
with our environment fingerprinting and provenance work.

**Part 2: new runtime and startup scripts.**

This replaces the current non-RL benchmark script with a new runtime benchmark.
This is the urgent one for us because our gate calls the old script today.

**Part 3: training benchmark.**

This creates one training benchmark entry point that can run different RL
libraries. This could be useful later if we want to gate training performance.

**Part 4: play/inference benchmark.**

This measures how fast a trained policy runs during inference. This could be
useful later if we want to gate inference performance.

**What we should take from this:**

- Short term: make sure our gate does not break when the old benchmark script is
  removed.
- Medium term: consume the new structured benchmark output instead of parsing our
  own custom result shape.
- Longer term: consider adding startup, training, and inference performance as
  separate gate dimensions.

---

## 7. Main Talking Points / Asks

**Talking point 1: The gate is now operational on RTX PRO 6000.**

It runs the matrix end-to-end and passes.

**Talking point 2: The expensive repeated image build problem is solved for the
POC.**

We moved from repeated image builds to a prebuilt image pull from GitHub's
registry.

**Talking point 3: The runners team confirmed there is no managed ECR cache for
this fleet.**

So we either own AWS/ECR ourselves or keep using GHCR.

**Talking point 4: We intentionally dropped OVRTX for now.**

It depends on an optional package not present in the image. Removing it makes the
gate reflect available supported backends instead of failing on missing optional
deps.

**Talking point 5: Antoine's refactor creates a near-term migration item.**

We need to switch from the old runtime benchmark script to the new one once the
upstream PR lands.

**Ask 1: Are we okay continuing with GHCR as the POC image registry path?**

My recommendation is yes.

**Ask 2: Do we want to productionize GHCR next, or prioritize baseline seeding
now that the gate is fast enough to run?**

Both are valid. If the goal is proving the gate's usefulness, baseline seeding is
probably the next most visible milestone.

**Ask 3: Who should own any optional OVRTX dependency decision?**

If OVRTX coverage is important, it probably needs a proper image/dependency
discussion, not a quick POC-only patch.

---

## 8. My Next Steps

Use this section if someone asks, "Okay, what are you doing next?"

### Path 1: Commit to GHCR

**The simplest next step is to keep going with GHCR, since it already worked
today.**

What this means:

- Treat GitHub's registry as the image cache for this POC.
- Keep the current fast path so the gate pulls the prebuilt image instead of
  rebuilding it.
- Turn the current manual process into a more automatic process.

What I would do next:

1. Keep using the current GHCR image for the POC so we can continue running the
   gate quickly.
2. Add a more production-like image flow:
   - build the image only when dependencies change
   - avoid baking the exact branch code into the image
   - mount the current checkout into the container at runtime
   - clean up old images
3. Keep GHCR private or internal depending on what access model we want.

How to explain it simply:

> GHCR is not the risky part. It's a normal registry and we already proved the
> org allows us to push and pull from it. The cleanup work is mostly automation:
> make sure images rebuild only when the environment changes, and make sure the
> gate always benchmarks the current branch code, not stale code baked into an
> old image.

Likely question:

**Q: Why choose GHCR if IsaacLab already uses ECR elsewhere?**

A: Because this runner fleet does not get a managed ECR cache, and GHCR already
works without adding another cloud account or new credentials. It is simpler for
the POC. ECR is still possible, but it adds AWS ownership work.

Likely question:

**Q: Is GHCR slower than ECR?**

A: For our practical case, GHCR already removed the expensive part: the 26-minute
build. The first pull took about 7.5 minutes on a cold runner. Warm pulls should
be much faster. ECR could be comparable, but the bigger win is using any shared
registry instead of rebuilding every job.

Likely question:

**Q: What is the biggest GHCR production concern?**

A: Correctness. We should avoid benchmarking code that was baked into an older
image. The long-term image should contain heavy dependencies only; the current
branch source should be mounted in when the benchmark runs.

### Path 2: Explore AWS/ECR

**The alternative next step is to explore whether owning our own AWS/ECR setup is
worth it.**

This would not require changing the existing IsaacLab CI system. We would bring
our own ECR repository and credentials for this gate.

What this means:

- Create or use an AWS account with an ECR repository.
- Create credentials that GitHub Actions can use.
- Store those credentials and the ECR repository URL in GitHub secrets or
  variables.
- Keep using the existing ECR-style action, but point it at our own ECR instead
  of relying on the runner fleet to provide one.

How to explain it simply:

> ECR is possible, but it is not something the RTX runner fleet gives us for
> free. We would own the AWS side: repository, permissions, secrets, and cost.
> The benefit is that it looks more like the existing IsaacLab CI path.

Likely question:

**Q: Why would we pick ECR at all?**

A: Familiarity and consistency. IsaacLab already has an ECR-based cache path on
other GPU runners, so using ECR would match that mental model. It may also be
easier to reuse existing code paths if the team strongly prefers ECR.

Likely question:

**Q: What is the downside of ECR?**

A: More operational ownership. Someone has to own AWS credentials, IAM
permissions, credential rotation, cost, and any debugging around cross-cloud
access from NVIDIA runners to AWS.

Likely question:

**Q: Is ECR technically better than GHCR?**

A: Not for the core problem. Both are registries. Either one solves the repeated
build problem. The decision is mostly ownership and operational preference:
GitHub-native registry versus AWS-managed registry.

### My Recommendation For The Sync

**My recommendation is to keep GHCR as the near-term path, while leaving ECR as a
backup option if the team strongly wants parity with the existing IsaacLab CI.**

Why:

- GHCR already worked today.
- It avoids AWS credentials and account ownership.
- It is enough to unblock fast gate runs and baseline collection.
- We can still switch to ECR later if someone wants to own the AWS side.

---

## 9. Next Steps From Antoine's Benchmark Refactor

Use this if someone asks, "What do Antoine's PRs mean for your POC?"

### Antoine's Refactor, In Plain English

**Antoine is cleaning up IsaacLab's benchmark scripts so they behave like one
benchmark system instead of a bunch of separate scripts.**

Before, IsaacLab had different benchmark scripts for different jobs:

- one script for simulator runtime/FPS
- one script for startup timing
- separate scripts for RL training
- separate play/inference paths

That makes it harder to maintain because each script can have its own flags,
output format, and helper code.

The refactor makes this cleaner:

- **Runtime benchmark:** measures how fast simulation steps run.
- **Startup benchmark:** measures how long startup takes.
- **Training benchmark:** measures how fast RL training runs.
- **Play/inference benchmark:** measures how fast a trained policy runs.
- **Shared output format:** each benchmark writes a structured result file with
  the run info, hardware, versions, and metrics.

How to explain it simply:

> Antoine is turning the benchmark scripts into a cleaner shared benchmark
> framework. Short term, we need to update our gate because the script we call is
> being replaced. Long term, this is good because the new benchmark output already
> contains much of the metadata and structure our POC was manually adding.

**The immediate risk is that our gate calls a benchmark script that upstream is
about to replace.**

Today the gate calls the old non-RL benchmark script. Antoine's Part 2 replaces
that with a new runtime benchmark. Once that lands, we need to update our gate or
it will call a file that no longer exists.

What I would do next:

1. Add a compatibility layer so the gate can run either the old benchmark script
   or the new runtime benchmark.
2. Update the result parser so it can read the new structured benchmark output.
3. Eventually switch fully to the new upstream output format.

How to explain it simply:

> This is less about performance right now and more about avoiding a future
> break. Upstream is changing the benchmark entry point. Since our gate is built
> around that entry point, we need to follow it instead of staying attached to the
> old script.

**There is also a positive side: Antoine's new output format overlaps with what
we built.**

The new benchmark output already records versions, hardware, resources, and run
identity. That is very close to our environment fingerprinting and provenance
work.

What that means:

- We can eventually read upstream's structured output directly.
- That may let us remove some custom parsing from our gate.
- It will make our baseline buckets line up better with upstream/OmniPerf data.

How to explain it simply:

> Instead of us maintaining a separate result format forever, we should probably
> consume the upstream benchmark result format once it lands. That makes our gate
> more future-proof and less custom.

**The multi-output feature could help baseline seeding.**

Antoine's benchmark refactor lets one benchmark run produce multiple output
formats. That means a single expensive benchmark run could produce:

- one output for our gate
- one output for OmniPerf or historical analysis

How to explain it simply:

> This matters because baseline collection is expensive. If one run can feed both
> the gate and the historical performance dataset, we avoid doing duplicate GPU
> work.

Likely question:

**Q: Should we block on Antoine's PRs before continuing?**

A: No. The current gate works now. But we should plan a migration task so we are
not surprised when the old benchmark script disappears.

Likely question:

**Q: Is this a big migration?**

A: It should be contained. The main pieces are changing the script we call,
aligning the command-line arguments, and updating how we parse the output. It is
important, but not a rewrite of the whole gate.

Likely question:

**Q: Does Antoine's work replace Angelina's architecture?**

A: No. They solve different layers. Angelina's work is the gate/oracle/baseline
architecture. Antoine's work is the benchmark script/output layer. We should
connect them, not replace one with the other.

Likely question:

**Q: What is the best long-term alignment?**

A: Use Angelina's gate architecture and consume Antoine's upstream benchmark
output format. That gives us a clean gate while staying aligned with upstream
benchmark infrastructure.

---

## 10. Bisection Agent Next Steps

Use this if someone asks, "How does this connect to an automated bisection
agent?"

### What A Bisection Agent Is, In Plain English

**A bisection agent is an automated helper that finds which commit introduced a
performance regression.**

Simple example:

- Last week was good.
- Today is bad.
- There are 100 commits in between.
- Instead of testing all 100 commits, the agent tests the middle commit.
- If the middle is good, the bad commit must be in the newer half.
- If the middle is bad, the bad commit must be in the older half.
- It keeps cutting the search space in half until it finds the first bad commit.

How to explain it simply:

> The perf gate tells us whether there is a regression. A bisection agent would
> answer the next question: which commit caused it?

### What The Bisection Agent Needs From Our Gate

**The gate needs to expose a clean, repeatable way to run one benchmark cell.**

For bisection, we do not want to run the full 9-job matrix for every commit. That
would be too expensive. We want to run the specific task/backend that regressed.

For example:

- Cartpole + Newton
- Shadow-Vision + PhysX
- Factory + PhysX

The agent needs:

- the task/backend that failed
- the baseline or threshold to compare against
- the commit range to search
- a way to run the benchmark on one commit
- a clear pass/fail result
- the logs and artifacts for the final suspected commit

How to explain it simply:

> For normal CI, we run the whole matrix. For bisection, we should run only the
> one cell that regressed, because the goal is to find the culprit quickly without
> wasting GPU time.

### Why Today's Work Helps Bisection

**The GHCR image path makes bisection much more realistic.**

Without the prebuilt image, each bisection step could spend about 26 minutes just
building the image. That would make bisection painfully slow.

With the prebuilt image:

- each tested commit can pull the image instead of building it
- most of the time is spent on the benchmark itself
- the agent can test multiple commits without wasting hours on repeated setup

How to explain it simply:

> The image fix does not only help the gate. It also makes future bisection
> practical, because the agent can focus on benchmarking commits instead of
> rebuilding the environment over and over.

### How Antoine's Refactor Helps Bisection

**Antoine's structured benchmark output would make the bisection agent easier to
write.**

If every benchmark writes a standard result file, the agent does not need custom
parsing for every benchmark type. It can read one consistent format and ask:

- did this commit pass?
- what was the measured FPS?
- what hardware/software was used?
- was the result comparable to the baseline?

How to explain it simply:

> A bisection agent needs stable inputs and outputs. Antoine's benchmark refactor
> gives us a more stable output format, while Angelina's gate architecture gives
> us the baseline/oracle logic. The bisection agent would sit on top of both.

### Concrete Next Steps For Bisection

**Step 1: Make single-cell runs first-class.**

We already added local task filtering. The CI side should also make it easy to
run exactly one task/backend, not the full matrix.

**Step 2: Define the pass/fail contract.**

The agent needs one clear answer per commit: pass, fail, or invalid run. It
should not scrape vague logs.

**Step 3: Reuse the prebuilt image path.**

Every bisection run should use the registry image so it does not rebuild the
environment for each commit.

**Step 4: Keep the environment stable while bisecting.**

The agent should use the same GPU type, same image, same task config, and same
baseline comparison throughout the search. Otherwise it might confuse environment
noise with a real regression.

**Step 5: Produce a short report.**

At the end, the agent should output:

- suspected first bad commit
- good commit / bad commit range
- task/backend that regressed
- measured numbers at the key commits
- links to logs/artifacts
- confidence level and caveats

How to explain it simply:

> The next step is not to build a fancy agent immediately. First, make the gate
> easy to call in a single-cell, repeatable way. Once that exists, the bisection
> logic is mostly orchestration: pick a commit, run the cell, read pass/fail, cut
> the search range in half, repeat.

Likely question:

**Q: Would the bisection agent run the whole matrix?**

A: No. It should start from the failed task/backend and run only that cell. Full
matrix bisection would be too expensive.

Likely question:

**Q: Does bisection need GHCR?**

A: It does not strictly need GHCR, but GHCR makes it practical. Without a
prebuilt image, each tested commit spends a lot of time rebuilding the same
environment.

Likely question:

**Q: Does Antoine's refactor replace the need for a bisection agent?**

A: No. Antoine's refactor standardizes benchmark output. The bisection agent
would use that output to decide whether each commit is good or bad.

Likely question:

**Q: What is the biggest bisection risk?**

A: Noise. Performance numbers vary. The agent needs stable hardware, consistent
environment, and possibly repeated samples around borderline commits so it does
not blame the wrong change.

---

## 11. Likely Questions And Short Answers

**Q: Why not just provision our own AWS credentials and use ECR?**

We can. It would work. The tradeoff is that we would now own AWS credentials,
permissions, rotation, and cost. GHCR already works with the repo's built-in
GitHub permissions, so it is simpler unless we specifically need ECR parity.

**Q: Is GHCR production-ready, or just a demo hack?**

GHCR is a real registry. The registry is fine. What needs production work is the
automation around it: rebuilding only when dependencies change, mounting fresh
PR code, cleanup, and access settings.

**Q: Does the current GHCR image test the latest PR code?**

For the POC run, it uses the code baked into the published image. For production,
we should change that so the image contains dependencies only and the current PR
source is mounted in at runtime.

**Q: Why did we remove OVRTX?**

Because the current CI image does not include the optional OVRTX runtime package.
Keeping that cell made the gate fail for a missing optional dependency, not a
performance regression. We can re-add it later once the dependency is officially
part of the image.

**Q: What did Angelina's changes add that the old POC did not have?**

Cleaner architecture: baseline management, GitHub context, runtime compatibility
contracts, richer result files, and more structured oracle logic. We then made it
runner-compatible and got it green.

**Q: What is the biggest remaining technical risk?**

The upstream benchmark refactor. One of Antoine's PRs removes the script our gate
currently calls. We need to migrate to the new script/output format before that
lands or soon after.

**Q: What should happen next?**

My suggested order:

1. Keep the GHCR fast path for the POC.
2. Decide whether to baseline-seed next.
3. Add compatibility with Antoine's new benchmark entry point.
4. Later, productionize the image path so it is automatic and not manually
   pinned.
