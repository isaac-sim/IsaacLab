# Perf Smoke Image Cache README

This README explains the image build/cache issue in simple terms, why it matters
for the performance smoke test, and what options we have.

## Short Version

The performance smoke test runs benchmarks in Docker containers on NVIDIA RTX PRO 6000
GitHub Actions runners.

Before each benchmark can run, the workflow needs a Docker image that contains:

* Isaac Sim
* Isaac Lab
* Python dependencies
* benchmark tooling

Right now, the expensive part is getting that Docker image ready.

If every benchmark job builds the image from scratch, we waste a lot of time.
For example, a cold image build can take around 26 minutes. If the smoke test has 9 or
10 benchmark jobs, that can become 9 or 10 separate cold builds running in
parallel. That is slow and wasteful.

The goal is simple:

* build the image once when needed
* reuse it across benchmark jobs
* avoid rebuilding it for every matrix cell

## What The Perf Smoke Is Doing

The workflow is `.github/workflows/perf-smoke-test.yaml`.

It has three main phases:

1. `config`
   Loads the benchmark matrix and image configuration.

2. `bench`
   Runs one benchmark job per task/backend pair. Each job needs the CI Docker
   image before it can run the benchmark.

3. `aggregate`
   Collects benchmark results, compares them to baselines, and posts the final
   verdict.

The slow part is in the `bench` jobs. Each matrix cell needs the same basic
Docker image.

## Important Terms

### Runner

A runner is the machine that executes a GitHub Actions job.

For this perf smoke, we care about two runner groups:

* L40S runners
* NVKS RTX PRO 6000 runners

The RTX PRO 6000 runners are the target fleet for this POC because they are the
NVIDIA-managed runners we are trying to use for the smoke test.

### Docker Image

A Docker image is like a saved machine environment.

Instead of installing Isaac Sim, Isaac Lab, and dependencies every time, we put
them into an image and run benchmarks inside containers created from that image.

### Docker Build

A Docker build creates the image from `docker/Dockerfile.base`.

This is expensive because it may need to pull a large Isaac Sim base image and
install dependencies.

### Docker Registry

A Docker registry stores Docker images.

Examples:

* ECR: Amazon Elastic Container Registry
* GHCR: GitHub Container Registry
* NVCR: NVIDIA NGC Container Registry, hosted at `nvcr.io`

If the image is already in a registry, the runner can pull it instead of
building it.

### ECR

ECR is Amazon's Docker registry.

The existing Isaac Lab CI path already has an action named
`.github/actions/ecr-build-push-pull`. That action can use ECR as a shared image
cache.

When ECR works, the workflow can do this:

1. Check whether the image already exists in ECR.
2. If it exists, pull it.
3. If it does not exist, build it and push it to ECR.
4. Future jobs reuse the pushed image.

That is the ideal behavior.

### ECR URL

The ECR URL tells the runner which ECR repository to use.

The current action tries to find it from:

* an explicit `ecr-url` input
* the `ECR_CACHE_URL` environment variable
* an AWS SSM parameter named like `/github-runner/<instance-id>/ecr-cache-url`

On the RTX PRO 6000 runners, this URL is not currently being resolved.

### AWS Credentials

Even if the runner knows the ECR URL, it also needs AWS credentials to pull and
push images.

On AWS EC2-backed runners, this often comes from an IAM role attached to the
machine.

The key question for the NVKS RTX PRO 6000 runners is:

Can these runners get AWS credentials for ECR, and if yes, how are those
credentials provided?

That is why Fatima said this is the important question to ask.

### NVKS

NVKS is the NVIDIA-managed runner environment for the RTX PRO 6000 fleet.

The important point is that these runners do not appear to behave like the
AWS-backed L40S runner setup. So copying the L40S mechanism may not be the right
framing.

### GHA Cache

GHA cache means GitHub Actions cache.

Docker BuildKit can store some build layers in GitHub's cache using
`type=gha`.

This helps, but it is not as good as a real image registry for this use case.

Why:

* it is a build-layer cache, not a finished image cache
* it has repository cache limits
* it can be evicted
* it still may require parts of the image build to run
* parallel matrix jobs can race each other and all miss the cache at the same
  time

So GHA cache is a useful fallback, but not the clean final solution.

### GHCR

GHCR is GitHub Container Registry.

It can store a prebuilt Docker image under the GitHub organization or repository
owner.

Example:

```text
ghcr.io/nvidia-omniverse/isaaclab-perf-smoke:sha-abcdef1
```

The temporary plan is to publish the image once to GHCR, then make every perf
smoke test job pull that image instead of building.

### NVCR / NGC

NVCR is NVIDIA's container registry, hosted at `nvcr.io`.

NGC is the NVIDIA service/account system used to authenticate to `nvcr.io`.

The workflow already uses `NGC_API_KEY` to pull the Isaac Sim base image from
`nvcr.io`.

If GHCR does not work because of package permissions or org policy, NVCR is the
fallback registry option.

## What The Current Problem Is

The current problem is not that Docker caching is impossible.

The problem is that the best existing cache path, ECR, does not appear to be
configured for the NVKS RTX PRO 6000 runners.

The existing action expects one of these to be true:

* the runner has `ECR_CACHE_URL`
* the runner can read an AWS SSM parameter to find the ECR URL
* the runner has AWS credentials that allow ECR pull/push

On the RTX PRO 6000 fleet, that does not seem to be true today.

So the action falls back to a slower local build path. We added GHA build cache
as a fallback, but it still does not give us the same behavior as pulling one
already-built image from a registry.

## Why This Matters

The perf smoke has a benchmark matrix.

For example, it may run:

* Cartpole with PhysX
* G1 with Newton
* Shadow Vision with RTX renderer
* Factory tasks
* other task/backend combinations

Each matrix cell is a separate job. If each job builds the same image, we waste
time and compute.

The benchmark itself may only need a few minutes, but the image build can take
much longer than the benchmark.

That means the smoke test becomes slow for the wrong reason.

We want the runtime to reflect benchmark cost, not repeated container setup.

## What Fatima Is Saying

Fatima's point is:

Do not ask only "can the RTX PRO 6000 runners copy the L40S ECR setup?"

Ask the more basic question:

Can the NVKS RTX PRO 6000 runners get the AWS credentials needed to authenticate
to ECR, and how would those credentials plus the ECR URL be provided to the
runners?

That is the real blocker.

Alexander should be looped in because he owns the current L40S ECR setup and can
explain how that setup works today.

But the NVKS runners may need a different mechanism.

## Current Temporary Fix

We added a temporary prebuilt-image override.

The repo variable is:

```text
PERF_SMOKE_CI_IMAGE
```

If this variable is set, the perf smoke does not build the image in every matrix
job.

Instead, each job does:

1. Log in to the registry if needed.
2. Pull the image from `PERF_SMOKE_CI_IMAGE`.
3. Retag it as the local workflow image name.
4. Run the benchmark exactly as before.

This means all benchmark jobs reuse the same prebuilt image.

The workflow that publishes the image is:

```text
.github/workflows/perf-smoke-publish-image.yaml
```

Default target:

```text
ghcr.io/<owner>/isaaclab-perf-smoke:sha-<short-sha>
```

Fallback target:

```text
nvcr.io/<org-or-team-path>/isaaclab-perf-smoke:<tag>
```

## Option 1: Keep Building In Every Matrix Job

This is the simplest technically.

How it works:

* every benchmark job builds or pulls the image itself
* if ECR is unavailable, it uses local build plus GHA layer cache

Pros:

* no new registry setup
* no separate publish step
* always uses the current commit's source

Cons:

* very slow when cache misses
* can rebuild the same image many times
* matrix jobs can race and all build at once
* wastes runner time
* not good for expensive baseline-seeding runs

This is acceptable only as a fallback.

## Option 2: Use GHA Build Cache

This is what we added as a fallback to the ECR action.

How it works:

* Docker BuildKit stores reusable build layers in GitHub Actions cache
* later builds can reuse those layers

Pros:

* works without ECR
* easy to enable
* useful when the same Dockerfile/deps are built repeatedly
* no external registry needed

Cons:

* not a real finished-image cache
* limited by GitHub cache size and eviction
* does not fully solve parallel matrix races
* still may require build steps to run
* less predictable than a registry

This is useful, but not enough by itself.

## Option 3: Warm-Up Build Job

A warm-up job means one job builds the image first, then the benchmark matrix
runs after it.

How it works:

1. Run one build job.
2. Then run the benchmark matrix.
3. The benchmark jobs try to reuse what the warm-up job produced.

Pros:

* avoids 9 or 10 matrix jobs all building at the same time
* reduces parallel build race waste
* simple idea

Cons:

* without a real registry, the built image is not automatically available on
  other runner machines
* each matrix job may land on a different runner
* Docker images are local to the machine that built them
* uploading/downloading a huge image artifact is usually not practical
* adds a serial step before benchmarks can start

This only becomes strong if the warm-up job pushes the image to a registry.
Without a registry, it is not very useful.

## Option 4: Prebuilt Image In GHCR

This is the temporary fast path we added.

How it works:

1. Run `perf-smoke-publish-image.yaml`.
2. It builds the CI image once.
3. It pushes the image to GHCR.
4. Set `PERF_SMOKE_CI_IMAGE` to that image reference.
5. The perf smoke pulls that image in each matrix job.

Pros:

* avoids repeated cold builds
* easy to test quickly
* uses GitHub's own registry
* good temporary solution while ECR is unresolved
* makes baseline-seeding runs much less wasteful

Cons:

* GHCR permissions may be blocked by org/package policy
* image must be refreshed when dependencies or source change
* not as automatic as ECR
* if the package is private, pull permissions must work from the workflow

This is the best immediate experiment.

## Option 5: Prebuilt Image In NVCR

This is the fallback if GHCR does not work.

How it works:

1. Build the CI image.
2. Push it to `nvcr.io`.
3. Set `PERF_SMOKE_CI_IMAGE` to that `nvcr.io` image.
4. The perf smoke pulls it using `NGC_API_KEY`.

Pros:

* NVIDIA-owned registry
* already familiar because Isaac Sim images come from `nvcr.io`
* uses `NGC_API_KEY`, which the workflow already needs
* may fit NVIDIA infrastructure better than GHCR

Cons:

* needs the right NGC org/team/repository permissions
* may require someone to create the target repository/path
* still a manual prebuilt-image flow unless automated later

This is a good fallback if GHCR package permissions become annoying.

## Option 6: Proper ECR Support For NVKS RTX PRO 6000

This is the clean long-term solution if the platform team supports it.

How it would work:

1. The NVKS RTX PRO 6000 runners get an ECR URL.
2. The runners get AWS credentials that can pull and push to that ECR repo.
3. The existing `ecr-build-push-pull` action automatically uses ECR.
4. The image is built only when needed.
5. Future jobs pull or reuse the cached image.

Pros:

* matches the existing Isaac Lab CI pattern
* most automatic
* supports commit-specific images and dependency-cache images
* avoids manual prebuild steps
* best long-term fit if ECR is supported for NVKS

Cons:

* depends on runner/platform team support
* requires AWS credential delivery to non-EC2 or NVKS runners
* requires a clear answer on how `ECR_CACHE_URL` is provided
* may not be the recommended registry for this runner fleet

This is what we should ask `#nv-gha-runners` and Alexander about.

## Recommended Path

Short term:

1. Use `perf-smoke-publish-image.yaml` to publish a prebuilt image to GHCR.
2. Set `PERF_SMOKE_CI_IMAGE` to that GHCR image.
3. Run the perf smoke and confirm matrix jobs pull instead of build.
4. If GHCR does not work, publish to NVCR and set `PERF_SMOKE_CI_IMAGE` to the
   `nvcr.io` image.

Medium term:

1. Ask `#nv-gha-runners` whether NVKS RTX PRO 6000 runners can get AWS
   credentials for ECR.
2. Ask how the ECR URL should be provided to the runner.
3. Loop in Alexander to explain the existing L40S ECR setup.

Long term:

Use the registry mechanism that the runner owners recommend:

* ECR, if NVKS supports AWS credentials and ECR URL injection
* NVCR, if NVIDIA wants these runners to use NVIDIA registry instead
* GHCR, if GitHub registry permissions are acceptable and simple

## The Key Question To Ask

The clean question is:

```text
For the NVKS RTX PRO 6000 GitHub Actions runners, what is the recommended way to
provide a shared Docker image/cache registry for project-built CI images?

Specifically, can these runners receive AWS credentials and an ECR repository URL
so the existing ecr-build-push-pull action can authenticate to ECR? If yes, how
should those be provided to the runner? If not, should we use GHCR or NVCR
instead for prebuilt CI images?
```

## Simple Mental Model

Think of the Docker image like a big prepared lunchbox.

Bad path:

* every benchmark job cooks the same lunch from scratch

Better path:

* cook once
* put it in a shared fridge
* every benchmark job grabs the same lunchbox

The shared fridge can be:

* ECR
* GHCR
* NVCR

Right now, ECR is the fridge used by some existing CI paths, but the RTX PRO
6000 runners do not appear to have the key or address for that fridge.

So we added a temporary GHCR/NVCR lunchbox path while we ask the runner owners
what the proper shared fridge should be.
