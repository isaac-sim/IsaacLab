# Runtime visual DR: what we need

Working draft for discussion. We want to vary the appearance of simulated camera
images during reinforcement-learning (RL) post-training, while keeping the robot
and task information reliable. The policy should see the changed background as
part of its normal environment observations.

This document describes the behavior we need. It stays independent of a specific
task, training framework, model or serving product. Implementation choices belong
in the separate sketch.

## What we need for the first version

### Enough scene information to preserve the foreground

For each camera we want to randomize, we need RGB and a segmentation image that
identifies the objects we want to keep. Depth-guided generation also needs depth.
These signals need to come from the same camera frame and episode, with known
resolution, layout, numeric format and depth units.

Scene assets need semantic tags or equivalent labels. We should be able to say
"preserve the robot, manipulated objects and this supporting surface" without
hard-coding the numeric IDs assigned by a renderer. Each camera needs a valid
mapping from its segmentation output to those labels.

Missing labels or stale masks cannot mean "regenerate everything." We need an
explicit choice between stopping with a useful error and returning the current
original image with a recorded reason. Camera warmup and retries need a limit.

### Images that remain usable by the policy

Background-only DR needs to preserve the selected foreground pixels, including
thin fingers and partly hidden objects. We need to agree on how much of the
boundary around those objects is protected.

The output keeps the expected image shape, numeric range, device, camera order
and timestamp. Resizing and normalization cannot misalign the foreground mask.
The simulation state, actions, rewards and non-image observations stay the same.

DR happens before policy image normalization and history assembly, or provides
the same result under those operations. Reading one observation twice should
return the same randomized image. Initialization checks and debug reads should
not trigger generation or advance the random seed. Raw sensor images remain
available without being overwritten.

Turning DR off should restore normal environment behavior without loading a
model or serving stack. Turning it on should work from an ordinary environment
loop, without depending on a particular policy or trainer.

### Generate the frames we actually use

An action chunk is a sequence of K environment actions between policy decisions.
For example, with K=16, we normally need the initial image and the image after
those 16 actions. Generating the other 15 images is unnecessary if nothing uses
them. We still execute every action, physics step, reward and termination check.
Skipping rendering is a separate decision because other consumers may need it.

The schedule needs to handle single actions, variable chunk lengths and a short
last chunk. If a policy uses image history, or another consumer needs intermediate
frames, those frames still need the appropriate processing.

Reset needs special care. The last image of an episode and the first image after
reset are different observations, even if they occur in the same environment
step. Resetting some environments cannot invalidate or change the others' images.

A value estimator may need images to estimate future return at the end of a
rollout. Those images cannot be skipped just because the policy will not act on
them. During learning, we reuse the processed observations or saved policy inputs
from collection; we do not generate a different background for the same sample.

### Keep the online image path on the GPU

RGB, depth, segmentation, masks and generated images stay in GPU memory from
camera extraction through DR and back into the observation. This includes
preprocessing and both directions of any serving transfer. GPU-to-GPU copies
are fine; staging image payloads through CPU/system memory is not.

CPU messages can carry small control information such as request IDs, shapes,
labels, prompts and buffer references. They cannot carry image bytes. Likewise,
encoding images into files, NumPy arrays or base64 for serving does not meet this
requirement.

We need to verify the actual transfer path on the supported hardware. GPU buffers
at both endpoints are not enough evidence. If the available path requires host
staging, report that the configuration is unsupported instead of silently using
it. Transfers between machines need the same guarantee in both directions.

Buffer lifetime matters too: finish writing before transfer, finish transfer
before inference, and finish reading before reuse. A timeout or worker restart
cannot make a buffer safe to overwrite while GPU work is still using it. Queues,
in-flight images and batch sizes all need bounds.

Optional debug exports can copy selected images to the CPU outside the serving
path. Model weights can also move to CPU memory for offloading. Neither is a
reason to stage live image payloads there.

### Fit alongside simulation and learning

The simulator, policy, DR model and learner share a memory budget. We need to
measure their combined usage, including model startup, warmup, temporary tensors
and transfer buffers.

When models share a GPU across collection and learning, offloading needs an
explicit handoff: stop accepting work, finish inference and transfers, release
the agreed memory, then let learning start. Activate the DR model before the next
consumed environment observation is prepared. A late request cannot bring it
back onto the GPU while the learner owns that memory.

Resetting an environment should not reload the model. Closing one client should
not shut down a model that other clients still use. The application controls
these phases through a common interface, without the runtime reading a trainer's
configuration files or depending on its worker classes.

### Make runs understandable and repeatable

We need explicit settings for how often DR applies, whether that decision is
per-environment or per-batch, and when a style changes. Record seeds, model and
configuration versions, and why an image was randomized or left unchanged.
Changing debug output or assigning work to another replica should not change the
sampling decision. Any limits on numerical reproducibility should be documented.

Clean evaluation and fixed-seed DR evaluation should be selectable independently.
An image should not accidentally get randomized twice by different components.

Before evaluating RL results, the application needs a working clean baseline:
a compatible starting policy, observation/action mappings, rewards, episode-ending
conditions and reset behavior. A convincing image demo alone does not show that
post-training works.

## Good to have, depending on the deployment

- **Replicas and batching:** run multiple model instances or combine requests to
  improve throughput, while keeping queues bounded and sample identities stable.
- **FP8:** reduce model memory or improve speed where supported. Report which
  parts use reduced precision and measure the effect on image quality.
- **Compilation:** reduce repeated inference cost. Include warmup, changing batch
  sizes and prompts, and repeated offload/onload when measuring the benefit.
- **Consistent styles:** keep an episode's background style stable and make views
  agree where possible. Generating coherent video is a separate capability from
  generating independent images.
- **Dedicated GPUs and multiple nodes:** scale when needed. Every supported mode
  still needs direct GPU image transfer, explicit device placement, globally unique
  worker/environment identities and isolation between simultaneous runs.
- **Visual diagnostics:** compare raw and processed observations with timestamps,
  masks and generation status, using a small, optional export budget.

These can become necessary for a particular memory or throughput target. Keeping
images on the GPU and preserving RL correctness are part of the baseline either
way. The generative model and serving implementation should be replaceable without
changing what the environment's observations mean.

## How we will check that it works

Start with a small run and inspect the actual images consumed by the policy.
Check foreground preservation, synchronized camera signals, repeated reads,
action chunks, partial resets and terminal images. Confirm that learning reuses
the collected inputs without making new DR calls.

Then trace the GPU path in both directions, including small and partial batches.
Separate model paging and optional debug downloads from live image traffic. Run
several collection/learning cycles and exercise queue saturation, cancellation
and worker restart to check memory ownership and recovery.

For each supported configuration, record end-to-end observation latency,
queue/transfer/inference time, generated frames per consumed observation,
throughput, combined peak memory and errors or skipped samples. Compare clean
and DR task results. Record the tested OS, GPUs, interconnect and software versions
so others know which configurations the results cover.

## Questions to settle together

- Which foreground classes and boundaries need exact preservation?
- Should styles stay fixed within an episode, and how often should DR apply?
- Does the policy or value estimator consume intermediate or historical images?
- What latency, throughput and shared-GPU memory budget are we targeting?
- Which deployment modes need to work first: in-process, separate processes,
  separate GPUs or multiple machines?
- When a valid image cannot be generated, should the run stop or use the current
  raw GPU image with a recorded reason?

Keep edits concrete: describe the behavior we want, why it matters and what is
still open. As we settle a question, fold the decision into the relevant section.
