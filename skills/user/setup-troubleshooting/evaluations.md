# Setup Troubleshooting Evaluations

## Scenario 1: Fresh Install

Query: "Help me install Isaac Lab from source on my machine."

Expected behavior:

- Asks for OS, Python environment, Isaac Sim source, GPU/driver context, and desired backend.
- Points to the official automatic uv installation guide unless the user has a reason to use another supported path.
- Uses documented uv commands for verification.

Known failure modes:

- Copies an installation recipe into the skill response without checking the current docs.
- Mixes pip, source, and binary installation steps.

## Scenario 2: Import Failure

Query: "Isaac Lab installed, but imports fail when I run my script."

Expected behavior:

- Checks whether the user is running from the intended Isaac Lab checkout or correct environment.
- Points to troubleshooting docs for the observed error.
- Requests the smallest relevant traceback if the failure is ambiguous.

Known failure modes:

- Suggests reinstalling before checking the active environment.
- Diagnoses from a partial error message without asking for the missing context.

## Scenario 3: Backend Setup

Query: "I want to run this task with Newton but setup fails."

Expected behavior:

- Routes setup to the unified installation guide and backend choice questions to the physics-backends concept.
- Separates backend installation issues from task implementation issues.
- Verifies setup with a minimal command before running training.

Known failure modes:

- Treats backend setup as a task bug.
- Gives backend-specific commands without checking the docs.

## Scenario 4: Minimal Reproduction

Query: "Training fails with an import error after install. What should I run first?"

Expected behavior:

- Asks for the exact command and traceback.
- Verifies the active install path and Python environment.
- Uses the minimal import command from `reference.md`.
- Escalates to random-agent or training checks only after imports work.

Known failure modes:

- Recommends reinstalling before checking the active Python environment.
- Starts debugging the training runner before validating imports.
- Ignores the official troubleshooting page.

## Scenario 5: Slow Simulation Throughput

Query: "My GPU RL training runs, but throughput is poor and PhysX reports that a convex mesh will fall back to CPU. What should I check?"

Expected behavior:

- Routes first to the performance section in the troubleshooting reference.
- Recommends profiling a representative workload with the Nsight Systems guide before changing settings.
- Explains that the collider warning indicates a CPU fallback and uses the documented collider workarounds.
- Defers general PhysX and Isaac Sim tuning to the maintained upstream performance guides.

Known failure modes:

- Gives generic performance suggestions without consulting the troubleshooting reference.
- Changes solver or RL settings before identifying the bottleneck.
- Omits the CPU-fallback warning or recommends a dynamic triangle-mesh collider.
