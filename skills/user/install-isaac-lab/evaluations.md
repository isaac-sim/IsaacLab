# Installing Isaac Lab Evaluations

## Scenario 1: Fresh Install On Ubuntu 22.04

Query: "Help me install Isaac Lab on my Ubuntu 22.04 machine."

Expected behavior:

- Asks for the target Isaac Lab ref (defaults to the latest release) and confirms Ubuntu 22.04 with an NVIDIA GPU and driver.
- Routes to `docs/source/setup/installation/pip_installation.rst` (the docs-Recommended path) using uv.
- Reads the pip install page's commands verbatim from that ref before prescribing.
- Ends with the docs-defined minimal verification command.

Known failure modes:

- Copies pip install commands or version pins from memory without reading the ref's current install page.
- Prescribes source or binary paths when the user has no reason to depart from the Recommended path.
- Skips the verification step or substitutes a paraphrased verification command.

## Scenario 2: Older Distro With GLIBC Below The Pip Minimum

Query: "I want to install Isaac Lab on Ubuntu 20.04. My GLIBC is 2.31."

Expected behavior:

- Confirms GLIBC before recommending any pip path.
- Routes to `docs/source/setup/installation/binaries_installation.rst` because pip Isaac Sim needs the GLIBC minimum documented in `docs/source/setup/installation/pip_installation.rst`.
- Notes the manual Isaac Sim binary download step and confirms the user is prepared for it.
- Ends with the binary path's verification command from the docs.

Known failure modes:

- Recommends the pip path anyway and lets the user hit a `GLIBC` version error at install time.
- Copies old GLIBC minimums from memory rather than reading the ref's `docs/source/setup/installation/index.rst`.
- Sends the user to the pip page and then improvises a binary workaround if pip fails.

## Scenario 3: Windows 11 Install

Query: "Install Isaac Lab on Windows 11."

Expected behavior:

- Confirms the NVIDIA Windows driver meets the ref's minimum, Python 3.12 is present, and Visual Studio Build Tools are installed if the user picks the source build.
- Follows the Windows tab in the ref's install page for command form and long-path support.
- Uses the Windows-specific commands from the install page rather than paraphrasing shell equivalents.
- Verifies with the docs-defined verification command.

Known failure modes:

- Prescribes bash commands on Windows or paraphrases them into cmd form.
- Skips the long-path support step documented for Windows.
- Recommends the source build without checking for Visual Studio Build Tools.

## Scenario 4: Docker Deployment

Query: "Run Isaac Lab in a Docker container."

Expected behavior:

- Confirms Docker Engine, Docker Compose, and NVIDIA Container Toolkit are installed before prescribing container commands.
- Routes to `docs/source/deployment/docker.rst`.
- Uses the container launch command from that page and does not paraphrase image tags.
- Runs verification inside the container as documented.

Known failure modes:

- Prescribes container commands without checking that NVIDIA Container Toolkit is installed.
- Copies container image tags from memory instead of reading the deployment doc.
- Recommends the Docker path on Windows without confirming Docker Desktop with the WSL2 GPU stack is set up.

## Scenario 5: Isaac Sim Contributor Building From Source

Query: "I want to modify Isaac Sim itself. Set me up with a source build alongside Isaac Lab."

Expected behavior:

- Confirms the user needs Isaac Sim source (not just Isaac Lab source), then routes to `docs/source/setup/installation/source_installation.rst`.
- Confirms disk headroom, the Isaac Sim source-build prerequisites documented on that page, and Visual Studio Build Tools if the user is on Windows.
- Prescribes the source-build commands verbatim in the order documented, including the build step for Isaac Sim.
- Runs the docs-defined verification command for the source path after the build completes.

Known failure modes:

- Confuses "Isaac Lab from source" (git-clone install of Isaac Lab) with "Isaac Sim from source" (build Isaac Sim itself). This skill routes to the source_installation page only when the user genuinely needs to modify Isaac Sim.
- Recommends the pip path when the user explicitly needs to modify Isaac Sim source.
- Skips the Isaac Sim build step and jumps to the Isaac Lab install commands.
