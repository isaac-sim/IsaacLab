# Installing Isaac Lab Examples

## Example 1: Recommended Pip Install On Ubuntu 22.04

User: "Install Isaac Lab on my Ubuntu 22.04 desktop."

Suggested agent response:

1. Confirm the NVIDIA driver is at or above the minimum in `docs/source/setup/installation/index.rst`.
2. Route to `docs/source/setup/installation/pip_installation.rst`.
3. Prescribe the install page's commands in order without paraphrasing.
4. After install, run `uv run python -c "import isaaclab; print('ok')"`.

## Example 2: Binary Install On Ubuntu 20.04

User: "Install Isaac Lab on Ubuntu 20.04. My GLIBC is 2.31."

Suggested agent response:

1. Confirm GLIBC is below the pip Isaac Sim minimum documented in `docs/source/setup/installation/pip_installation.rst`.
2. Route to `docs/source/setup/installation/binaries_installation.rst` instead of the pip page.
3. Confirm the user is prepared to download the Isaac Sim binary manually.
4. Follow the binary install page's steps and use the docs-defined verification once done.

## Example 3: Windows 11 Install

User: "Install Isaac Lab on Windows 11."

Suggested agent response:

1. Confirm the Windows tab in `docs/source/setup/installation/index.rst` and the checkout's chosen install page.
2. Confirm NVIDIA Windows driver, Python 3.12, and any Windows extras named in the page.
3. Enable Windows long-path support per the install page before starting the install.
4. Prescribe the Windows tab commands from the install page verbatim.
5. Run the docs-defined verification command.

## Example 4: Docker Deployment

User: "Set up Isaac Lab in Docker."

Suggested agent response:

1. Confirm Docker Engine, Docker Compose, and NVIDIA Container Toolkit are installed before running any install command.
2. Route to `docs/source/deployment/docker.rst`.
3. Use the container launch command from that page.
4. Run the verification command inside the container as documented.

## Example 5: Isaac Sim Contributor Building From Source

User: "I want to modify Isaac Sim itself. Set me up with a source build alongside Isaac Lab."

Suggested agent response:

1. Confirm the user needs to modify Isaac Sim source (not just clone Isaac Lab).
2. Route to `docs/source/setup/installation/source_installation.rst`.
3. Confirm disk headroom and the build prerequisites listed on that page. On Windows, confirm Visual Studio Build Tools.
4. Prescribe the source-build commands verbatim, including the Isaac Sim build step before the Isaac Lab install commands.
5. Run the docs-defined verification command for the source path after the build completes.
