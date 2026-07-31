# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-action agent executable for Isaac Lab environments."""

# PLACEHOLDER: Extension template (do not remove this comment)

from isaaclab_rl.entrypoints import run_zero_agent_cli


def main(argv: list[str] | None = None) -> int:
    """Run an environment with a zero-action agent."""
    return run_zero_agent_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
