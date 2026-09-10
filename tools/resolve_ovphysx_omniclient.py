# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print the OmniClient requirement embedded in the installed OvPhysX wheel."""

from isaaclab_ov._runtime import _installed_ovphysx_omniverseclient_version


def main() -> None:
    """Print the exact pip requirement for OvPhysX's matched OmniClient release."""
    version = _installed_ovphysx_omniverseclient_version()
    if version is None:
        raise RuntimeError("The installed OvPhysX wheel does not provide an OmniClient compatibility marker.")
    print(f"omniverseclient=={version}")


if __name__ == "__main__":
    main()
