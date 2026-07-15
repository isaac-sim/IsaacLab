# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import signal

import pytest

from isaaclab.app import AppLauncher


def test_sigterm_handler_terminates_process():
    """SIGTERM must terminate Python instead of resuming after the handler."""
    with pytest.raises(SystemExit) as exc_info:
        AppLauncher._terminate_signal_handle_callback(signal.SIGTERM, None)

    assert exc_info.value.code == 128 + signal.SIGTERM
