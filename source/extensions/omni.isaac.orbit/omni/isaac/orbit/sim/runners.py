# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module with runners to simplify running main and unittests."""

from __future__ import annotations

import traceback
import unittest

import carb
from omni.isaac.kit import SimulationApp

__all__ = ["run_tests", "run_main"]


def run_tests(simulation_app: SimulationApp, verbosity: int = 2, try_except: bool = True):
    """Wrapper for running tests via ``unittest`` with the provided simulation app.

    Args:
        simulation_app: An instance of the :class:`omni.isaac.kit.SimulationApp`.
        verbosity: Verbosity level for the test runner. Defaults to 2.
        try_except: Whether to wrap ``unittest.main()`` in a try-except block. Defaults to True.
            This is useful to remove the try-except block, as it causes issues with VSCode's debugger.
            When False, there is a bit more console spam, but the debugger works as expected.
    """
    _run_function(
        fn=unittest.main, kwargs={"verbosity": verbosity}, simulation_app=simulation_app, try_except=try_except
    )


def run_main(main_fn: callable[[], None], simulation_app: SimulationApp, try_except: bool = False):
    """Wrapper for running ``main`` with the provided simulation app.

    Args:
        main_fn: Main function to run.
        simulation_app: An instance of the :class:`omni.isaac.kit.SimulationApp`.
        try_except: Whether to wrap ``main()`` in a try-except block. Defaults to False.
            This is useful to remove the try-except block, as it causes issues with VSCode's debugger.
            When False, there is a bit more console spam, but the debugger works as expected.

    """
    _run_function(fn=main_fn, kwargs={}, simulation_app=simulation_app, try_except=try_except)


"""
Private methods.
"""


def _run_function(fn: callable, kwargs: dict, simulation_app: SimulationApp, try_except: bool = False):
    """Wrapper for running a function with the provided simulation app.

    Args:
        fn: Function to run.
        kwargs: Keyword arguments to pass to the function.
        simulation_app: An instance of the :class:`omni.isaac.kit.SimulationApp`.
        try_except: Whether to wrap ``fn()`` in a try-except block. Defaults to False.
            This is useful to remove the try-except block, as it causes issues with VSCode's debugger.
            When False, there is a bit more console spam, but the debugger works as expected.
    """
    if try_except:
        try:
            fn(**kwargs)
        except Exception as err:
            # Log out exceptions before re-raising them
            carb.log_error(err)
            carb.log_error(traceback.format_exc())
            raise
        finally:
            # Close sim app
            simulation_app.close()
    else:
        fn(**kwargs)
        simulation_app.close()
