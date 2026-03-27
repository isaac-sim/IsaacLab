# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import ClassVar

DEFAULT_TIMEOUT: int = 30
"""Default timeout for the task."""

_MAX_RETRIES: int = 3  # private constant (note the underscore)
"""Maximum number of retries for the task."""


def run_task(task_name: str):
    """Run a task by name.

    Args:
        task_name: The name of the task to run.
    """
    print(f"Running task: {task_name}")


class TaskRunner:
    """Runs and manages tasks."""

    DEFAULT_NAME: ClassVar[str] = "runner"
    """Default name for the runner."""

    _registry: ClassVar[dict] = {}
    """Registry of runners."""

    def __init__(self, name: str):
        """Initialize the runner.

        Args:
            name: The name of the runner.
        """
        self.name = name
        self._tasks = []  # private instance variable

    def __del__(self):
        """Clean up the runner."""
        print(f"Cleaning up {self.name}")

    def __repr__(self) -> str:
        return f"TaskRunner(name={self.name!r})"

    def __str__(self) -> str:
        return f"TaskRunner: {self.name}"

    """
    Properties.
    """

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    """
    Operations.
    """

    def initialize(self):
        """Initialize the runner."""
        print("Initializing runner...")

    def update(self, task: str):
        """Update the runner with a new task.

        Args:
            task: The task to add.
        """
        self._tasks.append(task)
        print(f"Added task: {task}")

    def close(self):
        """Close the runner."""
        print("Closing runner...")

    """
    Operations: Registration.
    """

    @classmethod
    def register(cls, name: str, runner: "TaskRunner"):
        """Register a runner.

        Args:
            name: The name of the runner.
            runner: The runner to register.
        """
        if name in cls._registry:
            _log_error(f"Runner {name} already registered. Skipping registration.")
            return
        cls._registry[name] = runner

    @staticmethod
    def validate_task(task: str) -> bool:
        """Validate a task.

        Args:
            task: The task to validate.

        Returns:
            True if the task is valid, False otherwise.
        """
        return bool(task and task.strip())

    """
    Internal operations.
    """

    def _reset(self):
        """Reset the runner."""
        self._tasks.clear()

    @classmethod
    def _get_registry(cls) -> dict:
        """Get the registry."""
        return cls._registry

    @staticmethod
    def _internal_helper():
        """Internal helper."""
        print("Internal helper called.")


"""
Helper operations.
"""


def _log_error(message: str):
    """Internal helper to log errors.

    Args:
        message: The message to log.
    """
    print(f"[ERROR] {message}")


class _TaskHelper:
    """Private utility class for internal task logic."""

    def compute(self) -> int:
        """Compute the result.

        Returns:
            The result of the computation.
        """
        return 42
