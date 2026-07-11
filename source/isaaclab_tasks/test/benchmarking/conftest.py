# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import os
from datetime import datetime

import pytest

# Local imports should be imported last
import env_benchmark_test_utils as utils  # isort: skip

# Global variable for storing KPI data
GLOBAL_KPI_STORE = {}
# Global variable for storing the start timestamp
START_TIMESTAMP = None


def _parse_sim_backend(value: str) -> str:
    """Normalize ``--sim-backend`` (accepts e.g. physx, physX, PHYSX)."""
    v = (value or "").strip().lower()
    if v not in ("physx", "newton"):
        raise argparse.ArgumentTypeError(
            f"Invalid --sim-backend {value!r}: expected 'physx' or 'newton' (case-insensitive)."
        )
    return v


def pytest_addoption(parser):
    parser.addoption(
        "--workflows",
        action="store",
        nargs="+",
        default=["rl_games", "rsl_rl", "sb3", "skrl"],
        help="List of workflows. Must be equal to or a subset of the default list.",
    )
    parser.addoption(
        "--config_path",
        action="store",
        default="configs.yaml",
        help="Path to config file for environment training and evaluation.",
    )
    parser.addoption(
        "--mode",
        action="store",
        default="fast",
        help="Coverage mode defined in the config file.",
    )
    parser.addoption("--num_gpus", action="store", type=int, default=1, help="Number of GPUs for distributed training.")
    parser.addoption(
        "--save_kpi_payload",
        action="store_true",
        help="To collect output metrics into a KPI payload that can be uploaded to a dashboard.",
    )
    parser.addoption(
        "--tag",
        action="store",
        default="",
        help="Optional tag to add to the KPI payload for filtering on the Grafana dashboard.",
    )
    parser.addoption(
        "--sim-backend",
        action="store",
        nargs="+",
        default=["physx"],
        type=_parse_sim_backend,
        help=(
            "One or more physics backends to parametrize over (case-insensitive). "
            "Pass multiple to run each env under each backend: --sim-backend physx newton. "
            "'physx': ``--menagerie-physics-variant physx`` (default env presets). "
            "'newton': ``--menagerie-physics-variant mujoco`` and ``presets=newton_mjwarp`` for Hydra. "
        ),
    )
    parser.addoption("--video", action="store_true", default=False, help="Record videos during training.")
    parser.addoption("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
    parser.addoption("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")


@pytest.fixture
def workflows(request):
    return request.config.getoption("--workflows")


@pytest.fixture
def config_path(request):
    return request.config.getoption("--config_path")


@pytest.fixture
def mode(request):
    return request.config.getoption("--mode")


@pytest.fixture
def num_gpus(request):
    return request.config.getoption("--num_gpus")


@pytest.fixture
def save_kpi_payload(request):
    return request.config.getoption("--save_kpi_payload")


@pytest.fixture
def tag(request):
    return request.config.getoption("--tag")


@pytest.fixture
def video(request):
    return request.config.getoption("--video")


@pytest.fixture
def video_length(request):
    return request.config.getoption("--video_length")


@pytest.fixture
def video_interval(request):
    return request.config.getoption("--video_interval")


# Fixture for storing KPI data in a global variable
@pytest.fixture(scope="session")
def kpi_store():
    return GLOBAL_KPI_STORE  # Using global variable for storing KPI data


# Shard parametrized test items across parallel CI jobs.
# Reads the same TEST_SHARD_INDEX / TEST_SHARD_COUNT env vars used by tools/conftest.py
# for file-level sharding, but applies them at the test-item level so a single
# parametrized file can be split across multiple runners.
# This is a pytest hook — pytest calls it automatically during test collection.
def pytest_collection_modifyitems(config, items):
    shard_index = os.environ.get("TEST_SHARD_INDEX", "")
    shard_count = os.environ.get("TEST_SHARD_COUNT", "")
    if shard_index and shard_count:
        shard_index = int(shard_index)
        shard_count = int(shard_count)
        items[:] = [item for i, item in enumerate(items) if i % shard_count == shard_index]
        print(f"Shard {shard_index}/{shard_count}: selected {len(items)} test items")


# This hook dynamically generates test cases based on the --workflows option.
# For any test that includes a 'workflow' fixture, this will parametrize it
# with all values passed via the command line option --workflows.
def pytest_generate_tests(metafunc):
    if "workflow" in metafunc.fixturenames:
        workflows = metafunc.config.getoption("workflows")
        metafunc.parametrize("workflow", workflows)
    if "sim_backend" in metafunc.fixturenames:
        backends = metafunc.config.getoption("sim_backend")
        metafunc.parametrize("sim_backend", backends)


# The pytest session start hook to capture the start timestamp
def pytest_sessionstart(session):
    global START_TIMESTAMP
    START_TIMESTAMP = datetime.now().isoformat()


# The pytest session finish hook
def pytest_sessionfinish(session, exitstatus):
    # Access global variable instead of fixture
    tag = session.config.getoption("--tag")
    utils.process_kpi_data(GLOBAL_KPI_STORE, tag=tag, timestamp=START_TIMESTAMP)
    print(json.dumps(GLOBAL_KPI_STORE, indent=2))
    save_kpi_payload = session.config.getoption("--save_kpi_payload")
    if save_kpi_payload:
        print("Saving KPI data...")
        output_path = f"logs/{tag}/kpi.json" if tag else None
        utils.output_payloads(GLOBAL_KPI_STORE, output_path=output_path)
