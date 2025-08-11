# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import env_benchmark_test_utils as utils
import pytest

# Global variable for storing KPI data
GLOBAL_KPI_STORE = {}


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


# Fixture for storing KPI data in a global variable
@pytest.fixture(scope="session")
def kpi_store():
    return GLOBAL_KPI_STORE  # Using global variable for storing KPI data


# This hook dynamically generates test cases based on the --workflows option.
# For any test that includes a 'workflow' fixture, this will parametrize it
# with all values passed via the command line option --workflows.
def pytest_generate_tests(metafunc):
    if "workflow" in metafunc.fixturenames:
        workflows = metafunc.config.getoption("workflows")
        metafunc.parametrize("workflow", workflows)


# The pytest session finish hook
def pytest_sessionfinish(session, exitstatus):
    # Access global variable instead of fixture
    tag = session.config.getoption("--tag")
    utils.process_kpi_data(GLOBAL_KPI_STORE, tag=tag)
    print(json.dumps(GLOBAL_KPI_STORE, indent=2))
    save_kpi_payload = session.config.getoption("--save_kpi_payload")
    if save_kpi_payload:
        print("Saving KPI data...")
        utils.output_payloads(GLOBAL_KPI_STORE)
