# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import env_test_utils as utils
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
    parser.addoption(
        "--disable_duration_check",
        action="store_true",
        help="Disable the duration check for the test.",
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
def disable_duration_check(request):
    return request.config.getoption("--disable_duration_check")


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


# Cache for env configs to avoid repeated file loading
_ENV_CONFIGS_CACHE = {}


def pytest_collection_modifyitems(config, items):
    """Mark tests as skipped before execution based on config/workflow compatibility."""
    import os
    import re
    import yaml

    mode = config.getoption("--mode")
    config_path = config.getoption("--config_path")

    # Load config once
    if config_path not in _ENV_CONFIGS_CACHE:
        if config_path.startswith("/"):
            full_config_path = config_path
        else:
            full_config_path = os.path.join(os.path.dirname(__file__), config_path)
        with open(full_config_path) as f:
            _ENV_CONFIGS_CACHE[config_path] = yaml.safe_load(f)

    env_configs = _ENV_CONFIGS_CACHE[config_path]

    for item in items:
        # Only process test_train_environments tests
        if "test_train_environments" not in item.name:
            continue

        # Extract workflow and task_spec from test parameters
        workflow = item.callspec.params.get("workflow") if hasattr(item, "callspec") else None
        task_spec = item.callspec.params.get("task_spec") if hasattr(item, "callspec") else None

        if workflow is None or task_spec is None:
            continue

        # Check if workflow is supported for this task
        if workflow + "_cfg_entry_point" not in task_spec.kwargs:
            item.add_marker(pytest.mark.skip(reason=f"Workflow {workflow} not supported for task {task_spec.id}"))
            continue

        # Check if config exists for this task
        task = task_spec.id
        extended_task = f"{workflow}:{task}"

        # Check for config match (same logic as get_env_config)
        config_found = False
        if mode in env_configs:
            if extended_task in env_configs[mode] or task in env_configs[mode]:
                config_found = True
            else:
                for env_config_key in env_configs[mode].keys():
                    if re.match(env_config_key, extended_task) or re.match(env_config_key, task):
                        config_found = True
                        break

        if not config_found:
            item.add_marker(pytest.mark.skip(reason=f"No config found for task {task} in {mode} mode"))


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
