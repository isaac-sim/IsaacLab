# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move trocar's single-frame Cosmos replicas at sequential RLinf rollout/update boundaries."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager

from isaaclab_tasks.utils import parse_env_cfg

from ..client.cosmos_model import CosmosModel, service_endpoints
from ..client.cosmos_model_cfg import CosmosModelCfg


@contextmanager
def cosmos_rollout_residency(runner, overrides):
    """Keep the services resident for collection and release their GPU memory for PPO."""
    task = parse_env_cfg(runner.cfg.env.train.init_params.id, overrides=overrides)
    if (
        not task.cosmos_offload_during_update
        or task.cosmos.generation_mode != "single_frame"
        or runner.cfg.runner.get("use_training_pipeline", False)
        or runner.cfg.runner.get("overlap_env_bootstrap", False)
    ):
        yield
        return

    with ExitStack() as stack:
        replicas = []
        for endpoint in service_endpoints(task.cosmos.endpoint):
            client = CosmosModel(
                CosmosModelCfg(generation_mode="single_frame", endpoint=endpoint, timeout=task.cosmos.timeout)
            )
            replicas.append(client)
            stack.callback(client.close)
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=len(replicas)))
        collect = runner.env.interact
        train = runner.actor.run_training
        last_collection = None

        def interact(*args, **kwargs):
            nonlocal last_collection
            list(pool.map(CosmosModel.activate, replicas))
            last_collection = collect(*args, **kwargs)
            return last_collection

        def update(*args, **kwargs):
            if last_collection is not None:
                last_collection.wait()
            list(pool.map(CosmosModel.offload, replicas))
            return train(*args, **kwargs)

        runner.env.interact = interact
        runner.actor.run_training = update
        try:
            list(pool.map(CosmosModel.offload, replicas))
            yield
        finally:
            runner.env.interact = collect
            runner.actor.run_training = train
            list(pool.map(CosmosModel.activate, replicas))
