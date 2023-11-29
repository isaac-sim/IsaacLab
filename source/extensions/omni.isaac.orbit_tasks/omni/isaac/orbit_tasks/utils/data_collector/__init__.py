# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for data collection utilities.

All post-processed robomimic compatible datasets share the same data structure.
A single dataset is a single HDF5 file. The stored data follows the structure provided
`here <https://robomimic.github.io/docs/datasets/overview.html#dataset-structure>`_.

The collector takes input data in its batched format and stores them as different
demonstrations, each corresponding to a given environment index. The demonstrations are
flushed to disk when the :meth:`RobomimicDataCollector.flush` is called for the
respective environments. All the data is saved when the
:meth:`RobomimicDataCollector.close()` is called.

The following sample shows how to use the :class:`RobomimicDataCollector` to store
random data in a dataset.

.. code-block:: python

   import os
   import torch

   from omni.isaac.orbit_tasks.utils.data_collector import RobomimicDataCollector

   # name of the environment (needed by robomimic)
   task_name = "Isaac-Franka-Lift-v0"
   # specify directory for logging experiments
   test_dir = os.path.dirname(os.path.abspath(__file__))
   log_dir = os.path.join(test_dir, "logs", "demos")
   # name of the file to save data
   filename = "hdf_dataset.hdf5"
   # number of episodes to collect
   num_demos = 10
   # number of environments to simulate
   num_envs = 4

   # create data-collector
   collector_interface = RobomimicDataCollector(task_name, log_dir, filename, num_demos)

   # reset the collector
   collector_interface.reset()

   while not collector_interface.is_stopped():
      # generate random data to store
      # -- obs
      obs = {
            "joint_pos": torch.randn(num_envs, 10),
            "joint_vel": torch.randn(num_envs, 10)
      }
      # -- actions
      actions = torch.randn(num_envs, 10)
      # -- rewards
      rewards = torch.randn(num_envs)
      # -- dones
      dones = torch.rand(num_envs) > 0.5

      # store signals
      # -- obs
      for key, value in obs.items():
            collector_interface.add(f"obs/{key}", value)
      # -- actions
      collector_interface.add("actions", actions)
      # -- next_obs
      for key, value in obs.items():
            collector_interface.add(f"next_obs/{key}", value.cpu().numpy())
      # -- rewards
      collector_interface.add("rewards", rewards)
      # -- dones
      collector_interface.add("dones", dones)

      # flush data from collector for successful environments
      # note: in this case we flush all the time
      reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
      collector_interface.flush(reset_env_ids)

   # close collector
   collector_interface.close()

"""

from .robomimic_data_collector import RobomimicDataCollector
