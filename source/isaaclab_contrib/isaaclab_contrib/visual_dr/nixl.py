# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional NIXL entry point, not a completed transport or server.

Communication sketch (both peers retain their CUDA allocations):
  register_memory(tensor) -> exchange agent metadata and serialized descriptors
  get_xfer_descs(local_tensor), deserialize_descs(remote_descriptors)
  initialize_xfer("WRITE", local, remote, peer) -> transfer(handle)
  check_xfer_state(handle) == "DONE" -> receiver may consume; repeat for result

Control carries IDs, layouts, prompts and descriptors only. Before production,
add CUDA producer/consumer synchronization, bounded buffer leases, deadlines,
error recovery and deregistration after all readers finish. UCX can host-stage:
qualify the actual transport/topology and message sizes before enabling it.
"""

import sys


def create_nixl_agent(name: str, listen_port: int = 0):
    """Create a UCX agent lazily; callers own agent and registered-buffer lifetime."""
    if sys.platform != "linux":
        raise RuntimeError("The NIXL sketch targets Linux CUDA workers; native Windows is unsupported")
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as error:
        raise ImportError("Install isaaclab_contrib[nixl] in the worker environment") from error
    return nixl_agent(name, nixl_agent_config(True, True, listen_port, backends=["UCX"]))
