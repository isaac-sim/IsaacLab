# syntax=docker/dockerfile:1

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Kit-less Isaac Lab with PhysicsNeMo Robotics and this extension, for recording, training, and
# evaluating learned dynamics on clusters. Build the kit-less image first, then pass the PhysicsNeMo
# Robotics checkout as the ``physicsnemo`` build context:
#
#   docker build -f docker/Dockerfile.kitless -t isaac-lab-kitless .
#   docker build -f source/isaaclab_physicsnemo/docker/Dockerfile \
#     --build-context physicsnemo=/path/to/physicsnemo-robotics -t isaaclab-physicsnemo .
ARG BASE_IMAGE=isaac-lab-kitless
FROM ${BASE_IMAGE}

USER root
COPY --from=physicsnemo --chown=isaaclab:isaaclab src/ /opt/physicsnemo-robotics/src/
COPY --from=physicsnemo --chown=isaaclab:isaaclab examples/ /opt/physicsnemo-robotics/examples/
COPY --from=physicsnemo --chown=isaaclab:isaaclab pyproject.toml README.md LICENSE.txt /opt/physicsnemo-robotics/
COPY --chown=isaaclab:isaaclab source/isaaclab_physicsnemo/ ${ISAACLAB_PATH}/source/isaaclab_physicsnemo/

# PhysicsNeMo Robotics pins older Newton, Warp, and Torch releases than Isaac Lab, so it is installed
# without its pins next to the locked Isaac Lab environment, followed by its remaining dependencies.
RUN export UV_NO_CACHE=1 \
 && uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps -e /opt/physicsnemo-robotics \
 && uv pip install --python "${VIRTUAL_ENV}/bin/python" \
      "nvidia-physicsnemo==2.1.1" "jaxtyping>=0.3.3" "tensordict>=0.11,<0.12" "zarr>=3,<4" \
 && uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps -e "${ISAACLAB_PATH}/source/isaaclab_physicsnemo" \
 && chown -R isaaclab:isaaclab "${VIRTUAL_ENV}" "${HOME}"

# Import as the runtime user, so that caches such as Warp's kernel cache belong to it.
USER isaaclab
WORKDIR ${ISAACLAB_PATH}
RUN python -c "import isaaclab_physicsnemo, physicsnemo_robotics, torch, newton; \
      print('torch', torch.__version__, 'newton', newton.__version__)"
