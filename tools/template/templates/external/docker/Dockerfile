ARG ISAACLAB_BASE_IMAGE_ARG

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH=${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}

# Copy the Isaac Lab Extension Template directory (files to exclude are defined in .dockerignore)
COPY --chown=isaaclab:isaaclab . ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}

# # Install whatever you need as additional dependencies.
RUN uv pip install -e ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}/source/{{ name }}

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR ${DOCKER_ISAACLAB_EXTENSION_TEMPLATE_PATH}
