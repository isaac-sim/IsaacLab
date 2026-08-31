.. _project-structure:


Project Structure
=================

Generated Isaac Lab projects have four layers: the **Project**, **Extension**, **Module**, and **Task**.

.. figure:: ../../_static/setup/walkthrough_project_setup.svg
    :align: center
    :figwidth: 100%
    :alt: The structure of the isaac lab template project.

The **Project** is the generated Git repository. Its root contains the uv workspace configuration, ``README.md``, project
utilities under ``scripts``, and Python packages under ``source``. Training, playback, dummy-agent, distributed, and
benchmark workflows use ``uv run isaaclab``.

The **Extension** is an installable package under ``source``. Its ``pyproject.toml`` defines Python packaging metadata and
the Isaac Lab task-discovery entry point. Its ``config/extension.toml`` provides metadata for the Isaac Sim Extension
Manager. A project can contain multiple extensions.

The **Module** contains the Python implementation. By default, it has the same name as the project. The uv workspace
installs it in editable mode, and its ``isaaclab.tasks`` entry point lets the Isaac Lab CLI import and register its tasks.

The **Task** defines an environment and its agent configurations. Shared logic lives in the task-family directory, while
robot-specific environment and agent configurations live under ``config/<robot>``.

For the template, ``gym.register`` is called from the generated Cartpole configuration package:
``isaac_lab_tutorial/source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/isaac_lab_tutorial_direct/config/cartpole/__init__.py``.
The extension's ``isaaclab.tasks`` entry point imports that package recursively.
