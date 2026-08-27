.. _project-structure:


Project Structure
=================

There are four nested structures you need to be aware of when working in the direct workflow with an Isaac Lab template
project: the **Project**, the **Extension**, the **Modules**, and the **Task**.

.. figure:: ../../_static/setup/walkthrough_project_setup.svg
    :align: center
    :figwidth: 100%
    :alt: The structure of the isaac lab template project.

The **Project** is the root directory of the generated template. It contains the ``source`` and ``scripts`` directories,
a ``README.md`` file, and the uv workspace configuration. When we created the template, we named the project
*IsaacLabTutorial* and this defined the root directory of a git repository. The ``scripts`` directory contains small project
utilities, while training, playback, dummy-agent, distributed, and benchmark workflows use ``uv run isaaclab``. The
``source`` directory contains the Python packages for the project.

The **Extension** is the name of the python package we installed via pip. By default, the template generates a project
with a single extension of the same name. A project can have multiple extensions, and so they are kept in a common ``source``
directory. Python packaging metadata and Isaac Lab task-discovery entry points live in the extension's ``pyproject.toml``.
Packages that also load as Isaac Sim extensions include a ``config`` directory and an ``extension.toml`` containing the
metadata used by the Isaac Sim Extension Manager.

The **Modules** are what actually gets loaded by Isaac Lab to run training (the meat of the code). By default, the template
generates an extension with a single module that is named the same as the project. The structure of the various sub-modules
in the extension is what determines the ``entry_point`` for an environment in Isaac Lab. The uv workspace installs the
generated package in editable mode, and its ``isaaclab.tasks`` package entry point lets the installed Isaac Lab CLI import and
register those tasks automatically.

Finally, the **Task** is the heart of the workflow. Each task family has its own directory. Shared task logic belongs
in that directory, while robot-specific environment and agent configurations live under ``config/<robot>``. This
matches the layout used by ``isaaclab_tasks`` and makes another robot configuration a sibling package instead of a
collection of files at the module root.

For the template, ``gym.register`` is called from the generated Cartpole configuration package:
``isaac_lab_tutorial/source/isaac_lab_tutorial/isaac_lab_tutorial/tasks/isaac_lab_tutorial_direct/config/cartpole/__init__.py``.
That package is imported recursively from the extension's ``isaaclab.tasks`` entry point.
