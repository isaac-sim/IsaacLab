.. _template-generator:


Create new project or task
==========================

The template generator creates external Isaac Lab projects and internal tasks.

The template generator enables you to create an:

* **External project** (recommended): An independent repository that can also load as an Isaac Sim extension.

  .. hint::

    The generator initializes a Git repository in the selected directory.

* **Internal task**: A task added directly to the Isaac Lab repository for contribution upstream.

  .. warning::

    Pip installations of Isaac Lab do not support *Internal* templates.
    If ``isaaclab`` is loaded from ``site-packages`` or ``dist-packages``, the *Internal* option is disabled
    and the *External* template will be used instead.

Running the template generator
------------------------------

Install Isaac Lab by following the `installation guide <../../setup/installation/index.html>`_.
Then run the generator from the uv-managed environment:

.. code-block:: bash

  uv run isaaclab --new  # or "uv run isaaclab -n"

The generator will guide you in setting up the project/task for your needs by asking you the following questions:

* Type of project/task (external or internal), and project/task path or names according to the selected type.
* Isaac Lab workflows (see :ref:`feature-workflows`).
* Reinforcement learning libraries (see :ref:`rl-frameworks`), and algorithms (if the selected libraries support multiple algorithms).

External project usage (once generated)
---------------------------------------

Once the external project is generated, a ``README.md`` file will be created in the specified directory.
This file will contain instructions on how to install the project and run the tasks.

The generated project is a uv workspace. From its root, create the environment and install the
generated package in editable mode:

.. code-block:: bash

  uv sync

* List the tasks and physics presets available in the project.

  .. code-block:: bash

    uv run python scripts/list_envs.py --show_presets

* Train and play a task with the installed Isaac Lab commands.

  .. code-block:: bash

    uv run isaaclab train --rl_library <library> --task <Task-Name>
    uv run isaaclab play --rl_library <library> --task <Task-Name> --checkpoint latest

  The same command surface provides ``zero_agent``, ``random_agent``, ``benchmark``, and
  ``train_multigpu``. Generated packages advertise their task modules through package metadata,
  so the commands discover downstream tasks without project-specific runner scripts.

For more details, please follow the instructions in the generated project's ``README.md`` file.

Internal task usage (once generated)
---------------------------------------

Once the internal task is generated, it will be available along with the rest of the Isaac Lab tasks.

* List the tasks available in Isaac Lab.

  .. code-block:: bash

    uv run python scripts/environments/list_envs.py --show_presets

* Run a task.

  .. code-block:: bash

    uv run isaaclab train --rl_library <library> --task <Task-Name>

* Run a task with dummy agents.

  Dummy agents help verify the environment before training.

  * Zero-action agent

    .. code-block:: bash

      uv run isaaclab zero_agent --task <Task-Name>

  * Random-action agent

    .. code-block:: bash

      uv run isaaclab random_agent --task <Task-Name>
