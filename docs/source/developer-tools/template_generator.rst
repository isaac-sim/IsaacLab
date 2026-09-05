.. _own-project:
.. _template-generator:

Build your own project or task
==============================

The template generator bootstraps the package structure, task registration, and
agent configurations needed to start developing an Isaac Lab task. Use it to
create either a standalone project outside Isaac Lab or a task intended for
contribution to the Isaac Lab repository. Both options include a working
Cartpole example that you can replace with your own environment.

Choose what to generate
-----------------------

The first prompt chooses where the new task will live.

.. list-table::
   :widths: 22 48 30
   :header-rows: 1

   * - Type
     - Use it when
     - Result
   * - External (recommended)
     - You are creating an application, experiment, or reusable project outside
       the Isaac Lab repository.
     - A standalone, installable uv project using a ``src`` layout.
   * - Internal
     - You intend to contribute the task to the Isaac Lab repository.
     - A task package under ``source/isaaclab_tasks``.

Installed Isaac Lab wheels only offer external projects. The internal option is
available from a source checkout because it writes directly into that checkout.

Next, choose one or more task workflows. See :ref:`feature-workflows` for the
complete comparison.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Workflow
     - Good fit
   * - Manager-based | single-agent
     - Most new tasks. Observations, actions, rewards, events, and terminations
       remain modular and easy to replace.
   * - Direct | single-agent
     - Tasks that need custom step and reset control or a compact environment
       implementation.
   * - Direct | multi-agent
     - Tasks with multiple policies or agent-specific observation and action
       spaces.

Finally, choose the RL libraries and algorithms whose configuration files you
want generated. The prompt adapts the available choices to the selected
workflow. See :ref:`rl-frameworks` for the framework comparison.

Create and run an external project
----------------------------------

First, :ref:`install Isaac Lab <isaaclab-installation-root>`. Run the generator
from the Isaac Lab source checkout or from a uv project that contains the
installed Isaac Lab package:

.. code-block:: bash

   uv run isaaclab --new

The command uses the dependencies from the active Isaac Lab environment. It
does not invoke ``pip`` or install another set of template dependencies, so it
also works in the pip-less virtual environments created by ``uv``.

The short form is equivalent:

.. code-block:: bash

   uv run isaaclab -n

Select the following options for a small first project:

* **External** project
* A parent directory outside the Isaac Lab repository
* A Python-compatible project name, such as ``my_robot_project``
* A task family name, such as ``balance``
* A robot/config name, such as ``cartpole``
* **No** Isaac Sim UI extension for a headless task package
* **Manager-based | single-agent** workflow
* **rsl_rl** with **PPO**

The prompts display the valid options and accept numbered, comma-separated
selections when more than one choice is allowed. The generator creates the
project under ``<parent-directory>/<project-name>`` and initializes a Git
repository there.

Enter the generated project and create its environment:

.. code-block:: bash

   cd <parent-directory>/my_robot_project
   uv sync

This default environment includes the selected RL library and the kit-less
Newton backend. It does **not** install Isaac Sim.

List the generated task name and its available presets:

.. code-block:: bash

   uv run python scripts/list_envs.py --show_presets

Copy the task name from the output, then run a quick smoke test:

.. code-block:: bash

   uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16 --viz newton

If the environment launches and the cart moves, the project is ready to edit.
You can then train and play a policy with the same command surface used by
Isaac Lab itself:

.. code-block:: bash

   uv run isaaclab train --rl_library rsl_rl --task <TASK_NAME>
   uv run isaaclab play --rl_library rsl_rl --task <TASK_NAME> --checkpoint latest --viz newton

Choose a simulation backend
---------------------------

The default ``uv sync`` installs the kit-less Newton backend without Isaac Sim.
Use a generated ``isaacsim``, ``ov``, ``ovphysx``, or ``ovrtx`` extra when a
command needs that optional runtime. For example:

.. code-block:: bash

   uv run --extra isaacsim isaaclab random_agent \
      --task <TASK_NAME> physics=isaacsim_physx

Place ``--extra`` before ``isaaclab`` and keep it on every command that needs
the optional runtime. See :ref:`backends-and-presets` for the backend and preset
model and :ref:`isaac-lab-quickstart` for supported physics, renderer, and
visualizer combinations.

.. _project-structure:

Understand the generated project
--------------------------------

An external project is a single installable Python package with the same
standard uv ``src`` layout used by maintained downstream examples. Its root
``pyproject.toml`` declares the package, development tools, backend extras, and
``isaaclab.tasks`` entry point.

The project name identifies the repository and Python package. Task-wide MDP
terms live under the separately named task family. Robot-specific scenes,
registrations, and agent configurations live under ``config/<robot-name>``.
This separation lets a project add another robot configuration without copying
the task MDP, or add another task family without creating another repository.

Code shared by several task families can live in a package such as
``tasks/mdp``. The generated task importer skips packages named ``mdp`` while it
searches for task registrations, so shared modules do not register as task
families.

A generated project resembles:

.. code-block:: text

   my_robot_project/
   ├── LICENSE
   ├── pyproject.toml
   ├── README.md
   ├── scripts/
   │   └── list_envs.py
   ├── src/
   │   └── my_robot_project/
   │       ├── __init__.py
   │       └── tasks/
   │           ├── __init__.py
   │           └── balance/
   │               ├── mdp/
   │               └── config/
   │                   └── cartpole/
   │                       ├── agents/
   │                       └── env_cfg.py
   └── tests/
       └── test_registration.py

The generated package ``__init__.py`` is intentionally passive. Installing the
project exposes ``my_robot_project.tasks`` through the ``isaaclab.tasks`` entry
point, so importing the package for utilities does not eagerly register tasks.

If you opt into the Isaac Sim UI extension, the generator additionally creates
``config/extension.toml`` and ``src/my_robot_project/ui_extension_example.py``.
Launch Isaac Sim with the generated ``isaacsim`` extra when using it. The
default is a headless task package and does not include these files.

Run commands from the project root so ``uv`` can find the package and task entry
point. Commit ``pyproject.toml`` and ``uv.lock`` to give collaborators the same
dependency resolution.

Develop the generated task
--------------------------

Start with a dummy agent before training. A zero-action agent is useful for
checking resets and passive dynamics, while a random-action agent also exercises
the action and observation paths:

.. code-block:: bash

   uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
   uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

Edit the generated environment configuration and task terms under
``src/<project-name>/tasks``. The generated package is
installed in editable mode, so you do not need to reinstall it after each
change.

Use the remaining project commands as the task matures:

.. code-block:: bash

   uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> \
      --task <TASK_NAME> --num_gpus 2
   uv run isaaclab benchmark runtime --task <TASK_NAME> \
      --num_envs 16 --num_steps 1000
   uv run pre-commit run --all-files

The generator includes ``tests/test_registration.py`` to verify the task IDs,
environment entry points, and default agent. Its ``pyproject.toml`` installs
pytest for development and registers the ``unit``, ``integration``, ``smoke``,
and ``kitless`` markers. Add project-owned behavioral tests under ``tests`` and
run them with:

.. code-block:: bash

   uv run pytest tests

The reusable-looking helpers under ``source/isaaclab_tasks/test`` belong to the
Isaac Lab repository test suite and are not installed with ``isaaclab_tasks``.
External projects should build their environment harness from public APIs and
maintain project-local fixtures. Copying ``env_test_utils.py`` into a project is
vendoring it, so the project must track upstream changes to that copy.

To configure VS Code or Cursor, run the generated setup task or invoke it directly:

.. code-block:: bash

   uv run python .vscode/tools/setup_vscode.py

The command selects the active interpreter and creates a git-ignored
``pyrightconfig.json``. This child configuration inherits the checked-in
Pyright policy from ``pyproject.toml`` and adds the generated project's
``src`` import root, installed Isaac Lab packages, and any discovered Isaac
Sim extensions. When using the ``isaacsim`` extra, include it while generating
the configuration:

.. code-block:: bash

   uv run --extra isaacsim python .vscode/tools/setup_vscode.py

In VS Code, use Pylance and select the interpreter that ran the setup command.
In Cursor, install the ``detachhead.basedpyright`` extension instead of Pylance,
select the same interpreter, and reload the window. Both language servers read
the generated ``pyrightconfig.json``.

Create an internal task
-----------------------

Choose **Internal** only when working from an Isaac Lab source checkout. The
generator writes the new task into ``source/isaaclab_tasks`` instead of creating
a separate project. From the Isaac Lab repository root, list and test it with:

.. code-block:: bash

   uv run python scripts/environments/list_envs.py --show_presets
   uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16
   uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>

Troubleshooting
---------------

**The project path is rejected**
   External projects must live outside the Isaac Lab repository. Enter the
   parent directory; the generator appends the project name automatically.

**The project name is rejected**
   Use a valid Python identifier containing letters, numbers, and underscores,
   without spaces or hyphens. The name cannot begin with a number.

**The task family or robot/config name is rejected**
   Use a valid Python identifier for each name. These names become package
   directories under ``src/<project-name>/tasks``.

**The CLI cannot find the generated task**
   Run ``uv sync`` and invoke the command from the generated project root. Then
   confirm the task appears in ``uv run python scripts/list_envs.py``.

**An optional backend module is missing**
   Add its extra before the command, such as ``uv run --extra ovphysx
   isaaclab ...`` or ``uv run --extra isaacsim isaaclab ...``.

**The generator reports a missing template dependency**
   Current versions obtain the renderer and prompts from the Isaac Lab
   environment; no manual ``pip install`` is required. Update the Isaac Lab
   checkout or installed package and run the generator again.

The generated ``README.md`` contains the same project-local commands and should
be kept up to date as the project evolves.
