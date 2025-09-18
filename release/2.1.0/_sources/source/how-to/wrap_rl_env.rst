.. _how-to-env-wrappers:


Wrapping environments
=====================

.. currentmodule:: isaaclab

Environment wrappers are a way to modify the behavior of an environment without modifying the environment itself.
This can be used to apply functions to modify observations or rewards, record videos, enforce time limits, etc.
A detailed description of the API is available in the :class:`gymnasium.Wrapper` class.

At present, all RL environments inheriting from the :class:`~envs.ManagerBasedRLEnv` or :class:`~envs.DirectRLEnv` classes
are compatible with :class:`gymnasium.Wrapper`, since the base class implements the :class:`gymnasium.Env` interface.
In order to wrap an environment, you need to first initialize the base environment. After that, you can
wrap it with as many wrappers as you want by calling ``env = wrapper(env, *args, **kwargs)`` repeatedly.

For example, here is how you would wrap an environment to enforce that reset is called before step or render:

.. code-block:: python

    """Launch Isaac Sim Simulator first."""


    from isaaclab.app import AppLauncher

    # launch omniverse app in headless mode
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    """Rest everything follows."""

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    # create base environment
    cfg = load_cfg_from_registry("Isaac-Reach-Franka-v0", "env_cfg_entry_point")
    env = gym.make("Isaac-Reach-Franka-v0", cfg=cfg)
    # wrap environment to enforce that reset is called before step
    env = gym.wrappers.OrderEnforcing(env)


Wrapper for recording videos
----------------------------

The :class:`gymnasium.wrappers.RecordVideo` wrapper can be used to record videos of the environment.
The wrapper takes a ``video_dir`` argument, which specifies where to save the videos. The videos are saved in
`mp4 <https://en.wikipedia.org/wiki/MP4_file_format>`__ format at specified intervals for specified
number of environment steps or episodes.

To use the wrapper, you need to first install ``ffmpeg``. On Ubuntu, you can install it by running:

.. code-block:: bash

    sudo apt-get install ffmpeg

.. attention::

  By default, when running an environment in headless mode, the Omniverse viewport is disabled. This is done to
  improve performance by avoiding unnecessary rendering.

  We notice the following performance in different rendering modes with the  ``Isaac-Reach-Franka-v0`` environment
  using an RTX 3090 GPU:

  * No GUI execution without off-screen rendering enabled: ~65,000 FPS
  * No GUI execution with off-screen enabled: ~57,000 FPS
  * GUI execution with full rendering: ~13,000 FPS


The viewport camera used for rendering is the default camera in the scene called ``"/OmniverseKit_Persp"``.
The camera's pose and image resolution can be configured through the
:class:`~envs.ViewerCfg` class.


.. dropdown:: Default parameters of the ViewerCfg class:
    :icon: code

    .. literalinclude:: ../../../source/isaaclab/isaaclab/envs/common.py
        :language: python
        :pyobject: ViewerCfg


After adjusting the parameters, you can record videos by wrapping the environment with the
:class:`gymnasium.wrappers.RecordVideo` wrapper and enabling the off-screen rendering
flag. Additionally, you need to specify the render mode of the environment as ``"rgb_array"``.

As an example, the following code records a video of the ``Isaac-Reach-Franka-v0`` environment
for 200 steps, and saves it in the ``videos`` folder at a step interval of 1500 steps.

.. code:: python

    """Launch Isaac Sim Simulator first."""


    from isaaclab.app import AppLauncher

    # launch omniverse app in headless mode with off-screen rendering
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    """Rest everything follows."""

    import gymnasium as gym

    # adjust camera resolution and pose
    env_cfg.viewer.resolution = (640, 480)
    env_cfg.viewer.eye = (1.0, 1.0, 1.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    # create isaac-env instance
    # set render mode to rgb_array to obtain images on render calls
    env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array")
    # wrap for video recording
    video_kwargs = {
        "video_folder": "videos/train",
        "step_trigger": lambda step: step % 1500 == 0,
        "video_length": 200,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)


Wrapper for learning frameworks
-------------------------------

Every learning framework has its own API for interacting with environments. For example, the
`Stable-Baselines3`_ library uses the `gym.Env <https://gymnasium.farama.org/api/env/>`_
interface to interact with environments. However, libraries like `RL-Games`_, `RSL-RL`_ or `SKRL`_
use their own API for interfacing with a learning environments. Since there is no one-size-fits-all
solution, we do not base the :class:`~envs.ManagerBasedRLEnv` and :class:`~envs.DirectRLEnv` classes on any particular learning framework's
environment definition. Instead, we implement wrappers to make it compatible with the learning
framework's environment definition.

As an example of how to use the RL task environment with Stable-Baselines3:

.. code:: python

    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    # create isaac-env instance
    env = gym.make(task_name, cfg=env_cfg)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)


.. caution::

  Wrapping the environment with the respective learning framework's wrapper should happen in the end,
  i.e. after all other wrappers have been applied. This is because the learning framework's wrapper
  modifies the interpretation of environment's APIs which may no longer be compatible with :class:`gymnasium.Env`.


Adding new wrappers
-------------------

All new wrappers should be added to the :mod:`isaaclab_rl` module.
They should check that the underlying environment is an instance of :class:`isaaclab.envs.ManagerBasedRLEnv`
or :class:`~envs.DirectRLEnv`
before applying the wrapper. This can be done by using the :func:`unwrapped` property.

We include a set of wrappers in this module that can be used as a reference to implement your own wrappers.
If you implement a new wrapper, please consider contributing it to the framework by opening a pull request.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _SKRL: https://skrl.readthedocs.io
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
