Using environment wrappers
==========================

Environment wrappers are a way to modify the behavior of an environment without modifying the environment itself.
This can be used to apply functions to modify observations or rewards, record videos, enforce time limits, etc.
A detailed description of the API is available in the `gym.Wrapper <https://gymnasium.farama.org/api/wrappers/>`_ class.


At present, all environments inheriting from the :class:`omni.isaac.orbit_envs.isaac_env.IsaacEnv` class
are compatible with ``gym.Wrapper``, since the base class implements the ``gym.Env`` interface.
In order to wrap an environment, you need to first initialize the base environment. After that, you can
wrap it with as many wrappers as you want by calling `env = wrapper(env, *args, **kwargs)` repeatedly.

For example, here is how you would wrap an environment to enforce that reset is called before step or render:

.. code-block:: python

    """Launch Isaac Sim Simulator first."""


    from omni.isaac.orbit.app import AppLauncher

    # launch omniverse app in headless mode
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    """Rest everything follows."""

    import gym

    import omni.isaac.orbit_envs  # noqa: F401
    from omni.isaac.orbit_envs.utils import load_default_env_cfg

    # create base environment
    cfg = load_default_env_cfg("Isaac-Reach-Franka-v0")
    env = gym.make("Isaac-Reach-Franka-v0", cfg=cfg)
    # wrap environment to enforce that reset is called before step
    env = gym.wrappers.OrderEnforcing(env)


Wrapper for recording videos
----------------------------

The :class:`gym.wrappers.RecordVideo <gym:RecordVideo>` wrapper can be used to record videos of the environment.
The wrapper takes a ``video_dir`` argument, which specifies where to save the videos. The videos are saved in
`mp4 <https://en.wikipedia.org/wiki/MP4_file_format>`__ format at specified intervals for specified
number of environment steps or episodes.

To use the wrapper, you need to first install ``ffmpeg``. On Ubuntu, you can install it by running:

.. code-block:: bash

    sudo apt-get install ffmpeg

The :class:`omni.isaac.orbit.envs.RlEnv` supports the rendering modes:

* **"human"**: When you want to render the environment to the screen. It does not return any image.
* **"rgb_array"**: When you want to get the rendered image as a numpy array. It returns the image as a numpy array.
  This mode is only possible when viewport is enabled, i.e. either the GUI window is open or off-screen rendering flag
  is enabled.

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
:class:`omni.isaac.orbit.envs.base_env_cfg.ViewerCfg` class.


.. dropdown:: :fa:`eye,mr-1` Default parameters of the :class:`ViewerCfg` in the ``base_env_cfg.py`` file:

   .. literalinclude:: ../../../source/extensions/omni.isaac.orbit/omni/isaac/orbit/envs/base_env_cfg.py
      :language: python
      :lines: 23-38
      :linenos:
      :lineno-start: 31


After adjusting the parameters, you can record videos by wrapping the environment with the
:class:`gym.wrappers.RecordVideo <gym:RecordVideo>` wrapper and enabling the off-screen rendering
flag. As an example, the following code records a video of the ``Isaac-Reach-Franka-v0`` environment
for 200 steps, and saves it in the ``videos`` folder at a step interval of 1500 steps.

.. code:: python

    """Launch Isaac Sim Simulator first."""


    from omni.isaac.orbit.app import AppLauncher

    # launch omniverse app in headless mode with off-screen rendering
    app_launcher = AppLauncher(headless=True, offscreen_render=True)
    simulation_app = app_launcher.app

    """Rest everything follows."""


    import gym

    # adjust camera resolution and pose
    env_cfg.viewer.resolution = (640, 480)
    env_cfg.viewer.eye = (1.0, 1.0, 1.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)
    # create isaac-env instance
    env = gym.make(task_name, cfg=env_cfg)
    # wrap for video recording
    video_kwargs = {
        "video_folder": "videos",
        "step_trigger": lambda step: step % 1500 == 0,
        "video_length": 200,
    }
    env = gym.wrappers.RecordVideo(env, **video_kwargs)


Wrapper for learning frameworks
-------------------------------

Every learning framework has its own API for interacting with environments. For example, the
`Stable Baselines3 <https://stable-baselines3.readthedocs.io/en/master/>`__ library uses the
`gym.Env <https://gymnasium.farama.org/api/env/>`__ interface to interact with environments.
However, libraries like `RL-Games <https://github.com/Denys88/rl_games>`__ or
`RSL-RL <https://github.com/leggedrobotics/rsl_rl>`__ use their own API for interfacing with a
learning environments. Since there is no one-size-fits-all solution, we do not base the :class:`IsaacEnv`
class on any particular learning framework's environment definition. Instead, we implement
wrappers to make it compatible with the learning framework's environment definition.

As an example of how to use the :class:`IsaacEnv` with Stable-Baselines3:

.. code:: python

    from omni.isaac.orbit_envs.utils.wrappers.sb3 import Sb3VecEnvWrapper

    # create isaac-env instance
    env = gym.make(task_name, cfg=env_cfg, render=headless)
    # wrap around environment for stable baselines
    env = Sb3VecEnvWrapper(env)


.. caution::

  Wrapping the environment with the respective learning framework's wrapper should happen in the end,
  i.e. after all other wrappers have been applied. This is because the learning framework's wrapper
  modifies the interpretation of environment's APIs which may no longer be compatible with ``gym.Env``.


To add support for a new learning framework, you need to implement a wrapper class that
converts the :class:`IsaacEnv` to the learning framework's environment definition. This
wrapper class should typically inherit from the ``gym.Wrapper`` class. We include a
set of these wrappers in the :mod:`omni.isaac.orbit_envs.utils.wrappers` module. You can
use these wrappers as a reference to implement your own wrapper for a new learning framework.
