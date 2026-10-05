:orphan:

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

    import gymnasium as gym

    from isaaclab.app import launch_simulation

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry

    cfg = load_cfg_from_registry("Isaac-Reach-Franka", "env_cfg_entry_point")
    # start the simulation runtime this configuration needs, and stop it on exit
    with launch_simulation(cfg):
        # create base environment
        env = gym.make("Isaac-Reach-Franka", cfg=cfg)
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

  We notice the following performance in different rendering modes with the  ``Isaac-Reach-Franka`` environment
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


To record videos, add a :class:`~envs.utils.video_recorder_cfg.VideoRecorderCfg` to the environment
configuration. It records from a visualizer (or a camera sensor) into ``mp4`` clips while the environment steps,
so no wrapper or ``render_mode`` is needed, and the runtime the visualizer needs starts automatically.

As an example, the following code records 200-step clips of the ``Isaac-Reach-Franka`` environment from a Kit
visualizer every 1500 steps into the ``videos/train`` folder. See :ref:`how_to_record_video` for the other
sources and clip options.

.. code:: python

    import gymnasium as gym

    from isaaclab_visualizers.kit import KitVisualizerCfg

    from isaaclab.app import launch_simulation
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    # record from a Kit visualizer with this camera pose
    env_cfg.sim.visualizer_cfgs = KitVisualizerCfg(eye=(1.0, 1.0, 1.0), lookat=(0.0, 0.0, 0.0))
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="viz:kit", output_dir="videos/train", video_length=200, video_interval=1500)
    ]
    # select the Kit visualizer, as ``--viz kit`` does; the config above only sets its camera pose
    with launch_simulation(env_cfg, {"visualizer": "kit"}):
        env = gym.make(task_name, cfg=env_cfg)


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

Stable-Baselines3 requires finite continuous action bounds. When the environment has an unbounded action
space, :class:`~isaaclab_rl.sb3.Sb3VecEnvWrapper` exposes normalized ``[-1, 1]`` bounds to Stable-Baselines3
without changing the underlying environment. Set ``action_bounds`` when the policy uses a different
finite action domain.

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
