Recording video clips during training
=====================================

Isaac Lab supports recording video clips during training using the
`gymnasium.wrappers.RecordVideo <https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/>`_ class.

This feature can be enabled by installing ``ffmpeg`` and using the following command line arguments with the training
script:

* ``--video``: enables video recording during training
* ``--video_length``: length of each recorded video (in steps)
* ``--video_interval``: interval between each video recording (in steps)

Make sure to also add the ``--enable_cameras`` argument when running headless.
Note that enabling recording is equivalent to enabling rendering during training, which will slow down both startup and runtime performance.

Example usage:

.. code-block:: shell

    python scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video --video_length 100 --video_interval 500


The recorded videos will be saved in the same directory as the training checkpoints, under
``IsaacLab/logs/<rl_workflow>/<task>/<run>/videos/train``.
