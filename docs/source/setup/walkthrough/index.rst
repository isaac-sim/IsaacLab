.. _walkthrough:

Walkthrough
========================

So you finished installing Isaac Sim and Isaac Lab, and you verified that everything is working as expected...

Now what?

The following walkthrough will guide you through setting up an Isaac Lab extension project, adding a new robot to lab, designing an environment, and training a policy for that robot.
For this walkthrough, we will be starting with the Jetbot, a simple two wheeled differential base robot with a camera mounted on top, but the intent is for these guides to be general enough that you can use them to add your own robots and environments to Isaac Lab!

The end result of this walkthrough can be found in our tutorial project repository `here <https://github.com/isaac-sim/IsaacLabTutorial/tree/main>`_. Each branch of this repository
represents a different stage of modifying the default template project to achieve our goals.

.. toctree::
  :maxdepth: 1
  :titlesonly:

  project_setup
  concepts_env_design
  api_env_design
  technical_env_design
  training_jetbot_gt
  training_jetbot_reward_exploration
