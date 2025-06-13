Sim2Real Deployment of Policies Trained in Isaac Lab
====================================================

Welcome to the Policy Deployment Guide! This section provides examples of training policies in Isaac Lab and deploying them to both simulation and real robots.

Below, you’ll find detailed examples of various policies for training and deploying them, along with essential configuration details.

.. toctree::
    :maxdepth: 1

    00_hover/hover_policy


Resources with Available/Open-Source Code
-----------------------------------------

Explore these external resources featuring practical implementations with available/open-source code:

- **Deploying Policies in Isaac Sim**:
    Step-by-step guide to deploying exported RL
    policies trained in Isaac Lab, covering demos for
    Unitree H1 and Boston Dynamics Spot and sim-to-real
    considerations -
    `Tutorial in Isaac Sim Documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_lab_tutorials/tutorial_policy_deployment.html>`_

- **Closing the Sim-to-Real Gap: Training Spot Quadruped Locomotion with NVIDIA Isaac Lab**
    Shows how to train a robust locomotion policy for the
    Boston Dynamics Spot quadruped in Isaac Lab and deploy
    it zero-shot on the real robot using NVIDIA Jetson Orin -
    `Blog post on NVIDIA Technical Blog <https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/>`_

- **Kinova Gen3 RL & Sim2Real Toolkit**
    Modular extension for Isaac Lab that trains RL reach-task
    policies on Kinova Gen3 and runs them in simulation or on
    the real arm via a minimal ROS 2 sim-to-real interface -
    `Project repository on GitHub <https://github.com/louislelay/kinova_isaaclab_sim2real>`_

- **Wheeled Lab for Mobile Robots**
    Demonstrates three sim-to-real RL policies (drifting,
    elevation traversal, and visual navigation) for small
    RC cars, all trained in Isaac Lab and tested on
    low-cost hardware -
    `Project website on UW Robot Learning GitHub Pages <https://uwrobotlearning.github.io/WheeledLab/>`_
