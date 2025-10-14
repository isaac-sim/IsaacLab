Sim2Real Deployment of Policies Trained in Isaac Lab
====================================================

Welcome to the Policy Deployment Guide! This section provides examples of training policies in Isaac Lab and deploying them to both simulation and real robots.

Below, you’ll find detailed examples of various policies for training and deploying them, along with essential configuration details.

.. toctree::
    :maxdepth: 1

    00_hover/hover_policy
    01_io_descriptors/io_descriptors_101


Resources with Available/Open-Source Code
-----------------------------------------

Explore these external resources featuring practical implementations with available/open-source code:

- **Deploying Policies in Isaac Sim**
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
    Modular Isaac Lab extension to train reach-task policies
    for the Kinova Gen3 arm, and deploy them on the real
    robot via a minimal ROS2 interface. Includes pre-trained 
    models -
    `Project repository on GitHub <https://github.com/louislelay/kinova_isaaclab_sim2real>`_

- **Wheeled Lab for Mobile Robots**
    Demonstrates three sim-to-real RL policies (drifting,
    elevation traversal, and visual navigation) for small
    RC cars, all trained in Isaac Lab and tested on
    low-cost hardware -
    `Project website on UW Robot Learning GitHub Pages <https://uwrobotlearning.github.io/WheeledLab/>`_

- **rl_sar: Sim2Real Framework for RL**
    C++/Python framework for sim and real deployment of RL
    policies on quadrupeds, humanoids, and wheeled robots.
    Supports ROS1/ROS2 and integrates with Isaac Lab via
    `robot_lab` -
    `Project repository on GitHub <https://github.com/fan-ziqi/rl_sar>`_

