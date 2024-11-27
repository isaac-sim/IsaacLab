# ur5_parallel_control

## Description
This repository contains the development and implementation of a control system in form of a ROS node on a Husky robot for the UR5 robotic arm, focusing on parallel execution in both simulation and the real world. Using ROS2 and Nvidia IsaacSim, the project aims to bridge the Sim-to-Real Gap by synchronizing control strategies, joint positions, and handling potential singularities.


The key components of the project include:
- **Direct Joint Control**: Implementing direct angle control of the UR5's joints both in simulation and on the real robot mounted on a Clearpath Husky platform.
- **Parallel Simulation**: Running parallel simulations in IsaacSim to compare real-world robot behavior with simulated environments.
- **Sim-to-Real Gap Minimization**: Developing techniques to approximate and align the dynamics and behavior between the simulation and the physical system.
- **Singularity Handling**: Testing and comparing the handling of kinematic singularities between the real-world system and the simulation using MoveIt.

This project will serve as part of a Master's thesis, with the goal of advancing the understanding and implementation of robotic control in industrial applications.

## Technologies
- **ROS2 (Robot Operating System 2)**: For real-world control and system orchestration.
- **Nvidia IsaacSim**: For GPU-accelerated simulation and testing of the UR5 arm.
- **MoveIt 2**: For motion planning and singularity analysis.
- **Python & C++**: Primary programming languages for control nodes and system development.

## Repository Contents
This repository will contain:
- Source code for ROS2 nodes handling control of the UR5 arm.
- Scripts and configuration files for parallel simulations in IsaacSim.
