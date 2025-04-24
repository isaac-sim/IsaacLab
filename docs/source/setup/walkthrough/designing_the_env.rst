.. _walkthrough_designing_the_env:

Designing the Environment
==========================

Now that we have our project installed, we can start designing the environment. In the traditional description 
of a reinforcement learning problem, the environment is responsible for using the actions produced by the agent to 
update the the state of the "world", and finally compute and return the observations and the reward signal.

Our template is set up for the **direct** workflow, which means the environment class will manage all of these details 
centrally. We will need to write the code that will...

1. Define the training simulation and manage cloning
2. Define the robot
3. Apply the actions from the agent to the robot
4. Calculate and return the rewards and observations
5. Manage resetting and terminal states

Class and Config
-----------------

To begin, Navigate to 
