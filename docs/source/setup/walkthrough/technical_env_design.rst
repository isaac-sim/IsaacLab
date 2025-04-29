.. _walkthrough_technical_env_design:

Environment Design
====================

Our template is set up for the **direct** workflow, which means the environment class will manage all of these details 
centrally. We will need to write the code that will...

1. Define the training simulation and manage cloning
2. Define the robot
3. Apply the actions from the agent to the robot
4. Calculate and return the rewards and observations
5. Manage resetting and terminal states

But before we can dive into the details of changing the template to suite our needs, we need to cover some basic concepts
regarding how the simulation is logically organized.