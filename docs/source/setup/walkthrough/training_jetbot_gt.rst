.. _walkthrough_training_jetbot_gt:

Training the Jetbot: Ground Truth
======================================

With the environment defined, we can now start modifying our observations and rewards in order to train a policy 
to act as a controller for the Jetbot. As a user, we would like to be able to specify the desired direction for the Jetbot to drive, 
and have the wheels turn such that the robot drives in that specified direction as fast as possible. How do we achieve this with 
Reinforcement Learning (RL)?

Expanding the Environment
--------------------------

The very first thing we need to do is create the logic for setting commands for each Jetbot on the stage. Each command will be a unit vector, and 
we need one for every clone of the robot on the stage, which means a tensor of shape ``[num_envs, 3]``. Even though the Jetbot only navigates in the 
2D plane, by working with 3D vectors we get to make use of all the math utilities provided by Isaac Lab.  

It would also be a good idea to setup visualizations, so we can more easily tell what the policy is doing during training and inference.  
In this case, we will define two arrow ``VisualizationMarkers``: one to represent the "forward" direction of the robot, and one to 
represent the command direction.  When the policy is fully trained, these arrows should be aligned! Having these visualizations in place 
early helps us avoid "silent bugs": issues in the code that do not cause it to crash. 

To begin, we need to define the marker config and then instantiate the markers with that config. Add the following to the global scope of ``isaac_lab_tutorial_env.py``

.. code-block:: python

  from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
  from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
  import isaaclab.utils.math as math_utils

  def define_markers() -> VisualizationMarkers:
      """Define markers with various different shapes."""
      marker_cfg = VisualizationMarkersCfg(
          prim_path="/Visuals/myMarkers",
          markers={
                  "forward": sim_utils.UsdFileCfg(
                      usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                      scale=(0.5, 0.5, 1.0),
                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                  ),
                  "command": sim_utils.UsdFileCfg(
                      usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                      scale=(0.5, 0.5, 1.0),
                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                  ),
          },
      )
      return VisualizationMarkers(cfg=marker_cfg)

The ``VisualizationMarkersCfg`` defines USD prims to serve as the "marker".  Any prim will do, but generally you want to keep markers as simple as possible because the cloning of markers occurs at runtime on every time step.
This is because the purpose of these markers is for *debug visualization only* and not to be a part of the simulation: the user has full control over how many markers to draw when and where. 
NVIDIA provides several simple meshes on our public nucleus server, located at ``ISAAC_NUCLEUS_DIR``.

.. dropdown:: Code for the markers.py demo
   :icon: code

   .. literalinclude:: ../../../../scripts/demos/markers.py
      :language: python
      :linenos:


Exploring the problem
-----------------------

The command to the Jetbot is a unit vector in specifying the desired drive direction and we must make the agent aware of this some how
so it can adjust its actions accordingly.  There are many possible ways to do this, with the "zeroth order" approach to simply change the observation space to include 
this command. Maybe something like...

.. code-block:: python

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w 
        obs = torch.hstack((self.velocity, self.commands))
        observations = {"policy": obs}
        return observations

where ``self.commands`` is a tensor of the desired driving directions, one for each Jetbot on the stage. The root 