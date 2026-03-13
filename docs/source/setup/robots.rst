.. _isaac-lab-robots:

Robot Configurations
=======================

Robots are entirely defined as instances of configurations within Isaac Lab.  If you examine ``source/isaaclab_assets/isaaclab_assets/robots``, you will see a number of files, each of which
contains configurations for the robot in question.  The purpose of these individual files is to better define scope for all the different robots, but there is nothing preventing
you from :ref:`adding your own <tutorial-add-new-robot>` to your project or even to the ``isaaclab`` repository! For example, consider the following configuration for
the Dofbot

.. code-block:: python

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets.articulation import ArticulationCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    DOFBOT_CONFIG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Dofbot/dofbot.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
            },
            pos=(0.25, -0.25, 0.0),
        ),
        actuators={
            "front_joints": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
            "joint3_act": ImplicitActuatorCfg(
                joint_names_expr=["joint3"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
            "joint4_act": ImplicitActuatorCfg(
                joint_names_expr=["joint4"],
                effort_limit_sim=100.0,
                velocity_limit_sim=100.0,
                stiffness=10000.0,
                damping=100.0,
            ),
        },
    )

This completely defines the dofbot! You could copy this into a ``.py`` file and import it as a module and you would be able to use the dofbot in
your own lab sims. One common feature you will see in any config defining things with state is the presence of an ``InitialStateCfg``.  Remember, the configurations
are what informs vectorization, and it's the ``InitialStateCfg`` that describes the state of the joints of our robot when it gets created in each environment. The
``ImplicitActuatorCfg`` defines the joints of the robot using the default actuation model determined by the joint time.  Not all joints need to be actuated, but you
will get warnings if you don't.  If you aren't planning on using those undefined joints, you can generally ignore these.
