# unitree_model

This is a repository providing Unitree's robot 3D models for different environments.

For more information about robots model, please visit [unitree_ros](https://github.com/unitreerobotics/unitree_ros).


## Generate from urdf

Please follow [import_urdf tutorial](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html#getting-started) to convert URDF. 
Due to some bugs in the python script in version 4.5, please use `Direct Import`.

Specify the settings as follows:
- Select **Movebale Base** in Links
- Select **Stiffness** in Joint Configuration
- Select **Force** in Drive Type
- **Allow Self-Collision**