Limitations
===========

During the early development phase of both Newton and this Isaac Lab integration,
you are likely to encounter breaking changes as well as limited documentation.

We do not expect to be able to provide support or debugging assistance until the framework has reached an official release.

Here is a non-exhaustive list of capabilities currently supported in the Newton experimental feature branch grouped by extension:

* isaaclab:
    * Articulation API (supports both articulations and single-body articulations as rigid bodies)
    * Contact Sensor
    * Direct & Manager single agent workflows
    * Omniverse Kit visualizer
    * Newton visualizer
* isaaclab_assets:
    * Quadrupeds
        * Anymal-B, Anymal-C, Anymal-D
        * Unitree A1, Go1, Go2
        * Spot
    * Humanoids
        * Unitree H1 & G1
        * Cassie
    * Arms and Hands
        * Franka
        * UR10
        * Allegro Hand
    * Toy examples
        * Cartpole
        * Ant
        * Humanoid
* isaaclab_tasks:
    * Direct:
        * Cartpole (State, RGB, Depth)
        * Ant
        * Humanoid
        * Allegro Hand Repose Cube
    * Manager based:
        * Cartpole (State)
        * Ant
        * Humanoid
        * Locomotion (velocity flat terrain)
            * Anymal-B
            * Anymal-C
            * Anymal-D
            * Cassie
            * A1
            * Go1
            * Go2
            * Unitree G1
            * Unitree H1
        * Manipulation reach
            * Franka
            * UR10
