Limitations
===========

The Newton backend is in beta. Breaking changes and incomplete documentation are
still expected, and official support or debugging assistance will only be
available once the integration reaches an official release.

Here is a non-exhaustive list of capabilities currently supported by the Newton
backend, grouped by extension:

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
