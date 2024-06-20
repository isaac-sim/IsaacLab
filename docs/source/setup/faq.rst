Frequently Asked Questions
==========================

Where does Isaac Lab fit in the Isaac ecosystem?
------------------------------------------------

Over the years, NVIDIA has developed a number of tools for robotics and AI. These tools leverage
the power of GPUs to accelerate the simulation both in terms of speed and realism. They show great
promise in the field of simulation technology and are being used by many researchers and companies
worldwide.

`Isaac Gym`_ :cite:`makoviychuk2021isaac` provides a high performance GPU-based physics simulation
for robot learning. It is built on top of `PhysX`_ which supports GPU-accelerated simulation of rigid bodies
and a Python API to directly access physics simulation data. Through an end-to-end GPU pipeline, it is possible
to achieve high frame rates compared to CPU-based physics engines. The tool has been used successfully in a
number of research projects, including legged locomotion :cite:`rudin2022learning` :cite:`rudin2022advanced`,
in-hand manipulation :cite:`handa2022dextreme` :cite:`allshire2022transferring`, and industrial assembly
:cite:`narang2022factory`.

Despite the success of Isaac Gym, it is not designed to be a general purpose simulator for
robotics. For example, it does not include interaction between deformable and rigid objects, high-fidelity
rendering, and support for ROS. The tool has been primarily designed as a preview release to showcase the
capabilities of the underlying physics engine. With the release of `Isaac Sim`_, NVIDIA is building
a general purpose simulator for robotics and has integrated the functionalities of Isaac Gym into
Isaac Sim.

`Isaac Sim`_ is a robot simulation toolkit built on top of Omniverse, which is a general purpose platform
that aims to unite complex 3D workflows. Isaac Sim leverages the latest advances in graphics and
physics simulation to provide a high-fidelity simulation environment for robotics. It supports
ROS/ROS2, various sensor simulation, tools for domain randomization and synthetic data creation.
Tiled rendering support in Isaac Sim allows for vectorized rendering across environments, along with
support for running in the cloud using `Isaac Automator`_.
Overall, it is a powerful tool for roboticists and is a huge step forward in the field of robotics
simulation.

With the release of above two tools, NVIDIA also released an open-sourced set of environments called
`IsaacGymEnvs`_ and `OmniIsaacGymEnvs`_, that have been built on top of Isaac Gym and Isaac Sim respectively.
These environments have been designed to display the capabilities of the underlying simulators and provide
a starting point to understand what is possible with the simulators for robot learning. These environments
can be used for benchmarking but are not designed for developing and testing custom environments and algorithms.
This is where Isaac Lab comes in.

Isaac Lab is built on top of Isaac Sim to provide a unified and flexible framework
for robot learning that exploits latest simulation technologies. It is designed to be modular and extensible,
and aims to simplify common workflows in robotics research (such as RL, learning from demonstrations, and
motion planning). While it includes some pre-built environments, sensors, and tasks, its main goal is to
provide an open-sourced, unified, and easy-to-use interface for developing and testing custom environments
and robot learning algorithms. It not only inherits the capabilities of Isaac Sim, but also adds a number
of new features that pertain to robot learning research. For example, including actuator dynamics in the
simulation, procedural terrain generation, and support to collect data from human demonstrations.

Isaac Lab replaces the previous `IsaacGymEnvs`_, `OmniIsaacGymEnvs`_ and `Orbit`_ frameworks and will
be the single robot learning framework for Isaac Sim. Previously released frameworks are deprecated
and we encourage users to follow our `migration guides`_ to transition over to Isaac Lab.


Why should I use Isaac Lab?
---------------------------

Since Isaac Sim remains closed-sourced, it is difficult for users to contribute to the simulator and build a
common framework for research. On its current path, we see the community using the simulator will simply
develop their own frameworks that will result in scattered efforts with a lot of duplication of work.
This has happened in the past with other simulators, and we believe that it is not the best way to move
forward as a community.

Isaac Lab provides an open-sourced platform for the community to drive progress with consolidated efforts
toward designing benchmarks and robot learning systems as a joint initiative. This allows us to reuse
existing components and algorithms, and to build on top of each other's work. Doing so not only saves
time and effort, but also allows us to focus on the more important aspects of research. Our hope with
Isaac Lab is that it becomes the de-facto platform for robot learning research and an environment *zoo*
that leverages Isaac Sim. As the framework matures, we foresee it benefitting hugely from the latest
simulation developments (as part of internal developments at NVIDIA and collaborating partners)
and research in robotics.

We are already working with labs in universities and research institutions to integrate their work into Isaac Lab
and hope that others in the community will join us too in this effort. If you are interested in contributing
to Isaac Lab, please reach out to us.


.. _PhysX: https://developer.nvidia.com/physx-sdk
.. _Isaac Sim: https://developer.nvidia.com/isaac-sim
.. _Isaac Gym: https://developer.nvidia.com/isaac-gym
.. _IsaacGymEnvs: https://github.com/isaac-sim/IsaacGymEnvs
.. _OmniIsaacGymEnvs: https://github.com/isaac-sim/OmniIsaacGymEnvs
.. _Orbit: https://isaac-orbit.github.io/
.. _Isaac Automator: https://github.com/isaac-sim/IsaacAutomator
.. _migration guides: ../migration/index.html
