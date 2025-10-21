# Isaac Lab: Framework

Isaac Lab includes its own set of interfaces and wrappers around Isaac Sim classes. One of the main goals behind this
decision is to have a unified description for different systems. While isaac Sim tries to be general for a wider
variety of simulation requires, our goal has been to specialize these for learning requirements. These include
features such as augmenting simulators with non-ideal actuator models, managing different observation and reward
settings, integrate different sensors, as well as provide interfaces to features that are currently not available in
Isaac Sim but are available from the physics side (such as deformable bodies).

We recommend the users to try out the demo scripts present in `scripts/demos` that display how different parts
of the framework can be integrated together.
