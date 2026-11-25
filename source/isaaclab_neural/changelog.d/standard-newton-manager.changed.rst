* Removed ``NewtonNeRDManager`` and the injected solver-construction factory. Neural solver adapters load and validate their checkpoints through the ordinary Newton solver path and consume Newton control buffers directly.
* Used the standard Newton collision lifecycle for standalone learned dynamics, including scenes without contact sensors.
