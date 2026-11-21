* Added GenCoord and MaxCoord model-based G1 tasks beside the core motion tracker, reusing its motor configuration and preserving reference-driven rewards and physical randomization. Return-based reference reset sampling is owned by the task.
* Integrated shared read-only CSV motion adapters for model-based tracking and clip-level distributed sharding without a dataset conversion step.
* Consolidated model-based reference goals, kinematics queries, and randomization with the motion command; motion storage reuses the shared library.
* Used the simulated articulation for reference kinematics without a dataset preparation script.
* Added a default camera for native model-based motion tracking video recording.
