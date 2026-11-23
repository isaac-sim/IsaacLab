* Changed the GR1T2 and G1 locomanipulation Mimic environments to prepare the ``ovphysx`` or
  ``newton_mjwarp`` backend selected through their base teleoperation tasks' physics selectors, as those
  tasks' environments do. The G1 locomanipulation SDG task (``Isaac-G1-SteeringWheel-Locomanipulation``)
  stays on Isaac Sim PhysX because its environment does not run that preparation.
* Changed ``scripts/imitation_learning/robomimic/train.py`` to train without launching a simulation runtime
  when Isaac Sim is not installed; training reads only the dataset. With Isaac Sim it still runs inside
  the headless runtime.
