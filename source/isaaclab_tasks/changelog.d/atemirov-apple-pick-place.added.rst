* Added the ``IsaacContrib-PickPlace-Apple-G1-Dex3`` environment, in which a Unitree G1 with Dex3 hands moves an
  apple onto a plate. Its scene, 16 saved starts, actuator gains and 30 Hz control match the evaluation of a GR00T
  N1.7 policy fine-tuned on real demonstrations, and its observation and action terms follow that policy's inputs
  and outputs. It runs on PhysX, OvPhysX (``physics=ovphysx``) or Newton (``physics=newton_mjwarp``).
* Added the ``IsaacContrib-PickPlace-Apple-G1-Dex3-Cosmos`` task: the G1 apple pick-and-place task with a head
  camera image that Cosmos generates from depth, with a prompt describing the real demonstrations and three
  appearance variations. ``env.cosmos`` and ``env.cosmos_seed`` set the prompts and seeds per environment.
