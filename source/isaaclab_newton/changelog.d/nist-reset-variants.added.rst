* Added reset-time rigid-object geometry and inertia selection using stock Newton's external contacts,
  without rebuilding solver topology. Backend-owned banks retained selected IDs and render descriptors;
  selection updated collision geometry, margins, and gaps while preserving material randomization.
  Compatible root-joint topology was required; internal MuJoCo contacts and sleeping were unsupported.
