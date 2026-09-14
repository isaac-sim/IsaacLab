Added
^^^^^

* Added filtered contact-position and friction-force tracking to the OvPhysX contact
  sensor, including friction-force history and selective reset. Both tracking options
  required non-empty filters and a positive ``max_contact_data_count_per_prim``;
  overflowing detailed-contact reads raised an error without publishing partial data.
  Contact positions and friction forces were aggregated in a single kernel launch
  while retaining independent raw-data buffers. Aggregate friction reporting and the
  existing normal-only force aliases remained unchanged.
