* Adopted shared PVA output storage and optional output groups; request acceleration state only when enabled, and normalize initialization/STOP defaults.
* Isolate optional PVA update compilation from other feature subsets and the shared reset path to reduce cold sensor startup time.
* Reused shared sensor math, packing, and reset operations while retaining native gathering and solver-specific policies.
