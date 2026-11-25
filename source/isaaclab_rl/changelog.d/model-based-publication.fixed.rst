* Preserved the published dance checkpoint sensor, history and motion-window interfaces on the shared physical task for playback and LEAPP export.
* Exported newly trained SuperMimic actors with explicit reference inputs and the shared task observation functions.
* Consolidated motion-policy LEAPP export into the RSL-RL backend and kept checkpoint inputs explicit as paths or URLs.
* Reject unsupported Direct environments before starting agents that require every-step pre-reset observations.
