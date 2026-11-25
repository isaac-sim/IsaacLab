* Preserved the published dance checkpoint sensor, history and motion-window interfaces on the shared physical task for playback and LEAPP export.
* Exported newly trained SuperMimic actors with explicit reference inputs and the shared task observation functions.
* Kept motion-policy LEAPP export in its dedicated backend and checkpoint inputs explicit as paths or URLs.
* Reject unsupported Direct environments before starting agents that require every-step pre-reset observations.
