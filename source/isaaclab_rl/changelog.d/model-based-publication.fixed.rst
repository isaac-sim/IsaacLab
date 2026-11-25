* Preserved the published dance checkpoint sensor, history and motion-window interfaces on the shared physical task for playback and LEAPP export.
* Exported newly trained SuperMimic actors with explicit reference inputs and the shared task observation functions.
* Reused the RSL-RL playback and LEAPP export flows, standard observation history and action annotations for motion policies; retained explicit checkpoint paths or URLs.
* Reject unsupported Direct environments before starting agents that require every-step pre-reset observations.
