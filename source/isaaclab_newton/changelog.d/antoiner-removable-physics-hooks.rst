Added
^^^^^

* Added removable post-actuator and state-force callbacks and invalidated captured physics graphs when callback registrations changed, deferring recapture to the next step.

Fixed
^^^^^

* Preserved eager execution for solvers that reject CUDA graph capture when physics callbacks were registered or removed through the base Newton manager.
