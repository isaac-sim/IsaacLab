Added
^^^^^

* Added removable post-actuator and state-force callbacks and invalidated captured physics graphs when callback registrations changed, deferring recapture to the next step.

Fixed
^^^^^

* Deferred Featherstone CUDA graph capture until the requested physics step so callback changes before the first replay kept lazy solver scratch valid.

* Preserved eager execution for solvers that reject CUDA graph capture when physics callbacks were registered or removed through the base Newton manager.
