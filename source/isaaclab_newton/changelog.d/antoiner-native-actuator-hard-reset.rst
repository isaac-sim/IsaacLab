Changed
^^^^^^^

* **Breaking:** Cleared model-bound execution hooks on hard reset. Extensions must register hooks during each ``PHYSICS_READY`` event and release model-bound resources on ``MODEL_INIT`` and ``STOP``; lifecycle subscriptions persist across hard resets.

Fixed
^^^^^

* Released native actuator owners before hard model resets, then rebuilt their bindings and CUDA graphs.
