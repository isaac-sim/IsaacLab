* Added typed SAC model, distribution, network, and algorithm configurations with native RSL-RL defaults.
* Forwarded terminal observations through the vector wrapper without changing the environment's extras.
* Defaulted the SAC cosine learning-rate schedule length to the run's ``max_iterations`` times ``num_updates``.
