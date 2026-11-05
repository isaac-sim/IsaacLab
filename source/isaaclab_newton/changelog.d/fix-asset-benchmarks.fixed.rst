* Allocated mocked Newton manager buffers on the benchmark device and removed timed callback call histories that distorted latency measurements.
* Initialized the joint-limit clamping counter in mocked Newton articulation benchmarks so joint position limit writes completed.
* Kept mocked rigid-object root and body kinematics consistent when refreshing benchmark data.
