* Implemented the Conveyor controller using a pinned extraction of Newton's force model and exposed solved-contact callbacks.
* Made solver-output requests consumer-owned and preserved late contact-sensor bindings. Releasing a request keeps cached CUDA graphs when the active outputs stay the same.
