* Removed PPISP from the OVRTX renderer; cameras apply it through ``CameraCfg.tensor_processors``.
* Changed ``OVRTXRenderer.read_output`` to return ``False`` when asynchronous rendering has no new capture.
