* Fixed LEAPP exports using zero PD gains for Newton-native explicit actuators by reading their live controller parameters.
* Fixed missing live LEAPP inputs for camera pose and intrinsics, joint wrenches, cable state, and articulation dynamics.
* Fixed LEAPP feedback for observation delays, digital filters, and integrators, including traceable digital-filter history updates.
* Fixed perspective-depth observations passing camera intrinsics as a ProxyArray instead of a Torch tensor.
