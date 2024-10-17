Contact Sensor
================

The contact sensor is designed to return the net normal force acting on a given ridged body. The sensor is written to behave as a physical object, and so the "scope" of the contact sensor is limited to the body that defines it.  A multi-leged robot that needs contact information for its feet would require one sensor per foot to be defined in the environment.

To define a contact sensor, you must specify both the rigid body of the sensor, but the bodies with wich it may collide. These data are used to index the parts of the GPU simulation state relavent to retrieving contact information, and allows for Isaac Lab 