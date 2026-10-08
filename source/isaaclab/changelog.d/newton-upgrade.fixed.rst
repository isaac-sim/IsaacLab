* Preserved TorchScript actuator checkpoint support with the updated Newton runtime by exporting
  dynamic-batch checkpoints during USD authoring. Existing ``ActuatorNetMLPCfg`` and
  ``ActuatorNetLSTMCfg`` configurations require no changes.
* Fixed native actuator parameter access on PhysX and OVPhysX with Newton's cached DOF mappings.
