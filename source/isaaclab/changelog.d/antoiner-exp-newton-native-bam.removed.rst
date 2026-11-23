* **Breaking:** Removed ``isaaclab.actuators.actuator_bam_cfg.BAM_DRIVE_API``. BAM servos author Newton's
  ``NewtonPDControlAPI`` with duty output, ``NewtonCurrentDutyClampingAPI`` and ``NewtonDCMotorAPI`` instead; look up BAM
  actuator prims by those schemas.
