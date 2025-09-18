# Foot Pedal Controls

## Overview

Three modes are available and you switch them with quick pedal taps (press past ~0.6 and release within ~0.3s):

- FORWARD_MODE (default)
- REVERSE_MODE
- VERTICAL_MODE (squat)

Mode switches:
- Left tap: FORWARD_MODE ↔ REVERSE_MODE
- Right tap: cycles between VERTICAL_MODE and FORWARD_MODE (or REVERSE_MODE depending on current)

## Movements

- Always available
  - Yaw (waist turn): twist the rudder axis (center bar)

- FORWARD_MODE (default)
  - Forward: press both pedals together (balanced, deeper = faster)
  - Strafe left: press only the left pedal
  - Strafe right: press only the right pedal

- REVERSE_MODE
  - Backward: press the right pedal

- VERTICAL_MODE (squat)
  - Squat down: press the left pedal (lower hips)
  - Stand up: release the left pedal (return to standing height)

## Limits and Sensitivity

- Max linear speed: ~1.0 m/s (configurable)
- Max yaw rate: ~1.0 rad/s (configurable)
- Hip height range: ~0.60 m (low) to ~0.72 m (stand)
- Yaw deadzone: |rudder| < ~0.1 ignored
