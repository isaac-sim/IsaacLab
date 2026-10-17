* Added reset-time rigid-object geometry and inertia selection using stock Newton's external contacts,
  without rebuilding solver topology. Internal MuJoCo contacts and sleeping are unsupported for mesh variants.
* Published external-contact forces through MuJoCo Warp's public observables without overwriting
  collision rows used by filtered contact sensors.
