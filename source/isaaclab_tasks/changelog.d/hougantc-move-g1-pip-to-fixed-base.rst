Changed
^^^^^^^

* Removed the default XR camera PiP from the G1 locomanipulation task. Neither the locomanipulation
  nor fixed-base G1 task creates a PiP panel by default, preventing the articulated head camera from
  capturing the panel and producing a recursive view. The locomanipulation task retains its recorded
  robot camera.
