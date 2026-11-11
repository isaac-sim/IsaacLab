* Moved the recording status text out of the HUD header into the Recording section: a red REC line with the elapsed
  time, an unsaved-video note, and a green confirmation after saving. The section title turns red while recording.
* The default recording output folder is now ``recording`` in the working directory, which Git ignores. A relative
  recording output folder lands in the repository when running from a source checkout.
