* The Newton GL Recording section keeps an unsaved video until the next recording has been encoded, so a failed Start
  no longer deletes it, and a screenshot without the HUD no longer shows the HUD when the video includes it. A video
  that cannot be saved when the window closes now logs a warning naming the file that was left behind.
* A fixed-length recording whose encode failed no longer saves the earlier unsaved clip under a "Saved" message. A
  window resize during a recording, and a failed frame read, now show a message instead of silently dropping frames or
  stopping the simulation loop.
