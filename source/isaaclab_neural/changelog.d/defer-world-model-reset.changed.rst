* Published original reset-record images immediately and deferred model-history reset until just before the next
  chunk is generated. ``WorldModel.reset`` initializes history without returning camera frames; the explicit reset
  event and configured default joint positions are retained.
