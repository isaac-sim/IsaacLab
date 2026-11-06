* Added optional :class:`~isaaclab.managers.ActionInputCfg` declarations on action
  controllers. Interactive runners activate inputs separately, preserving existing
  controller types, configuration paths, and custom APIs during training and playback.
* Added input selection before controller processing so recorded actions, action
  observations, and rewards reflect the selected raw inputs. Activated inputs drain
  before controllers and simulation resources close, with retryable cleanup failures.
