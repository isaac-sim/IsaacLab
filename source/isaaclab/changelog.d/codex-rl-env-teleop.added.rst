* Added optional :class:`~isaaclab.managers.ActionInputCfg` declarations on action
  controllers. Interactive runners activate inputs separately, preserving existing
  controller types, configuration paths, and custom APIs during training and playback.
* Added input selection before controller processing so recorded actions, action
  observations, and rewards reflect the selected raw inputs. Activated inputs drain
  before controllers and simulation resources close, with retryable cleanup failures.
* Added an input preparation phase before any action input resolves controller commands,
  allowing shared devices to supply one sample regardless of controller ordering.
* Added selected-row operator reset requests to the normal manager-based RL episode
  boundary, preserving the completed physics transition, rewards, and task success labels.
* Added a configurable minimum number of completed recorder steps for episode export,
  allowing demonstration collection to exclude empty reset-only episodes.
