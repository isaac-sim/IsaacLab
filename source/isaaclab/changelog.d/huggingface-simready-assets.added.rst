* Added public Hugging Face ``hf://`` file URI support to asset loading and file helpers, sharing
  the existing HTTPS cache and USD dependency localization.
* Added :func:`isaaclab.utils.assets.discover_simready_assets` to discover commit-pinned packages
  registered in SimReady Central, and automatic root USD selection when loading package definitions.
