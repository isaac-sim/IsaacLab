Fixed
^^^^^

* Fixed ``isaaclab -i isaacsim`` failing in uv environments because uv rejects
  Isaac Sim's transitive pre-release pins (e.g. ``tinyobjloader==2.0.0rc13``);
  the install now passes ``--prerelease=allow`` alongside the existing index
  strategy flag.
* Fixed comma-separated RL framework selectors (e.g. ``-i "rl[rsl_rl,rl_games]"``)
  installing nothing; each listed framework extra is now installed, mirroring the
  ``ov`` selector handling.
