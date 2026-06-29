# Use-Case Recipes

Pre-packaged answers to the recommender's interview for each common use case. The skill's recommender uses these mappings via the `recommended_for` field in `resources/combos.py`. This page exists so contributors can see them at a glance and so users can spot-check a recommendation.

## RL research / training

- **Default**: `pip-uv-source`
- **Why**: fastest install, full feature set, headless-friendly.
- **Fallbacks**: `binary-uv-source` (older distros), `kitless-uv` (when Isaac Sim features aren't needed).

Install command:

```bash
python3 scripts/recommend.py --non-interactive --use-case rl_research \
    --env-manager uv --isaacsim-source any
```

## Manipulation / teleop / vision

- **Default**: `pip-uv-source`
- **Why**: needs full Isaac Sim (PhysX, RTX, URDF importer). Pip path is the lightest weight install of that.
- **Alternates**: `binary-*` if you prefer a known-good Isaac Sim drop.

## Sim-to-real transfer

- **Default**: `pip-uv-source`
- **Why**: same reasoning as manipulation — needs full Isaac Sim. Often paired with rerun visualizer; the recommender's auto-selected `-i all` includes it.

## Contributing to Isaac Lab

- **Default**: `pip-uv-source`
- **Why**: standard contributor setup. You'll modify Isaac Lab code, so source clone is required.

## Contributing to Isaac Sim source

- **Default**: `source-uv-source`
- **Why**: only path that lets you modify Isaac Sim itself.
- **Caveat**: 80 GB disk, 30-60 minute build. Use the develop-branch compatibility note in `references/compatibility-matrix.md`.

## External extension / downstream package

- **Default**: `pip-only-uv`
- **Why**: no source clone, no `scripts/` — you're consuming Isaac Lab as a pip dependency from your own package.
- **Caveat**: does NOT include training scripts.

## Kit-less / Newton-only

- **Default**: `kitless-uv`
- **Why**: fastest install (no Isaac Sim). Targets Newton physics + RSL-RL.
- **Caveat**: PhysX / RTX / ROS / URDF importer unavailable. See `references/compatibility-matrix.md` for the feature gap.

## "Just exploring" / not sure yet

- **Default**: `pip-uv-source` (the docs' recommendation).
- **Fallback**: if the user doesn't want a 30 GB install just to try things, suggest `kitless-uv`.

---

## Selective `-i` tokens by use case

These are the `./isaaclab.sh -i ...` argument values typical for each use case. Combos default to `-i all`; advanced users can pick narrower sets.

| Use case            | Tokens                                       |
| ------------------- | -------------------------------------------- |
| RL research         | `all` (default) — includes `rl[rsl-rl,skrl,sb3,rl-games]` |
| Manipulation        | `all` + `mimic` if teleop / imitation learning |
| Sim2real            | `all` + `contrib[rlinf]` if using rlinf      |
| External extension  | `none` (pip-only combos do this implicitly)  |
| Kit-less RL         | `'newton,rl[rsl-rl]'`                        |
| Vision benchmark    | `'newton,ov[ovrtx],rl[rsl-rl]'`              |

The recommender does not currently expose `-i` token customization. To override, edit `combos.py` (the `_isaaclab_install_step("all")` calls) or pass `--combo` with a custom id.
