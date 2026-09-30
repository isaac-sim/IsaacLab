# Handoff: Manim animations for the IsaacLab actuator documentation

## Context

You are producing conceptual animations for `docs/source/concepts/actuators.rst` in IsaacLab
(branch `antoine/tmp_articulation`, PR #6839, worktree
`/home/antoiner/Documents/IsaacLab/.worktrees/tmp-articulation`). The page already has two kinds
of media in `docs/source/_static/actuators/`:

- Simulator clips (`*-clip.webp`): five-pendulum parameter sweeps rendered from the simulator.
- Static curve plots (`*-curve-light.png` / `*-curve-dark.png`): matplotlib step responses.

What is missing is the conceptual/mathematical layer — how the machinery works — which is what
you will build with Manim (Manim Community Edition). Five animations, specified below.

## Hard constraints

- **Do not add Manim (or anything else) to the repository's dependency tree.** Render offline in
  your own environment (e.g. a scratch venv with `manim` installed).
- **Do not commit rendered artifacts or scene scripts yet.** Put scene sources in a working
  directory (e.g. `manim-src/` outside the repo or in your scratchpad) and rendered outputs next
  to them. Hosting is being settled separately (likely
  `https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/`, the server the walkthrough
  webp clips already use); the doc integration happens after that decision.
- **Output format**: animated `.webp` (looping, no audio), matching the existing clips. Target
  ≤ 600 KB each at ~960px wide; the existing clips range 470 KB–1.2 MB and reviewers already
  flagged size, so stay lean (short loops, limited palette, moderate frame rate ~24 fps).
- **Light and dark variants** of every animation, matching the repo convention
  (`<name>-light.*` / `<name>-dark.*`). Existing figures use `:class: only-light` /
  `:class: only-dark` in the RST. Dark background `#1a1a1a`-ish; check the existing
  `*-curve-dark.png` files for the palette in use and match it.
- **Technical accuracy is non-negotiable.** Every equation and behavior shown must match the
  implementation. Source-of-truth files are listed per animation; read them before animating.
  Where this handoff and the code disagree, the code wins — and flag the discrepancy.
- No NVIDIA logos, no 3Blue1Brown branding; neutral style consistent with the existing figures.

## The five animations

### 1. `pipeline-flow` — the actuator pipeline (highest priority)

Replaces/augments the static `pipeline-light.png` / `pipeline-dark.png` (see their alt text in
`actuators.rst` around lines 98–112 for the intended semantics).

Animate a command pulse flowing through the architecture:

1. User calls `actuators.target_command.set_position_index(...)` → pulse enters the
   **ActuatorCollection** (joint-indexed staging buffers).
2. During `write_data_to_sim()`, the pulse splits into **three paths**:
   - **Lab explicit model**: model computes torque on the host, clips it, submits effort to the
     solver. Show `output_command` being filled on this path only.
   - **Implicit drive**: targets pass through unchanged; a PD block *inside the solver box*
     consumes them (gains live in the solver).
   - **Native**: on Newton the actuator block sits *inside the solver* (CUDA-graph region);
     on PhysX/OVPhysX a "host adapter" block processes during `write_data_to_sim()` and submits.
3. Telemetry (`computed_effort`, `applied_effort`) flows back to the collection; annotate that
   native paths bypass `output_command`.

Timing is the point: make it visible what happens during `write_data_to_sim()` versus inside the
solver step. Sources: `docs/source/concepts/actuators.rst` (sections "The actuator pipeline",
"Backend submission"), `source/isaaclab/isaaclab/actuators/actuator_collection.py`
(`ActuatorCollection.compute`).

### 2. `implicit-vs-explicit-stability` — why explicit actuators diverge at high gains

Two synchronized panels driving the same 1-DOF mass:

- **Implicit**: PD evaluated continuously (solver-side) — smooth restoring force.
- **Explicit**: error sampled once per control step, torque held constant between samples
  (zero-order hold). Draw the staircase torque against the continuous one.

Start at moderate kp where both trajectories overlap; ramp kp until the explicit panel
oscillates and diverges while the implicit one stays stable. Annotate: "solver applies the
drive" vs "model evaluates PD once per step". End card: the practical rule from the doc —
policies trained on implicit gains may need adjustment for explicit actuators.

Sources: `actuators.rst` "Implicit vs. explicit" section; `IdealPDActuator.compute` in
`source/isaaclab/isaaclab/actuators/actuator_pd.py`. The PD law shown must be
τ = k_p (q_des − q) + k_d (q̇_des − q̇) + τ_ff, clipped to ±`actuator_effort_limit`.

### 3. `dc-motor-envelope` — the four-quadrant torque–speed envelope

The static `velocity-limit-curve-*.png` shows the envelope; animate the **operating point**
living in it. Plane: velocity (x) vs torque (y). Draw the envelope from the three parameters:
`saturation_effort` (stall torque at q̇ = 0), `actuator_effort_limit` (flat cap), and
`actuator_velocity_limit` (no-load speed where available torque reaches zero). During a
simulated swing, show the *demanded* PD torque as a ghost point and the *applied* torque as its
projection onto the envelope — clipping is a projection, and the available torque shrinks as
speed rises. Cover at least two quadrants (motoring and braking).

Sources: `DCMotor.compute` / `DCMotor._clip_effort` in
`source/isaaclab/isaaclab/actuators/actuator_pd.py` — **derive the exact envelope shape from
this code**, not from memory; the corner behavior (how the linear speed-dependent limit
intersects the flat effort cap, and the four-quadrant symmetry) must match.

### 4. `effort-limit-damping-loss` — why saturation kills damping

Companion to the existing `effort-limit-clip.webp` swing-up. Show the PD equation as three
signed stacked bars (kp term, kd term, τ_ff) summing to the demanded torque, next to a
horizontal ±τ_max band. When the stack exceeds the band, the *whole sum* is clipped — visually,
the damping contribution is eaten by saturation. Sync with a small pendulum inset that starts
oscillating exactly when the kd term is being lost, and settles once the demand re-enters the
band. This animates the doc paragraph: "When the PD demand exceeds the limit, the applied torque
and damping term are both clipped… An effort limit below the load's static demand prevents the
controller from damping the joint effectively."

Sources: `actuators.rst` "Effort limit" section; `_clip_effort` in
`source/isaaclab/isaaclab/actuators/actuator_base.py`. Use the doc's numbers: ~2.94 N·m
gravity-hold torque, limits swept over [1, 2, 3, 4, 6] N·m if you show specific values.

### 5. `delay-buffer` — the DelayedPDActuator circular buffer

Left: a circular/FIFO buffer of N slots; commands (colored tokens) push in each physics step and
pop out `delay` steps later. Right: command-vs-response timeline for a square-wave position
target, response trailing by the delay. Then show a **reset**: the delay is re-sampled uniformly
from [`min_delay`, `max_delay`] and stays fixed until the next reset (annotate this — it is
per-reset randomization, not per-step noise). Optional end note from the doc: under native
execution the delay is fixed at `max_delay` (`min_delay` ignored).

Sources: `DelayedPDActuator` in `source/isaaclab/isaaclab/actuators/actuator_pd.py`,
`DelayBuffer` in `source/isaaclab/isaaclab/utils/buffers/`; `actuators.rst` "Command delay"
section and the native-execution note near the end of the page.

## Deliverable 0: the design charter (do this FIRST)

Before animating, produce a short design charter (`animation-design-charter.md`) and build the
five scenes against it. It will be published alongside the docs (likely in the contributing
guide) so that every future animation is consistent with these and with the existing figures.
It must pin down, concretely (hex values, point sizes, seconds):

- **Palette**: background, primary/secondary accents, per-role colors (command flow, torque,
  velocity, limits/clipping, telemetry), for both light and dark variants. Derive from the
  existing `*-curve-{light,dark}.png` figures so static plots and animations read as one family.
- **Typography**: one font for labels, one for math; minimum on-screen size at 960px width.
- **Layout**: margins, title placement, where equations live vs. where the "world" (pendulum,
  buffer, envelope) lives; light/dark structural parity (same layout, only colors change).
- **Motion conventions**: standard durations (intro, emphasis pulse, loop length target
  6–12 s), easing, how clipping/saturation is always depicted (same visual metaphor everywhere),
  how "inside the solver" vs "on the host" regions are drawn (this must match animation 1 and be
  reused by all others).
- **Legibility rules**: max simultaneous moving elements, label-every-axis, units in brackets
  per the docs convention (e.g. `[N·m]`, `[rad/s]`).

The charter is a deliverable of equal weight to the animations: hand it back even if some
scenes are unfinished.

## Naming and delivery

- File names: `pipeline-flow-{light,dark}.webp`, `implicit-vs-explicit-stability-{light,dark}.webp`,
  `dc-motor-envelope-{light,dark}.webp`, `effort-limit-damping-loss-{light,dark}.webp`,
  `delay-buffer-{light,dark}.webp`.
- Deliver: the design charter, the rendered webp files, the Manim scene sources, and a one-line
  render command per scene (so they can be regenerated), plus a short note per animation listing
  which source files you verified the behavior against and anything where the doc and code
  disagreed.
- Do not edit `actuators.rst` yet; the figure integration happens with the hosting decision.

## Verification checklist before handing back

- [ ] Design charter delivered; all five scenes conform to it.
- [ ] Every equation on screen matches the implementation (files listed above).
- [ ] DC-motor envelope derived from `DCMotor` code, including corner/quadrant behavior.
- [ ] Light/dark variants for all five; palette consistent with existing `*-dark.png` figures.
- [ ] Each webp ≤ ~600 KB, loops cleanly, readable at 960px and at 50% zoom.
- [ ] No repo modifications: no new dependencies, no committed artifacts.
