# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO101 keyboard letter-typing command."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import CommandTermCfg, EventTermCfg
from isaaclab.utils.configclass import configclass

from ...keyboards.keyboard_schema import KEY_ACTUATION_FRACTION
from .typing_commands import LetterTypingCommand


@configclass
class LetterTypingCommandCfg(CommandTermCfg):
    """Configuration for :class:`LetterTypingCommand`."""

    @configclass
    class ResetCfg:
        """All reset behavior for the typing task: the per-reset differential-IK arm snap (the ``ik*`` fields;
        runs on every reset whenever :attr:`ik` is set) plus an optional success-conditioned replay curriculum
        layered on top (:attr:`enabled` and the sampling/buffer fields).

        Named ``ResetCfg`` (not the env-level ``CurriculumCfg`` consumed by the curriculum manager). The replay
        curriculum and :attr:`pre_solve_reset` both require :attr:`ik`; without it there is no buffer and no
        pre-solve step. When :attr:`enabled`, the first reset builds a fixed buffer of reset snapshots by
        **oversampling** many candidate typing states, **grid-bucket-downsampling** them to uniform coverage over
        a weighted ``(target, next_key, unmatched, remaining, prefix_complete)`` feature (so no hand-tuned prefill
        probability), then running the reset-IK once per survivor. A per-snapshot monitor tracks recent success.
        Subsequent resets use the normal random reset or restore a buffered snapshot, biased toward states the
        policy half-solves.
        """

        # --- Reset pose: differential-IK snap of the arm above the first key (SO101-specific; set in the env
        # cfg). Runs on EVERY reset whenever `ik` is set - independent of the replay curriculum below (which
        # `enabled` gates); the curriculum's snapshots are built by running this same snap once per candidate.
        ik: DifferentialInverseKinematicsActionCfg | None = MISSING
        """Differential-IK setup that snaps the arm so the moving-jaw tip hovers above the first key, angled
        for a downward press, on every reset (no sim stepping; see
        :meth:`~...typing_commands.LetterTypingCommand._solve_reset_pose`). SO101-specific, so it is set in the
        env cfg rather than defaulted here; set to ``None`` to leave the arm wherever the other reset events
        place it. ``pose`` command type is required (the solve also drives the approach pitch); ``scale`` and
        ``clip`` are unused."""

        ik_rpy_deg: tuple[float, float, float] = MISSING
        """Approach orientation ``(roll, pitch, yaw)`` [deg] of the moving-jaw finger axis at reset.

        ``pitch`` tilts the finger axis below horizontal (``90`` points straight down, ``0`` is horizontal);
        ``yaw`` offsets the heading the arm settles into (``0`` keeps it); ``roll`` rotates the jaw about its
        approach (finger) axis. These describe the *desired* approach pose only: the 5-DoF arm solves
        orientation in the null space of the key-position task, so roll/pitch/yaw are tracked best-effort and
        never move the tip off the key."""

        ik_hover_height: float = MISSING
        """Height [m] above the target key at which the jaw tip is placed on reset."""

        ik_iters: int | tuple[int, int] = MISSING
        """Damped-least-squares iterations per reset solve (the lazy data layer refreshes FK and the Jacobian
        after each joint write, so no physics step is taken). The first ~60% of the iterations position the tip;
        the rest also drive the approach pitch.

        An ``int`` uses a fixed count for every env. A ``(min, max)`` tuple samples the count per env uniformly
        in ``[min, max]`` each solve and freezes the pose once an env hits its count, so snapshots land at varying
        convergence (different tip-to-key distances) - a reach-difficulty gradient for the reset curriculum.
        ``max`` still sets the position/pitch phase split, so envs stopping before ~60% of ``max`` are
        position-only (un-pitched)."""

        ik_seed_joint_noise: float = 0.0
        """Uniform joint offset [rad] applied to the default pose before each reset-IK solve: the arm is seeded
        at ``default + U(-noise, noise)`` (clamped to the soft joint limits, zero velocity), mirroring the
        ``reset_joints_by_offset`` event this replaces. It gives the reset start-states the same joint diversity
        as the no-IK case - most visible on low :attr:`ik_iters` snapshots, since a fully-converged solve reaches
        the same hover pose regardless of the seed. ``0.0`` seeds the exact default (deterministic convergence)."""

        match_prob: float = 0.5
        """Probability that each sampled pre-typed key matches the target (a correct prefix) vs a random key,
        when generating the candidate pool. Only needs to be large enough to *populate* the correct-prefix /
        overshoot cells - it does NOT set the final mix (coverage does), so the default is fine for short words;
        raise it to populate deeper correct prefixes for longer words."""

        oversample_factor: int = 64
        """Candidate pool size as a multiple of ``buffer_size`` (``M = oversample_factor * buffer_size``) for
        the coverage sampler. Bigger fills more feature cells (less random fill) at negligible cost (integer
        draws); diminishing once every reachable cell is populated."""

        # Feature weights for the coverage downsampling. Every axis is normalized to ~[0, 1] first, so each
        # weight is that axis's grid resolution (buckets along it scale with the weight) - a *direct* priority,
        # not swamped by the raw key-index range as it would be on unnormalized features. The final
        # forward:backspace mix is set by the RATIO of these: raise ``w_next_key`` (or lower ``w_unmatched``)
        # for less backspace + more next-key diversity; raise ``w_unmatched`` for more backspace-depth variety;
        # raise ``w_remaining`` to spread the number of keys still to type (how far from success).
        w_target: float = 1.0
        """Feature weight on the target word, each slot normalized by ``num_keys`` (word/key diversity)."""

        w_next_key: float = 1.0
        """Feature weight on the next key to press (normalized by ``num_keys``; spreads forward states,
        higher -> less backspace)."""

        w_unmatched: float = 1.0
        """Feature weight on the length-normalized backspace depth ``(typed_len - prefix_len) / target_len``
        (spreads backspace states; higher -> more backspace share and more depth variety). Normalizing by length
        makes a fully-wrong short word cost the same as a fully-wrong long one, removing the structural length
        bias so only the natural word-diversity bias remains."""

        w_prefix_complete: float = 1.0
        """Feature weight on the ``prefix == target_len`` flag, so "overshoot -> backspace to success" states
        get their own cells rather than collapsing into ordinary wrong-key backspace states."""

        w_remaining: float = 1.0
        """Feature weight on the *absolute* keys-still-to-type ``(target_len - prefix_len) / max_len`` - the
        forward half of the edit distance to success (the backspace half is ``w_unmatched``). Normalized by the
        fixed ``max_len`` (not ``target_len``), so difficulty is absolute: a len-3 word with 1 key left maps to
        the same value as a len-1 word from scratch. Without this axis, "2/3 already typed" and "empty, type
        all 3" collapse into the same cells and the easy ones dominate; raise it to spread how-far-from-success
        so the buffer (and the curriculum frontier) is not biased toward near-done states."""

        enabled: bool = False
        """Enable the success-conditioned replay curriculum (needs :attr:`ik`; a silent no-op without it). Does
        NOT gate the reset IK snap itself - that runs on every reset whenever :attr:`ik` is set."""

        buffer_size: int = 4096
        """Number of reset snapshots to cache (also the number of slots the success monitor tracks). Built in
        ``ceil(size / num_envs)`` reset-IK batches on the first reset, so keep it a small multiple of
        ``num_envs`` to bound the one-time build cost."""

        beta_target: float = 0.5
        """Target success rate the replay sampler is peaked at. A snapshot's sampling score is the Beta kernel
        ``rate**(a-1) * (1-rate)**(b-1)`` with ``a = 1 + kappa*target`` and ``b = 1 + kappa*(1-target)`` (mode
        at ``target``), so snapshots near this rate are replayed most and mastered/unsolved ones fade."""

        beta_kappa: float = 1.0
        """Concentration of the Beta sampling kernel (see :attr:`beta_target`); larger values peak the replay
        more tightly around the target success rate."""

        normal_weight: float = 0.5
        """Sampling weight of the normal (random) reset path, competing against the **mean** buffered Beta score
        (the weight is internally scaled by ``buffer_size``), so ``p(normal) = normal_weight / (normal_weight +
        mean_score)`` - **independent of** ``buffer_size``. At init every snapshot sits at the target, so
        ``mean_score = peak`` (``0.5`` for ``beta_target=0.5, beta_kappa=1``) and
        ``p(normal)_init = normal_weight / (normal_weight + peak)`` (e.g. ``0.1`` -> ~17%). As snapshots are
        mastered or stay unsolved their scores vanish (``mean_score -> 0``), so the normal path takes over."""

        sample_eps: float = 1e-4
        """Floor applied to every sampling weight (normal path + snapshots) so the replay distribution can
        never sum to 0 - which otherwise makes ``torch.multinomial`` assert ``sum of probabilities <= 0``.
        That degenerate case arises when ``normal_weight == 0`` and every snapshot's Beta score has collapsed
        to 0 (all rates at 0 or 1); the floor then falls back to ~uniform replay. Clamped to be strictly
        positive; raise it for a stronger uniform-exploration floor over the buffer."""

        history_len: int = 20
        """Sliding-window length of the per-snapshot success monitor."""

        reset_assets: tuple[str, ...] | None = None
        """Scene entity names whose physical state (root + joints) is snapshotted and restored for the replay.
        Defaults (``None``) to ``(asset_name, object_name)`` - i.e. the robot and the keyboard."""

        pre_solve_reset: EventTermCfg | None = None
        """Event term applied to the resetting envs right before the reset-IK solve (and once per build
        batch) - typically randomizing the keyboard root pose so the arm is posed to *this* reset's keyboard.
        Running it inside the solve is what gives the snapshot buffer diverse keyboard poses (each build batch
        re-randomizes), while buffer-restored envs never re-solve and so keep their snapshot's keyboard. Called
        directly as ``func(env, env_ids, **params)`` (its ``mode`` is unused). ``None`` skips it - then place
        any keyboard randomization back in the normal reset events. Only applies when :attr:`ik` is set."""

    class_type: type = LetterTypingCommand

    asset_name: str = MISSING
    """Robot asset name. Used by the reset-pose IK solve (see ``reset_above_key``)."""

    object_name: str = MISSING
    """Keyboard articulation asset name whose ``key_*_joint`` joints are read."""

    letter_length: tuple[int, int] = (1, 5)
    """``(min, max)`` target sequence length sampled per episode."""

    max_len: int | None = None
    """Fixed width [key slots] of the target/typed buffers and the one-hot observations. ``None`` uses
    ``letter_length[1]``. Set it explicitly (must be ``>= letter_length[1]``) to keep the observation size -
    and hence a trained policy's input layer - fixed while varying ``letter_length``. For example, evaluate
    a policy trained with ``letter_length=(1, 2)`` on single letters by keeping ``max_len=2`` and setting
    ``letter_length=(1, 1)``."""

    command_mode: str = "letter_full"
    """``letter_full`` -> ``[target, typed]`` (absolute); ``letter_left`` -> remaining target only."""

    actuation_fraction: float = KEY_ACTUATION_FRACTION
    """Fraction of a key's travel it must be depressed to register a keystroke (see
    :data:`~...keyboards.keyboard_schema.KEY_ACTUATION_FRACTION` for the shared convention)."""

    typeable_slots: tuple[int, ...] = MISSING
    """Global key-slot ids eligible as typing targets (resolved once at the config layer)."""

    backspace_slot: int = MISSING
    """Global key-slot id whose press deletes the last typed key."""

    reset: ResetCfg = ResetCfg()
    """All reset behavior: the per-reset IK arm-snap (:attr:`ResetCfg.ik` and the other ``ik*`` fields) plus the
    optional success-conditioned replay curriculum layered on top (see :class:`ResetCfg`; curriculum disabled by
    default)."""

    # --- Debug visualization (LED dot-matrix command banner) --------------------------------------
    slot_labels: tuple[str, ...] = ()
    """Per global key-slot label character used to draw letters when ``debug_vis`` is enabled (empty
    string for slots without a drawable glyph). Resolved once at the config layer; required only for
    the visualization."""

    visualizer_prim_path: str = "/Visuals/TypingCommand"
    """Prim path for the typing command's debug-visualization markers."""

    viz_pixel_size: float = 0.0045
    """Pixel pitch [m] of the LED dot-matrix letters (each dot is drawn slightly smaller)."""

    viz_letter_gap: float = 1.0
    """Gap between adjacent letters, in pixel-pitch units."""

    viz_row_gap: float = 0.02
    """Vertical gap [m] between the target row (top) and the typed row (bottom)."""

    viz_banner_height: float = 0.25
    """Height [m] of the letter banner above the keyboard's key centroid."""

    viz_key_marker_radius: float = 0.009
    """Radius [m] of the next-key halo marker."""

    viz_key_marker_height: float = 0.003
    """Height [m] (thickness) of the next-key halo marker."""

    viz_key_marker_lift: float = 0.003
    """Height [m] the next-key halo hovers above the key it marks. Kept small (well under the key
    pitch) so the halo sits on the keycap and does not appear shifted to a neighbor under perspective."""

    viz_env_ids: tuple[int, ...] | None = None
    """Environment indices whose debug-vis markers (letter banner + next-key halo) are drawn. ``None`` draws
    every environment; set a tuple to restrict the markers."""
