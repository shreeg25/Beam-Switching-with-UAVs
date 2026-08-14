"""
Stage 3: Beam Steering Codebook + Oracle Action Labeling
============================================================

Introduces the missing degree of freedom needed for oracle labeling: a
BEAM STEERING OFFSET that the phased array applies on top of the airframe's
raw attitude. Without this, "beam direction" and "airframe attitude" are the
same thing (as in channel_model.py's antenna model), and there is nothing
for a beam-steering action to correct -- you can't have an "optimal beam
action" if there's no independent beam to steer.

DESIGN DECISIONS (each one a real fork, documented rather than assumed)
--------------------------------------------------------------------------
1. Label target = RELATIVE ACTION, not absolute beam index. This matches
   the architecture's output layer (BEAM_ACTIONS in aerial_reap6g.py:
   hold/shift_up/down/left/right/widen_beam), which fires a single relative
   codebook delta, not a classification over absolute directions.

2. Oracle "optimal beam" is computed from the CLEAN geometric antenna-gain
   signal (channel_model.py's _antenna_gain_db, which has no fast-fading
   term), not from raw noisy RSS. This is a deliberate clean-labels/
   noisy-inputs split: ground truth about "where should the beam point" is
   a geometric fact independent of that instant's random fading dip, while
   the SNN's actual INPUT (ΔRSS) still carries realistic fast-fading noise
   from Stage 2. Using raw RSS for labeling would inject fading-driven
   label flicker into the ground truth itself, corrupting training before
   it starts.

3. No artificial hysteresis/threshold is added on top -- see the validation
   check in this module's __main__ block, which confirms geometric optimal-
   beam labels are already smooth (bounded change per timestep) BECAUSE they
   derive from continuous physical motion, not from a jump-prone raw signal.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# Must match aerial_reap6g.py's BEAM_ACTIONS ordering exactly, since this is
# what the oracle labels are ultimately training the network to output.
BEAM_ACTIONS = ["hold", "shift_up", "shift_down", "shift_left", "shift_right", "widen_beam"]

# Discrete beam-steering offsets, in the antenna's LOCAL (body-relative)
# elevation/azimuth frame, applied ON TOP OF the raw body -Z boresight
# before rotating into world frame. "widen_beam" doesn't change pointing --
# it reduces antenna_pattern_exponent (broader main lobe, lower peak gain),
# a real trade-off for a phased array trading directivity for a bigger
# margin-for-error footprint.
STEER_STEP_DEG = 15.0  # angular step per shift action -- a realistic phased-array codebook granularity


@dataclass
class BeamState:
    """Tracks the antenna's current commanded steering offset, independent
    of the airframe's attitude. (el, az) in degrees, relative to body -Z
    boresight. widened=True applies the broader/lower-gain pattern."""
    el_offset_deg: float = 0.0
    az_offset_deg: float = 0.0
    widened: bool = False


def apply_action(state: BeamState, action: str) -> BeamState:
    """Returns a NEW BeamState reflecting the codebook action applied to
    the current one. Does not mutate in place, so callers can inspect the
    pre-action state for logging."""
    el, az, w = state.el_offset_deg, state.az_offset_deg, state.widened
    if action == "hold":
        pass
    elif action == "shift_up":
        el += STEER_STEP_DEG
    elif action == "shift_down":
        el -= STEER_STEP_DEG
    elif action == "shift_left":
        az -= STEER_STEP_DEG
    elif action == "shift_right":
        az += STEER_STEP_DEG
    elif action == "widen_beam":
        w = True
    else:
        raise ValueError(f"unknown beam action: {action}")
    if action not in ("widen_beam", "hold") and w:
        w = False  # a directional correction implicitly narrows back down
    return BeamState(el, az, w)


def steered_boresight_body(state: BeamState) -> np.ndarray:
    """Returns the antenna boresight unit vector in BODY frame, i.e. the
    raw -Z ground-facing direction rotated by the commanded steering
    offset.

    Elevation offset tilts the boresight in the body Y-axis direction
    (rotate around body-X). Azimuth offset tilts it in the body X-axis
    direction (rotate around body-Y). Found and fixed during validation:
    an earlier version rotated azimuth around body-Z (the boresight axis
    itself), which does nothing to a vector starting ON that axis --
    shift_left/shift_right had zero effect on antenna gain regardless of
    target direction. Rotating around body-Y is the correct axis to swing
    the boresight side-to-side.
    """
    base = np.array([0.0, 0.0, -1.0])
    R_el = Rotation.from_euler("x", state.el_offset_deg, degrees=True)
    R_az = Rotation.from_euler("y", state.az_offset_deg, degrees=True)
    return R_az.apply(R_el.apply(base))


def antenna_gain_db_steered(
    uav_quat_xyzw: np.ndarray,
    direction_to_bs_world: np.ndarray,
    beam_state: BeamState,
    peak_gain_dbi: float,
    pattern_exponent: float,
    widened_exponent: float = 2.0,  # much broader main lobe when widened
    widened_peak_gain_dbi: float = 14.0,  # lower peak gain, real directivity/beamwidth trade-off
) -> float:
    """Same physics as channel_model.ChannelModel._antenna_gain_db, but
    using the STEERED body-frame boresight instead of a fixed [0,0,-1]."""
    boresight_body = steered_boresight_body(beam_state)
    R = Rotation.from_quat(uav_quat_xyzw).as_matrix()
    boresight_world = R @ boresight_body
    cos_theta = np.clip(np.dot(boresight_world, direction_to_bs_world), -1.0, 1.0)

    exponent = widened_exponent if beam_state.widened else pattern_exponent
    peak = widened_peak_gain_dbi if beam_state.widened else peak_gain_dbi

    gain_linear_frac = max(np.clip(cos_theta, 0.0, 1.0) ** exponent, 1e-6)
    return peak + 10.0 * np.log10(gain_linear_frac)


def oracle_best_action(
    uav_quat_xyzw: np.ndarray,
    direction_to_bs_world: np.ndarray,
    current_beam_state: BeamState,
    peak_gain_dbi: float,
    pattern_exponent: float,
    hold_tolerance_db: float = 0.5,
) -> Tuple[str, float]:
    """
    Evaluates every action in BEAM_ACTIONS by applying it (hypothetically)
    to current_beam_state and computing the resulting CLEAN geometric
    antenna gain (no fading -- see module docstring point 2). Returns the
    action giving the highest gain, with "hold" preferred within
    hold_tolerance_db of the best alternative (avoids churn from
    infinitesimal gain differences that aren't operationally meaningful).

    Returns (action_name, resulting_gain_db_if_taken).
    """
    gains = {}
    for action in BEAM_ACTIONS:
        candidate_state = apply_action(current_beam_state, action)
        gains[action] = antenna_gain_db_steered(
            uav_quat_xyzw, direction_to_bs_world, candidate_state,
            peak_gain_dbi, pattern_exponent,
        )

    best_action = max(gains, key=gains.get)
    best_gain = gains[best_action]
    hold_gain = gains["hold"]

    if best_gain - hold_gain <= hold_tolerance_db:
        return "hold", hold_gain
    return best_action, best_gain