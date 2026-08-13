"""
Stage 2: 140 GHz Air-to-Ground Channel Model for Aerial-REAP-6G
===================================================================

Computes RSS(t) for each UAV in a swarm, coupled to its ACTUAL 3D trajectory
and attitude from Stage 1 (uav_dynamics.py / swarm_scenario.py) -- not a
synthetic/independent RSS signal. This is what makes ΔRSS in the blueprint's
sense meaningful: it responds to real geometry, real UAV banking, and a real
antenna beam misalignment, not scripted noise.

PHYSICS COMPONENTS (each cited to a real source, not guessed)
----------------------------------------------------------------
1. Free-space path loss (FSPL) -- Friis' law, close-in (CI) reference-distance
   form. Measurements across 28-140 GHz show path loss exponents are similar
   across this range in LOS conditions when using a 1m free-space reference
   distance (Ma et al., "Millimeter Wave and sub-THz Indoor Radio Propagation
   Channel Measurements... in an Office Environment," arXiv:2103.00385).
   For AERIAL (not indoor) links specifically, measured path loss exponents
   run higher than terrestrial 3GPP values -- e.g. n=2.25 for a UAV-to-UAV
   60GHz link vs 2.1 for 3GPP UMi LOS (Cuvelier et al., "An Experimental
   mmWave Channel Model for UAV-to-UAV Communications," arXiv:2007.11869) --
   confirming aerial links need their own exponent, not a reused terrestrial
   one. We use n=2.2 as a defensible middle value for an A2G (not A2A) link.

2. Molecular absorption -- real, frequency-dependent loss at sub-THz that is
   negligible at sub-6GHz. At 140 GHz, near-surface absorption is ~1.04 dB/km
   under standard atmosphere, decaying roughly exponentially with altitude
   (Zhang et al., "Terahertz Channel performance in ULEO Satellite-to-Ground
   Communications," arXiv:2509.12769). Our UAVs fly at 15-25m -- essentially
   surface altitude for this purpose -- so we treat the coefficient as fixed
   at the surface value rather than implementing the full ITU-R P.676
   line-by-line model (a defensible simplification given the flight
   envelope, documented here rather than silently assumed).

3. LoS probability -- standard sigmoid elevation-angle model:
       P_LoS(theta) = 1 / (1 + a*exp(-b*(theta - a)))
   (Al-Hourani-family model, as used e.g. in "Spatiotemporal Continual
   Learning for Mobile Edge UAV Networks," arXiv:2601.21861). Average path
   loss is the LoS/NLoS-probability-weighted mix.

4. Rician small-scale fading with an ELEVATION-ANGLE-DEPENDENT K-factor --
   measurements show K-factor rises with elevation angle (less blockage,
   stronger dominant LoS path), with typical values ~3.5-10 dB rising toward
   ~34 dB in clean high-elevation LOS ("A Survey of Path Loss Prediction and
   Channel Models for Unmanned Aerial Systems," MDPI Drones, 2023). We
   interpolate K linearly in dB between a low-elevation and high-elevation
   value from this measured range.

5. Antenna beam misalignment -- THE mechanism coupling kinematics to RF.
   The UAV carries a body-mounted directional antenna; its WORLD-FRAME
   pointing direction depends on the UAV's actual attitude quaternion (from
   Stage 1). A sudden bank/roll rotates the antenna away from the base
   station even if the commanded beam index (relative to the body) hasn't
   changed -- this is the physical mechanism behind "LoS threatened by a
   violent maneuver" in the blueprint. Gain falls off with a standard
   cosine^n directional pattern.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation


C = 299_792_458.0  # m/s, speed of light
BOLTZMANN = 1.380649e-23  # J/K


@dataclass
class ChannelParams:
    freq_hz: float = 140e9              # 140 GHz carrier
    tx_power_dbm: float = 20.0          # UAV transmit power (typical small-cell-class)
    noise_figure_db: float = 8.0        # receiver noise figure
    bandwidth_hz: float = 400e6         # channel bandwidth (typical sub-THz numerology)
    temperature_k: float = 290.0        # standard noise temperature

    # Path loss exponent for aerial A2G links -- see module docstring (2.2,
    # between the 2.1 terrestrial UMi LOS value and the 2.25 measured
    # UAV-to-UAV aerial value).
    path_loss_exponent: float = 2.2
    reference_distance_m: float = 1.0   # close-in (CI) model reference distance

    # Molecular absorption, 140 GHz, near-surface, standard atmosphere.
    # See module docstring point 2.
    molecular_absorption_db_per_km: float = 1.04

    # LoS probability sigmoid params (Al-Hourani-family, suburban-like default;
    # a,b are environment-dependent constants from the cited literature family).
    los_sigmoid_a: float = 4.88
    los_sigmoid_b: float = 0.43

    # Rician K-factor (dB) interpolation endpoints vs elevation angle,
    # from the measured 3.5-10 dB (low elevation) to ~34 dB (high elevation,
    # clean LOS) range cited in the module docstring.
    k_factor_db_low_elev: float = 4.0
    k_factor_db_high_elev: float = 30.0
    k_factor_low_elev_deg: float = 10.0
    k_factor_high_elev_deg: float = 80.0

    # Antenna directional gain pattern: gain(theta) = G0 * cos(theta)^n,
    # theta = angle between antenna boresight and true direction to BS.
    antenna_peak_gain_dbi: float = 24.0   # realistic for a small phased array at 140GHz
    antenna_pattern_exponent: float = 8.0  # controls beamwidth; higher = narrower beam


@dataclass
class BaseStation:
    position: np.ndarray  # (3,) world frame, meters


def _db(x_linear: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x_linear, 1e-30))


def _fading_db_clamped(fading_power_linear: float, floor_db: float = -30.0) -> float:
    """
    Converts Rician fading power to dB with a physically motivated floor.
    Deep destructive-interference fades (real+imag components both near
    zero) are a genuine Rician phenomenon, but an unbounded dB conversion
    can produce numerically absurd single-step drops (a -300+ dB spike was
    observed during validation from near-zero fading power hitting the
    generic _db() 1e-30 floor). Real channels don't actually go to zero
    power in a single fade -- diffuse multipath, receiver noise floor, and
    imperfect nulling all bound how deep an instantaneous fade can be. A
    -30 dB floor (1000x power reduction) is a standard practical bound for
    a single Rician fade realization in link-budget literature.
    """
    return max(_db(np.array([fading_power_linear]))[0], floor_db)


def _from_db(x_db: np.ndarray) -> np.ndarray:
    return 10.0 ** (x_db / 10.0)


class ChannelModel:
    def __init__(self, params: ChannelParams, base_station: BaseStation):
        self.p = params
        self.bs = base_station

    # -- geometry --------------------------------------------------------
    def _geometry(self, uav_pos: np.ndarray):
        """Returns (distance_m, elevation_angle_rad, direction_to_bs_unit_world).
        Elevation is measured as seen from the ground station looking up at
        the UAV (standard convention in the cited A2G literature): positive
        when the UAV is above the BS."""
        vec = self.bs.position - uav_pos
        distance = np.linalg.norm(vec)
        horiz = np.linalg.norm(vec[:2])
        elevation = np.arctan2(uav_pos[2] - self.bs.position[2], horiz)
        direction_unit = vec / (distance + 1e-9)
        return distance, elevation, direction_unit

    # -- large-scale path loss -------------------------------------------
    def _path_loss_db(self, distance_m: float) -> float:
        """Close-in free-space path loss at reference distance + log-distance term."""
        d0 = self.p.reference_distance_m
        fspl_d0 = 20 * np.log10(4 * np.pi * d0 * self.p.freq_hz / C)
        pl = fspl_d0 + 10 * self.p.path_loss_exponent * np.log10(distance_m / d0)
        return pl

    def _molecular_absorption_db(self, distance_m: float) -> float:
        return self.p.molecular_absorption_db_per_km * (distance_m / 1000.0)

    def _los_probability(self, elevation_rad: float) -> float:
        theta_deg = np.degrees(elevation_rad)
        a, b = self.p.los_sigmoid_a, self.p.los_sigmoid_b
        return 1.0 / (1.0 + a * np.exp(-b * (theta_deg - a)))

    def _k_factor_db(self, elevation_rad: float) -> float:
        theta_deg = np.clip(
            np.degrees(elevation_rad), self.p.k_factor_low_elev_deg, self.p.k_factor_high_elev_deg
        )
        frac = (theta_deg - self.p.k_factor_low_elev_deg) / (
            self.p.k_factor_high_elev_deg - self.p.k_factor_low_elev_deg
        )
        return self.p.k_factor_db_low_elev + frac * (
            self.p.k_factor_db_high_elev - self.p.k_factor_db_low_elev
        )

    # -- antenna beam misalignment (the kinematics <-> RF coupling) -----
    def _antenna_gain_db(self, uav_quat_xyzw: np.ndarray, direction_to_bs_world: np.ndarray) -> float:
        """
        Antenna boresight is body-frame -Z (ground-facing -- NOT the same as
        the +Z thrust axis in uav_dynamics.py, which points "up" through the
        rotor plane). A body+Z-mounted antenna would always point at the sky
        during normal level flight; -Z is the physically sensible mount for
        a UAV communicating with a ground base station. Found and corrected
        during validation: with +Z, antenna gain was pinned at the -60dB
        "no gain" floor for the entire flight regardless of geometry.

        Rotated into world frame by the UAV's ACTUAL current attitude. This
        is the direct mechanism by which a sudden Δq (violent maneuver) can
        degrade RSS even with zero change in the commanded beam index -- the
        antenna physically points away from the BS as the airframe rotates.
        """
        R = Rotation.from_quat(uav_quat_xyzw).as_matrix()
        boresight_world = R @ np.array([0.0, 0.0, -1.0])
        cos_theta = np.clip(np.dot(boresight_world, direction_to_bs_world), -1.0, 1.0)

        # Continuous pattern, no discrete branch at theta=90deg. Floor
        # cos_theta at 0 (rather than branching to a fixed -60dB the instant
        # theta crosses 90deg) so the gain curve approaches its floor
        # smoothly as the antenna sweeps past broadside -- avoids the
        # discontinuous jump found during validation (a single 2ms step
        # producing a 24dB antenna-gain jump purely from crossing the
        # theta>=90deg branch, with no corresponding physical event).
        gain_linear_frac = max(np.clip(cos_theta, 0.0, 1.0) ** self.p.antenna_pattern_exponent, 1e-6)
        return self.p.antenna_peak_gain_dbi + _db(np.array([gain_linear_frac]))[0]

    # -- full RSS computation for one timestep ----------------------------
    def compute_rss_dbm(
        self,
        uav_pos: np.ndarray,
        uav_quat_xyzw: np.ndarray,
        rng: np.random.Generator,
    ) -> dict:
        distance, elevation, direction_to_bs = self._geometry(uav_pos)

        pl_db = self._path_loss_db(distance)
        abs_db = self._molecular_absorption_db(distance)
        antenna_gain_db = self._antenna_gain_db(uav_quat_xyzw, direction_to_bs)

        p_los = self._los_probability(elevation)
        is_los = rng.random() < p_los

        k_db = self._k_factor_db(elevation) if is_los else -10.0  # heavily Rayleigh-like if NLoS
        k_linear = _from_db(np.array([k_db]))[0]
        # Rician fading amplitude: sum of a deterministic LOS component and a
        # complex Gaussian scattered component, normalized to unit mean power.
        los_amp = np.sqrt(k_linear / (k_linear + 1))
        scatter_std = np.sqrt(1.0 / (2 * (k_linear + 1)))
        real = los_amp + rng.normal(0, scatter_std)
        imag = rng.normal(0, scatter_std)
        fading_power_linear = real**2 + imag**2
        fading_db = _fading_db_clamped(fading_power_linear)

        nlos_extra_loss_db = 0.0 if is_los else rng.uniform(15.0, 30.0)  # typical NLoS excess loss

        rss_dbm = (
            self.p.tx_power_dbm
            + antenna_gain_db
            - pl_db
            - abs_db
            - nlos_extra_loss_db
            + fading_db
        )

        return {
            "rss_dbm": rss_dbm,
            "distance_m": distance,
            "elevation_deg": np.degrees(elevation),
            "path_loss_db": pl_db,
            "absorption_db": abs_db,
            "antenna_gain_db": antenna_gain_db,
            "is_los": is_los,
            "k_factor_db": k_db,
            "fading_db": fading_db,
        }

    def noise_floor_dbm(self) -> float:
        """Thermal noise floor, for downstream SNR/SE computation (not used
        directly by the LIF-SNN, but needed for oracle SE labels later)."""
        noise_w = BOLTZMANN * self.p.temperature_k * self.p.bandwidth_hz
        noise_dbm = 10 * np.log10(noise_w * 1000.0) + self.p.noise_figure_db
        return noise_dbm