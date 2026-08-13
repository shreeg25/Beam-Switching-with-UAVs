"""
Stage 1: Physics-Informed UAV Kinematics for Aerial-REAP-6G Dataset
=====================================================================

Produces real 6DoF quadrotor flight trajectories for a small swarm, driven
by Newton-Euler rigid-body dynamics + a PD attitude/position controller +
stochastic wind gusts. This is the foundation the RF channel model (Stage 2)
will later be coupled to via actual 3D geometry -- not scripted independently.

PHYSICS REFERENCES
-------------------
- Rigid body translational/rotational dynamics: standard Newton-Euler
  formulation for quadrotors, e.g. Mellinger & Kumar (2011), "Minimum Snap
  Trajectory Generation and Control for Quadrotors," ICRA.
- Quaternion kinematics: dq/dt = 0.5 * q ⊗ [0, ω], integrated via
  scipy.spatial.transform.Rotation (avoids gimbal lock, matches the
  blueprint's Δq formulation directly).
- Wind gust model: Dryden-style continuous stochastic gust, implemented as
  an Ornstein-Uhlenbeck process per axis (mean-reverting colored noise) --
  standard substitute for full Dryden filters in control-systems literature
  when a full MIL-F-8785C implementation isn't required.

STATE REPRESENTATION (per UAV)
-------------------------------
  position       p   = (x, y, z)                  [m], world frame, z-up
  velocity       v   = (vx, vy, vz)                [m/s]
  orientation    q   = (w, x, y, z) quaternion     body -> world
  angular vel.   omega = (wx, wy, wz)              [rad/s], body frame

CONTROL
-------
A cascaded PD controller (position -> desired attitude -> attitude -> body
torques) drives each UAV through a sequence of waypoints. Aggressive
waypoint changes and gust disturbances are what PRODUCE violent maneuvers
here -- they are a controller RESPONSE to real inputs, not scripted spikes.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.integrate import solve_ivp


G = 9.81  # m/s^2


@dataclass
class QuadrotorParams:
    """Physical parameters, representative of a small commercial quadrotor
    (roughly DJI Matrice-class), not tuned per-UAV -- shared across swarm."""
    mass: float = 1.5          # kg
    arm_length: float = 0.25   # m
    Ixx: float = 0.0347        # kg*m^2 (roll inertia)
    Iyy: float = 0.0347        # kg*m^2 (pitch inertia)
    Izz: float = 0.0617        # kg*m^2 (yaw inertia)
    drag_coeff: float = 0.25   # linear translational drag coefficient
    max_thrust: float = 30.0   # N, total across 4 rotors
    max_torque: float = 3.75   # N*m, derived from per-rotor thrust (30N/4) * arm_length * 2
                                # (differential thrust authority across an opposing rotor pair)


@dataclass
class GustParams:
    """Ornstein-Uhlenbeck wind gust model per axis:
        dW = -theta*(W - mean)*dt + sigma*sqrt(dt)*N(0,1)
    theta controls mean-reversion speed (gust 'gustiness'), sigma controls
    intensity. Defaults represent moderate turbulence, not calm air --
    calm-vs-violent flight is later distinguished by WAYPOINT AGGRESSIVENESS,
    not by toggling gusts on/off (gusts are always-on background physics).
    """
    theta: float = 0.8
    sigma: float = 1.2         # m/s^2 equivalent force noise intensity
    mean: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class PDGains:
    # Position -> desired attitude
    kp_pos: float = 4.0
    kd_pos: float = 3.0
    max_accel_cmd: float = 6.0  # m/s^2 (~0.6g) -- caps commanded acceleration from
                                 # position error so a fresh, large waypoint-distance
                                 # error doesn't itself look like a violent maneuver;
                                 # real flight controllers always saturate this term.
                                 # Disturbance EVENTS (gust bursts, evasive reroutes)
                                 # are what should stand out against this baseline.
    # Attitude -> body torque
    # kp/kd derived so saturation (max_torque=3.75 N*m) occurs near a genuinely
    # aggressive ~35 deg bank, not a trivial few-degree error -- gives wn~2.1Hz,
    # zeta=0.75, a believable inner attitude-loop bandwidth for this airframe class.
    kp_att: float = 6.14
    kd_att: float = 0.69
    kp_yaw: float = 10.0
    kd_yaw: float = 4.0


@dataclass
class DisturbanceEvent:
    """A discrete, time-localized disturbance -- this is the actual source of
    'violent maneuver' data, decoupled from waypoint aggressiveness (see
    swarm_scenario.py docstring for rationale)."""
    t_start: float
    duration: float
    kind: str  # "gust_burst" | "evasive_reroute"
    magnitude: float  # gust: N of extra force; evasive: meters of injected offset
    direction: np.ndarray = None  # unit vector; random if None
    urgent_slew_rate: float = 25.0  # m/s -- evasive_reroute only: overrides the
        # normal cruise target_slew_rate for the event's duration. Physically
        # justified: a collision-avoidance trigger is exactly the case where a
        # real flight controller relaxes trajectory smoothing to react fast,
        # unlike ordinary waypoint-to-waypoint cruising which stays smoothed.


class UAVState:

    def __init__(self, position: np.ndarray, yaw0: float = 0.0):
        self.p = np.array(position, dtype=float)
        self.v = np.zeros(3)
        self.q = Rotation.from_euler("z", yaw0).as_quat()  # scipy order: (x,y,z,w)
        self.omega = np.zeros(3)
        self.gust = np.zeros(3)  # current OU gust force state


class Quadrotor:
    """
    Single UAV instance: rigid-body dynamics + waypoint-following PD control
    + wind gusts + discrete disturbance events. Call `.step(dt)` to advance
    one integration step.
    """

    def __init__(
        self,
        uav_id: int,
        initial_position: np.ndarray,
        waypoints: List[np.ndarray],
        params: Optional[QuadrotorParams] = None,
        gust_params: Optional[GustParams] = None,
        gains: Optional[PDGains] = None,
        waypoint_radius: float = 0.75,
        rng: Optional[np.random.Generator] = None,
        events: Optional[List[DisturbanceEvent]] = None,
    ):
        self.uav_id = uav_id
        self.params = params or QuadrotorParams()
        self.gust_params = gust_params or GustParams()
        self.gains = gains or PDGains()
        self.waypoints = waypoints
        self.waypoint_idx = 0
        self.waypoint_radius = waypoint_radius
        self.rng = rng or np.random.default_rng()
        self.events = sorted(events or [], key=lambda e: e.t_start)
        self._injected_waypoint = None  # active evasive-reroute target, if any
        self.target_slew_rate = 8.0     # m/s -- max speed the SMOOTHED setpoint may
                                          # move at; prevents discontinuous target jumps
        self._smoothed_target = np.array(waypoints[0], dtype=float)
        self._raw_target = np.array(waypoints[0], dtype=float)
        self._injected_slew_rate = self.target_slew_rate  # overridden during evasive events
        self._active_evasive_event = None  # tracks which event object set the injected target, to fix it once
        self._last_dt = 0.002  # updated each step(); default matches typical usage

        self.state = UAVState(initial_position)
        self.t = 0.0
        self._current_total_gust = np.zeros(3)

        # log for post-hoc inspection / dataset export
        self.history = {
            "t": [], "p": [], "v": [], "q": [], "omega": [],
            "thrust": [], "torque": [], "gust": [], "event_active": [],
        }

    # -- waypoint management -------------------------------------------------
    def _current_target(self) -> np.ndarray:
        """
        Returns the CURRENT smoothed target, not a raw waypoint. Raw waypoints
        are used only to update self._raw_target when the drone gets close
        enough to the current one; the actual setpoint fed to the position
        controller is exponentially slewed toward that raw target.

        This fixes a real bug found during validation: switching pos_err's
        target discretely (a hard jump to the next waypoint) caused a
        discontinuous one-timestep jump in commanded thrust direction, which
        the attitude loop chased as a large, fast torque transient -- this
        was the actual source of "violent"-looking omega spikes during
        ordinary waypoint transitions, unrelated to any real physical
        disturbance. Real flight controllers always reference a smoothed
        (e.g. minimum-jerk or slewed) trajectory, never a raw waypoint list,
        for exactly this reason.
        """
        if self._injected_waypoint is not None:
            self._raw_target = self._injected_waypoint
            slew_rate = self._injected_slew_rate
        else:
            candidate = self.waypoints[self.waypoint_idx % len(self.waypoints)]
            if np.linalg.norm(self.state.p - candidate) < self.waypoint_radius:
                self.waypoint_idx += 1
                candidate = self.waypoints[self.waypoint_idx % len(self.waypoints)]
            self._raw_target = candidate
            slew_rate = self.target_slew_rate

        # Exponential slew: smoothed_target moves toward raw_target at a rate
        # bounded by slew_rate (m/s), so pos_err's DIRECTION can't jump
        # discontinuously even when the raw waypoint itself does. Evasive
        # events use a faster urgent_slew_rate (set in _apply_events).
        direction = self._raw_target - self._smoothed_target
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            step = min(dist, slew_rate * self._last_dt)
            self._smoothed_target = self._smoothed_target + (direction / dist) * step
        return self._smoothed_target

    # -- discrete event handling -----------------------------------------------
    def _active_events(self):
        return [e for e in self.events if e.t_start <= self.t < e.t_start + e.duration]

    def _apply_events(self, active):
        """Returns extra_gust_force (added on top of background OU gust) and
        sets/clears self._injected_waypoint for evasive reroutes."""
        extra_gust = np.zeros(3)
        evasive = [e for e in active if e.kind == "evasive_reroute"]
        if evasive:
            e = evasive[0]
            # Fix the injected target ONCE at event onset, not every step --
            # recomputing "current position + offset" every step while the
            # event is active made the target recede as the drone approached
            # it (an un-catchable moving goalpost), which was a real bug
            # found during validation.
            if self._active_evasive_event is not e:
                direction = e.direction if e.direction is not None else _unit(self.rng.normal(size=3))
                self._injected_waypoint = self.state.p + direction * e.magnitude
                self._active_evasive_event = e
            self._injected_slew_rate = e.urgent_slew_rate
        else:
            self._injected_waypoint = None
            self._active_evasive_event = None

        for e in active:
            if e.kind == "gust_burst":
                direction = e.direction if e.direction is not None else _unit(self.rng.normal(size=3))
                extra_gust += direction * e.magnitude

        return extra_gust

    # -- gust update (Euler-Maruyama for the OU process) ----------------------
    def _update_gust(self, dt: float, extra_gust_force: np.ndarray):
        gp = self.gust_params
        noise = self.rng.normal(size=3)
        d_gust = -gp.theta * (self.state.gust - gp.mean) * dt + gp.sigma * np.sqrt(dt) * noise
        self.state.gust = self.state.gust + d_gust
        # extra_gust_force is a discrete-event impulse layered on top of the
        # always-on background OU process, not blended into its mean-reversion.
        self._current_total_gust = self.state.gust + extra_gust_force

    # -- PD control: position error -> desired thrust vector + yaw -----------
    def _position_control(self, target: np.ndarray):
        g = self.gains
        pos_err = target - self.state.p
        vel_err = -self.state.v  # desired velocity at waypoint = 0 (hover-to-point tracking)
        accel_cmd = g.kp_pos * pos_err + g.kd_pos * vel_err
        accel_horizontal_mag = np.linalg.norm(accel_cmd[:2])
        if accel_horizontal_mag > g.max_accel_cmd:
            # Cap horizontal commanded acceleration only -- prevents a large,
            # fresh waypoint-distance error from itself producing a violent
            # bank, without limiting vertical thrust authority.
            accel_cmd[:2] *= g.max_accel_cmd / accel_horizontal_mag
        accel_cmd[2] += G  # gravity feedforward on z
        thrust_vec = self.params.mass * accel_cmd
        return thrust_vec

    # -- attitude control: desired thrust direction -> body torques ----------
    def _attitude_control(self, thrust_vec: np.ndarray, yaw_target: float = 0.0):
        g = self.gains
        thrust_mag = np.linalg.norm(thrust_vec)
        thrust_mag = np.clip(thrust_mag, 0.1, self.params.max_thrust)

        z_body_des = thrust_vec / (np.linalg.norm(thrust_vec) + 1e-6)
        x_world_des = np.array([np.cos(yaw_target), np.sin(yaw_target), 0.0])
        y_body_des = np.cross(z_body_des, x_world_des)
        y_body_des /= (np.linalg.norm(y_body_des) + 1e-6)
        x_body_des = np.cross(y_body_des, z_body_des)
        R_des = np.stack([x_body_des, y_body_des, z_body_des], axis=1)

        R_cur = Rotation.from_quat(self.state.q).as_matrix()
        R_err = R_des.T @ R_cur - R_cur.T @ R_des
        att_err = 0.5 * np.array([R_err[2, 1], R_err[0, 2], R_err[1, 0]])

        torque = -g.kp_att * att_err - g.kd_att * self.state.omega
        torque = np.clip(torque, -self.params.max_torque, self.params.max_torque)
        return thrust_mag, torque

    # -- full rigid-body dynamics step ----------------------------------------
    def step(self, dt: float):
        self._last_dt = dt
        active = self._active_events()
        extra_gust = self._apply_events(active)
        self._update_gust(dt, extra_gust)

        target = self._current_target()
        thrust_vec = self._position_control(target)
        thrust_mag, torque = self._attitude_control(thrust_vec)

        R_cur = Rotation.from_quat(self.state.q).as_matrix()
        thrust_world = R_cur @ np.array([0, 0, thrust_mag])

        drag = -self.params.drag_coeff * self.state.v * np.linalg.norm(self.state.v)
        gravity = np.array([0, 0, -self.params.mass * G])
        gust_force = self._current_total_gust  # background OU + any active burst

        accel = (thrust_world + gravity + drag + gust_force) / self.params.mass
        self.state.v = self.state.v + accel * dt
        self.state.p = self.state.p + self.state.v * dt
        self.state.p[2] = max(self.state.p[2], 0.05)  # ground clamp

        I = np.array([self.params.Ixx, self.params.Iyy, self.params.Izz])
        omega_dot = (torque - np.cross(self.state.omega, I * self.state.omega)) / I
        self.state.omega = self.state.omega + omega_dot * dt

        omega_quat_form = np.concatenate([self.state.omega, [0]])  # scipy order (x,y,z,w)
        dq = 0.5 * _quat_multiply(self.state.q, omega_quat_form)
        q_new = self.state.q + dq * dt
        q_new = q_new / (np.linalg.norm(q_new) + 1e-9)
        self.state.q = q_new

        self.t += dt
        self.history["t"].append(self.t)
        self.history["p"].append(self.state.p.copy())
        self.history["v"].append(self.state.v.copy())
        self.history["q"].append(self.state.q.copy())
        self.history["omega"].append(self.state.omega.copy())
        self.history["thrust"].append(thrust_mag)
        self.history["torque"].append(torque.copy())
        self.history["gust"].append(self._current_total_gust.copy())
        self.history["event_active"].append(len(active) > 0)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _quat_multiply(q1, q2):
    """Hamilton product, scipy quaternion order (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])