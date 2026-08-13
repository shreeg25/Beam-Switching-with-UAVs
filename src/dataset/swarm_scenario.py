"""
Swarm wrapper + scenario generation for the UAV kinematics stage.

REVISED DESIGN (v2): calm vs. aggressive regimes are now separated by
DISCRETE DISTURBANCE EVENTS (gust bursts, evasive reroutes), not by waypoint
geometry. Earlier iteration tried varying waypoint aggressiveness alone and
found the PD controller simply tracked both regimes equally well -- a
well-tuned controller absorbs gradual path differences without producing
distinct violent maneuvers. Discrete events instead inject a genuine,
time-localized physical disturbance, which is the actual claim the
architecture needs to detect and react to.

Both regimes share:
  - the SAME waypoint style (moderate, not scripted to be gentle or extreme)
  - the SAME background OU wind gust intensity
  - the SAME PD controller and gains

They differ ONLY in whether disturbance events are injected:
  - "calm":       no discrete events -- background gust + waypoint tracking only
  - "aggressive": 1-3 discrete events (gust bursts and/or evasive reroutes)
                  injected at random times during the flight
"""

from dataclasses import dataclass
from typing import List

import numpy as np
from uav_dynamics import Quadrotor, QuadrotorParams, GustParams, PDGains, DisturbanceEvent


@dataclass
class ScenarioResult:
    uavs: List[Quadrotor]
    dt: float
    duration: float
    regime: str


def _moderate_waypoints(center: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    """Single waypoint style shared by BOTH regimes -- moderate radius loop
    with natural heading variation. Neither scripted-gentle nor
    scripted-extreme; the regime distinction now lives entirely in whether
    disturbance events are injected, not in the flight plan."""
    radius = 10.0
    n_points = 7
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    angles += rng.normal(scale=0.15, size=n_points)
    pts = [
        center + np.array([radius * np.cos(a), radius * np.sin(a), 18.0 + rng.normal(scale=1.0)])
        for a in angles
    ]
    return pts


def _generate_events(
    duration: float, rng: np.random.Generator, n_events: int
) -> List[DisturbanceEvent]:
    events = []
    for _ in range(n_events):
        t_start = rng.uniform(2.0, duration - 3.0)  # keep clear of start/end transients
        kind = rng.choice(["gust_burst", "evasive_reroute"])
        if kind == "gust_burst":
            events.append(DisturbanceEvent(
                t_start=t_start,
                duration=rng.uniform(0.4, 0.6),
                kind="gust_burst",
                magnitude=rng.uniform(30.0, 45.0),  # N; calibrated during validation --
                    # below ~30N the gust force is absorbed too smoothly by the position
                    # loop to stand out from cruise-baseline omega noise.
            ))
        else:
            events.append(DisturbanceEvent(
                t_start=t_start,
                duration=rng.uniform(0.06, 0.12),  # snap-reaction timescale (~80ms), not a
                    # lingering detour -- tuned during validation: at ~1s duration the PD
                    # controller simply chases the injected target smoothly and produces
                    # only modest separation from cruise baseline; a short, sharp trigger
                    # (well under human/controller reaction time) is what actually produces
                    # a distinct high-omega spike, matching the "sudden emergency" framing.
                kind="evasive_reroute",
                magnitude=rng.uniform(3.0, 6.0),  # m; magnitude barely affects outcome at
                    # this duration since the drone can't travel far in <120ms -- kept from
                    # earlier calibration rather than re-derived, since it's not the driver.
            ))
    return events


def build_swarm_scenario(
    n_uavs: int,
    regime: str,
    duration: float = 20.0,
    dt: float = 0.002,  # 500 Hz -- matches the blueprint's sub-ms sampling intent
    base_center: np.ndarray = None,
    seed: int = 0,
) -> ScenarioResult:
    """
    Builds and simulates an N-UAV swarm for `duration` seconds under the
    given regime ("calm" or "aggressive"). See module docstring: regimes
    now differ ONLY in discrete disturbance-event injection.
    """
    if regime not in ("calm", "aggressive"):
        raise ValueError(f"unknown regime: {regime}")

    rng = np.random.default_rng(seed)
    base_center = base_center if base_center is not None else np.array([0.0, 0.0, 0.0])

    uavs = []
    for i in range(n_uavs):
        offset = np.array([10.0 * i, 5.0 * (i % 2), 0.0])
        start_pos = base_center + offset + np.array([0.0, 0.0, 15.0])
        waypoints = _moderate_waypoints(base_center + offset, rng)

        events = []
        if regime == "aggressive":
            n_events = rng.integers(1, 4)  # 1-3 discrete events per UAV
            events = _generate_events(duration, rng, n_events)

        uav = Quadrotor(
            uav_id=i,
            initial_position=start_pos,
            waypoints=waypoints,
            params=QuadrotorParams(),
            gust_params=GustParams(),  # identical background gust in both regimes
            gains=PDGains(),
            rng=np.random.default_rng(seed * 100 + i),
            events=events,
        )
        uavs.append(uav)

    n_steps = int(duration / dt)
    for _ in range(n_steps):
        for uav in uavs:
            uav.step(dt)

    return ScenarioResult(uavs=uavs, dt=dt, duration=duration, regime=regime)