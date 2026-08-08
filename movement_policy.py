#!/usr/bin/env python3
"""
SpiderBot 12-DOF movement / policy controller.

Simulation bridge:
    cargo run
The Rust app starts this file automatically with --bridge.

Raspberry Pi / Teensy HIL:
    python3 movement_policy.py --pi

Self test:
    python3 movement_policy.py --self-test

The controller deliberately separates:
    navigation -> forward/turn command -> locomotion -> 12 joint targets

Potential fields are used for high-level obstacle avoidance only.
The gait itself is a conservative one-leg-at-a-time crawl:
    FR -> FL -> BR -> BL -> repeat

This baseline gait is intentionally replaceable by a trained ML policy later.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(ROOT, "robot_config.json"), "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

CONTROL_HZ = float(CONFIG["control_hz"])
DT_DEFAULT = 1.0 / CONTROL_HZ

JOINT_ORDER = [
    "FL_hip_yaw", "FL_hip_pitch", "FL_knee",
    "FR_hip_yaw", "FR_hip_pitch", "FR_knee",
    "BL_hip_yaw", "BL_hip_pitch", "BL_knee",
    "BR_hip_yaw", "BR_hip_pitch", "BR_knee",
]
LEFT_JOINT_ORDER = [
    "FL_hip_yaw", "FL_hip_pitch", "FL_knee",
    "BL_hip_yaw", "BL_hip_pitch", "BL_knee",
]
RIGHT_JOINT_ORDER = [
    "FR_hip_yaw", "FR_hip_pitch", "FR_knee",
    "BR_hip_yaw", "BR_hip_pitch", "BR_knee",
]

# Exact requested crawl order.
LEG_ORDER = ["FR", "FL", "BR", "BL"]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def rad(degrees: float) -> float:
    return math.radians(degrees)


def normalize2(x: float, z: float) -> Tuple[float, float]:
    n = math.hypot(x, z)
    if n < 1e-9:
        return 0.0, 0.0
    return x / n, z / n


def joint_kind(name: str) -> str:
    if name.endswith("_hip_yaw"):
        return "hip_yaw"
    if name.endswith("_hip_pitch"):
        return "hip_pitch"
    return "knee"


def clamp_joint(name: str, target: float) -> float:
    lo, hi = CONFIG["joint_limits_deg"][joint_kind(name)]
    return clamp(target, rad(lo), rad(hi))


@dataclass
class Obstacle:
    # Robot-local coordinates. x<0 left, x>0 right, z>0 ahead.
    x: float
    z: float
    radius: float
    height: float = 0.15
    climbable: bool = False


class PotentialFieldNavigator:
    """
    High-level navigation only.

    Output convention:
        forward > 0 : forward
        forward < 0 : backward
        turn < 0    : LEFT
        turn > 0    : RIGHT

    This convention deliberately fixes the old left/right reversal.
    """

    def __init__(self) -> None:
        self.attractive_gain = 1.0
        self.repulsive_gain = 0.11
        self.influence_m = 0.70
        self.robot_radius_m = 0.14
        self.emergency_clearance_m = 0.055

        self.escape_timer = 0.0
        self.escape_turn = 1.0

    def find_climb_step(self, obstacles: List[Obstacle]) -> float:
        max_climb = float(CONFIG["gait"]["max_climb_height_m"])
        best = 0.0

        for obs in obstacles:
            if not obs.climbable:
                continue
            if obs.height > max_climb:
                continue
            if not (0.06 < obs.z < 0.48):
                continue
            if abs(obs.x) > obs.radius + 0.10:
                continue
            best = max(best, obs.height)

        return best

    def act(
        self,
        dt: float,
        goal_local: Tuple[float, float],
        obstacles: List[Obstacle],
    ) -> Tuple[float, float, float]:
        # Stop at the destination instead of orbiting/overshooting it.
        if math.hypot(*goal_local) < 0.12:
            return 0.0, 0.0, 0.0

        climb_height = self.find_climb_step(obstacles)

        gx, gz = normalize2(*goal_local)
        fx = gx * self.attractive_gain
        fz = gz * self.attractive_gain

        closest = 999.0
        left_pressure = 0.0
        right_pressure = 0.0

        for obs in obstacles:
            # A low box intentionally marked "climbable" is treated as terrain,
            # not something to route around.
            if (
                obs.climbable
                and obs.height <= float(CONFIG["gait"]["max_climb_height_m"])
                and 0.0 < obs.z < 0.55
                and abs(obs.x) < obs.radius + 0.12
            ):
                continue

            center = max(1e-4, math.hypot(obs.x, obs.z))
            clearance = center - obs.radius - self.robot_radius_m
            closest = min(closest, clearance)

            pressure = 1.0 / max(0.05, center)
            if obs.x < 0.0:
                left_pressure += pressure
            else:
                right_pressure += pressure

            if clearance >= self.influence_m:
                continue

            d = max(0.025, clearance)
            away_x = -obs.x / center
            away_z = -obs.z / center

            strength = (
                self.repulsive_gain
                * (1.0 / d - 1.0 / self.influence_m)
                / (d * d)
            )
            strength = min(strength, 4.0)

            fx += away_x * strength
            fz += away_z * strength

        if closest < self.emergency_clearance_m and self.escape_timer <= 0.0:
            self.escape_timer = 0.85
            # Choose the clearer side.
            self.escape_turn = -1.0 if left_pressure < right_pressure else 1.0

        if self.escape_timer > 0.0:
            self.escape_timer = max(0.0, self.escape_timer - dt)
            if self.escape_timer > 0.48:
                return -0.40, self.escape_turn * 0.30, 0.0
            return 0.0, self.escape_turn, 0.0

        dx, dz = normalize2(fx, fz)
        heading_error = math.atan2(dx, dz)

        # Negative x/error means turn left, positive means turn right.
        turn = clamp(heading_error / rad(34.0), -1.0, 1.0)

        forward = clamp(dz, 0.0, 1.0) * (1.0 - 0.58 * abs(turn))

        if climb_height > 0.0:
            # Slow down and use the high-clearance foot trajectory.
            forward = min(forward, 0.48)

        return forward, turn, climb_height


class BaselineLocomotionPolicy:
    """
    Slow conservative crawl intended for realistic servo testing.

    One leg is in swing while the other three remain in stance:
        FR -> FL -> BR -> BL

    `step_period_s = 1.0` means:
        1.0 s per swing leg
        4.0 s per full four-leg crawl cycle

    That is intentionally much slower than the earlier demo gait.
    """

    def __init__(self) -> None:
        gait = CONFIG["gait"]

        self.cycle_phase = 0.0
        self.step_period = float(gait["step_period_s"])
        self.yaw_sweep = rad(float(gait["yaw_sweep_deg"]))

        self.normal_lift_pitch = rad(float(gait["normal_lift_pitch_deg"]))
        self.normal_lift_knee = rad(float(gait["normal_lift_knee_deg"]))

        self.climb_lift_pitch = rad(float(gait["climb_lift_pitch_deg"]))
        self.climb_lift_knee = rad(float(gait["climb_lift_knee_deg"]))

    def stand_pose(self) -> Dict[str, float]:
        # Logical zero is the configured standing geometry in Rust.
        return {name: 0.0 for name in JOINT_ORDER}

    def phase_status(self) -> Tuple[str, str]:
        active_i = int(self.cycle_phase) % 4
        local = self.cycle_phase - math.floor(self.cycle_phase)

        if local < 0.25:
            phase = "LIFT"
        elif local < 0.50:
            phase = "SWING"
        elif local < 0.75:
            phase = "LOWER"
        else:
            phase = "SETTLE"

        return LEG_ORDER[active_i], phase

    def act(
        self,
        dt: float,
        forward_cmd: float,
        turn_cmd: float,
        climb_height: float,
        observation: dict,
    ) -> Dict[str, float]:
        del observation  # kept for future closed-loop / ML policy.

        if abs(forward_cmd) < 0.025 and abs(turn_cmd) < 0.025:
            return self.stand_pose()

        self.cycle_phase = (
            self.cycle_phase + max(0.0, dt) / self.step_period
        ) % 4.0

        targets = self.stand_pose()
        climb_mode = climb_height > 0.0

        lift_pitch = (
            self.climb_lift_pitch if climb_mode else self.normal_lift_pitch
        )
        lift_knee = (
            self.climb_lift_knee if climb_mode else self.normal_lift_knee
        )

        for index, leg in enumerate(LEG_ORDER):
            # Phase relative to this leg:
            # [0,1) swing, [1,4) stance.
            leg_phase = (self.cycle_phase - index) % 4.0

            side = -1.0 if leg.endswith("L") else 1.0

            # Correct differential-turn convention:
            # turn<0 (LEFT): right legs drive forward, left legs backward.
            # turn>0 (RIGHT): left legs drive forward, right legs backward.
            drive = clamp(
                forward_cmd - turn_cmd * side * 0.72,
                -1.0,
                1.0,
            )

            magnitude = abs(drive)
            if magnitude < 0.04:
                targets[f"{leg}_hip_yaw"] = 0.0
                continue

            travel_sign = 1.0 if drive >= 0.0 else -1.0
            sweep = self.yaw_sweep * max(0.35, magnitude)

            if leg_phase < 1.0:
                # Swing from rear to front.
                p = leg_phase

                # Fore/aft yaw.
                targets[f"{leg}_hip_yaw"] = (
                    (-1.0 + 2.0 * p) * travel_sign * sweep
                )

                # Smooth high-clearance foot arc.
                if p < 0.25:
                    t = p / 0.25
                    lift = t * t * (3.0 - 2.0 * t)
                elif p < 0.75:
                    lift = 1.0
                else:
                    t = (p - 0.75) / 0.25
                    t = t * t * (3.0 - 2.0 * t)
                    lift = 1.0 - t

                targets[f"{leg}_hip_pitch"] = lift_pitch * lift
                targets[f"{leg}_knee"] = lift_knee * lift

            else:
                # Three-legged stance phase: slowly sweep front -> rear,
                # producing propulsion while the fourth leg is repositioned.
                stance = (leg_phase - 1.0) / 3.0
                targets[f"{leg}_hip_yaw"] = (
                    (1.0 - 2.0 * stance) * travel_sign * sweep
                )

        return {
            name: clamp_joint(name, angle)
            for name, angle in targets.items()
        }


class SpiderBotController:
    def __init__(self) -> None:
        self.navigator = PotentialFieldNavigator()
        self.policy = BaselineLocomotionPolicy()

    def update(self, observation: dict) -> dict:
        # A simulation reset should restart both navigation recovery state and
        # the gait phase, instead of continuing halfway through an old step.
        if bool(observation.get("episode_reset", False)):
            self.navigator = PotentialFieldNavigator()
            self.policy = BaselineLocomotionPolicy()

        dt = float(observation.get("dt", DT_DEFAULT))

        control = observation.get("control", {})
        mode = str(control.get("mode", "automatic")).lower()

        nav = observation.get("navigation", {})
        goal = nav.get("goal_local", [0.0, 2.0])

        obstacles = [
            Obstacle(
                x=float(o["x"]),
                z=float(o["z"]),
                radius=float(o.get("radius", 0.10)),
                height=float(o.get("height", 0.15)),
                climbable=bool(o.get("climbable", False)),
            )
            for o in nav.get("obstacles", [])
        ]

        # Climb detection is useful in manual mode too.
        climb_height = self.navigator.find_climb_step(obstacles)

        if mode == "manual":
            forward_cmd = clamp(float(control.get("forward", 0.0)), -1.0, 1.0)
            turn_cmd = clamp(float(control.get("turn", 0.0)), -1.0, 1.0)
        else:
            forward_cmd, turn_cmd, climb_height = self.navigator.act(
                dt,
                (float(goal[0]), float(goal[1])),
                obstacles,
            )

        targets = self.policy.act(
            dt,
            forward_cmd,
            turn_cmd,
            climb_height,
            observation,
        )

        active_leg, gait_phase = self.policy.phase_status()

        return {
            "action_joint_targets_rad": [targets[name] for name in JOINT_ORDER],
            "joint_order": JOINT_ORDER,
            "forward_command": forward_cmd,
            "turn_command": turn_cmd,
            "active_leg": active_leg,
            "gait_phase": gait_phase,
            "climb_mode": climb_height > 0.0,
            "controller": "crawl12_potential_field_v2",
        }


# =====================================================================
# SIMULATION BRIDGE
# =====================================================================

def run_bridge() -> None:
    controller = SpiderBotController()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            observation = json.loads(line)
            result = controller.update(observation)
        except Exception as exc:
            result = {
                "action_joint_targets_rad": [0.0] * 12,
                "joint_order": JOINT_ORDER,
                "forward_command": 0.0,
                "turn_command": 0.0,
                "active_leg": "FR",
                "gait_phase": "STOP",
                "climb_mode": False,
                "controller": "ERROR",
                "error": str(exc),
            }

        print(json.dumps(result), flush=True)


# =====================================================================
# PI -> TWO TEENSY I2C ENCODING
# =====================================================================

def servo_angle_deg(joint: str, logical_target_rad: float) -> float:
    mapping = CONFIG["servo_mapping"][joint]
    angle = (
        float(mapping["center_deg"])
        + float(mapping["sign"]) * math.degrees(logical_target_rad)
    )
    return clamp(angle, 0.0, 180.0)


def encode_teensy_packet(
    joint_targets: Dict[str, float],
    names: List[str],
) -> bytes:
    payload = bytearray([int(CONFIG["i2c"]["command_set_angles"])])

    for name in names:
        centi_deg = int(round(servo_angle_deg(name, joint_targets[name]) * 100))
        centi_deg = max(0, min(18000, centi_deg))
        payload += centi_deg.to_bytes(2, byteorder="big", signed=False)

    payload.append(sum(payload) & 0xFF)

    if len(payload) != 14:
        raise RuntimeError(f"Expected 14-byte Teensy packet, got {len(payload)}")

    return bytes(payload)


def packets_for_targets(targets: Dict[str, float]) -> Tuple[bytes, bytes]:
    return (
        encode_teensy_packet(targets, LEFT_JOINT_ORDER),
        encode_teensy_packet(targets, RIGHT_JOINT_ORDER),
    )


def run_pi() -> None:
    """
    HIL skeleton.

    This intentionally does not pretend that the Pi already has final
    perception / IMU / encoders. Add those observations later and leave the
    12-joint output interface unchanged.
    """

    try:
        from smbus2 import SMBus, i2c_msg
    except ImportError as exc:
        raise RuntimeError("Install `smbus2` on the Raspberry Pi.") from exc

    bus = SMBus(1)
    left_addr = int(CONFIG["i2c"]["left_address"])
    right_addr = int(CONFIG["i2c"]["right_address"])

    controller = SpiderBotController()
    period = 1.0 / CONTROL_HZ
    last = time.monotonic()

    print(
        f"SpiderBot HIL: {CONTROL_HZ:.0f} Hz, "
        f"L=0x{left_addr:02X}, R=0x{right_addr:02X}"
    )

    try:
        while True:
            start = time.monotonic()
            dt = max(0.001, min(0.10, start - last))
            last = start

            # Replace these placeholders with real sensors/perception.
            observation = {
                "dt": dt,
                "joint_angles_rad": [0.0] * 12,
                "joint_velocities_rad_s": [0.0] * 12,
                "chassis_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                "chassis_linear_velocity_m_s": [0.0, 0.0, 0.0],
                "chassis_angular_velocity_rad_s": [0.0, 0.0, 0.0],
                "foot_contacts": [False] * 4,
                "control": {
                    "mode": "automatic",
                    "forward": 0.0,
                    "turn": 0.0,
                },
                "navigation": {
                    "goal_local": [0.0, 5.0],
                    "obstacles": [],
                },
            }

            result = controller.update(observation)
            targets = dict(zip(JOINT_ORDER, result["action_joint_targets_rad"]))
            packet_l, packet_r = packets_for_targets(targets)

            bus.i2c_rdwr(i2c_msg.write(left_addr, packet_l))
            bus.i2c_rdwr(i2c_msg.write(right_addr, packet_r))

            remaining = period - (time.monotonic() - start)
            if remaining > 0.0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass

    finally:
        neutral = {name: 0.0 for name in JOINT_ORDER}
        packet_l, packet_r = packets_for_targets(neutral)

        try:
            bus.i2c_rdwr(i2c_msg.write(left_addr, packet_l))
            bus.i2c_rdwr(i2c_msg.write(right_addr, packet_r))
        except Exception:
            pass

        bus.close()


def run_self_test() -> None:
    base = {
        "dt": 0.02,
        "joint_angles_rad": [0.0] * 12,
        "joint_velocities_rad_s": [0.0] * 12,
        "chassis_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "chassis_linear_velocity_m_s": [0.0, 0.0, 0.0],
        "chassis_angular_velocity_rad_s": [0.0, 0.0, 0.0],
        "foot_contacts": [True] * 4,
        "control": {"mode": "automatic", "forward": 0.0, "turn": 0.0},
        "navigation": {"goal_local": [0.0, 2.0], "obstacles": []},
    }

    result = SpiderBotController().update(base)
    assert len(result["action_joint_targets_rad"]) == 12

    # Manual LEFT must be negative turn; RIGHT must be positive turn.
    manual_left = dict(base)
    manual_left["control"] = {"mode": "manual", "forward": 0.0, "turn": -1.0}
    left_result = SpiderBotController().update(manual_left)
    assert left_result["turn_command"] < 0.0

    manual_right = dict(base)
    manual_right["control"] = {"mode": "manual", "forward": 0.0, "turn": 1.0}
    right_result = SpiderBotController().update(manual_right)
    assert right_result["turn_command"] > 0.0

    # Obstacle on left should push automatic navigator RIGHT.
    auto_left_obstacle = dict(base)
    auto_left_obstacle["navigation"] = {
        "goal_local": [0.0, 2.0],
        "obstacles": [{
            "x": -0.16, "z": 0.38, "radius": 0.08,
            "height": 0.15, "climbable": False,
        }],
    }
    avoid_right = SpiderBotController().update(auto_left_obstacle)
    assert avoid_right["turn_command"] > 0.0

    # Low climb box ahead should activate climb mode instead of avoidance.
    climb = dict(base)
    climb["navigation"] = {
        "goal_local": [0.0, 2.0],
        "obstacles": [{
            "x": 0.0, "z": 0.30, "radius": 0.10,
            "height": 0.06, "climbable": True,
        }],
    }
    climb_result = SpiderBotController().update(climb)
    assert climb_result["climb_mode"] is True

    targets = dict(zip(JOINT_ORDER, result["action_joint_targets_rad"]))
    packet_l, packet_r = packets_for_targets(targets)
    assert len(packet_l) == 14 and len(packet_r) == 14
    assert packet_l[-1] == sum(packet_l[:-1]) & 0xFF
    assert packet_r[-1] == sum(packet_r[:-1]) & 0xFF

    print("12 joint output: PASS")
    print("Manual LEFT/RIGHT sign: PASS")
    print("Potential-field avoidance: PASS")
    print("Low-step climb detection: PASS")
    print("14-byte Teensy packets: PASS")
    print(
        f"Gait pace: {CONFIG['gait']['step_period_s']:.2f}s/leg, "
        f"{CONFIG['gait']['step_period_s'] * 4:.2f}s/full crawl cycle"
    )

    # Analytic peak servo slew rate this gait demands, so a Pi build can be
    # checked against real hobby-servo limits before touching hardware.
    step_period = float(CONFIG["gait"]["step_period_s"])
    yaw_peak_deg_s = (2.0 * float(CONFIG["gait"]["yaw_sweep_deg"])) / step_period
    lift_peak_deg_s = abs(float(CONFIG["gait"]["normal_lift_pitch_deg"])) / (0.25 * step_period)
    peak_deg_s = max(yaw_peak_deg_s, lift_peak_deg_s)
    servo_limit = float(CONFIG["motors"]["servo_max_speed_deg_s"])

    status = "OK" if peak_deg_s <= servo_limit else "TOO FAST for real servo"
    print(
        f"Estimated peak servo speed: {peak_deg_s:.0f} deg/s "
        f"(Pi/servo limit {servo_limit:.0f} deg/s) -> {status}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--pi", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.bridge:
        run_bridge()
    elif args.pi:
        run_pi()
    elif args.self_test:
        run_self_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
