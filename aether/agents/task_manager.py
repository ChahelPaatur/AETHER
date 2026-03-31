"""
TaskManagerAgent: top-level orchestrator. Coordinates all domain agents, drives the
main loop, detects mission-level failure, and maintains global mission state.

Agent topology (per-tick execution order):
  PerceptionAgent → [PowerAgent, ThermalAgent, NavigationAgent, CameraAgent]
  → FaultAgent (DRL-First Hybrid FDIR) → AdaptationAgent
  → PlannerAgent → MovementAgent
"""
import logging
from typing import Dict, List, Optional

import numpy as np

from ..core.goal_parser import GoalParser
from ..core.message_bus import MessageBus
from ..core.metrics import MetricsTracker
from ..capabilities.capability_loader import CapabilityLoader
from .planner_agent import PlannerAgent
from .perception_agent import PerceptionAgent
from .fault_agent import FaultAgent
from .adaptation_agent import AdaptationAgent
from .memory_agent import MemoryAgent
from .camera_agent import CameraAgent
from .movement_agent import MovementAgent
from .power_agent import PowerAgent
from .thermal_agent import ThermalAgent
from .navigation_agent import NavigationAgent

logger = logging.getLogger("AETHER.TaskManager")

# Map FaultInjector fault_type keys → FaultDetector fault_type strings
_INJECTOR_TO_DETECTOR = {
    "battery_drain":         "POWER_CRITICAL",
    "bus_voltage_drop":      "POWER_CRITICAL",
    "thermal_spike":         "THERMAL_ANOMALY",
    "imu_drift":             "IMU_DRIFT",
    "sensor_noise_imu":      "IMU_DRIFT",
    "intermittent":          "INTERMITTENT_FAULT",
    "sensor_dropout_camera": "SENSOR_FAILURE",
    "actuator_degrade":      "ACTUATOR_DEGRADATION",
}


class TaskManagerAgent:
    """
    Queen agent: parses goals, ticks all domain agents, manages mission state.
    """

    name = "task_manager"

    def __init__(self, env, config_path: str, bus: MessageBus,
                 log_dir: str = "logs", seed: int = 42,
                 fault_injector=None, verbose: bool = False,
                 no_learning: bool = False):
        self.env = env
        self.bus = bus
        self.verbose = verbose

        cap_loader = CapabilityLoader()
        self.cap_graph = cap_loader.load_from_file(config_path)
        self.available_actions = self.cap_graph.available_actions()

        self.goal_parser = GoalParser()
        self.metrics = MetricsTracker()

        # Build hardware adapter
        adapter = self._build_adapter(config_path)

        # Domain agents
        self.camera     = CameraAgent(bus)
        self.movement   = MovementAgent(adapter, bus)
        self.power      = PowerAgent(bus)
        self.thermal    = ThermalAgent(bus)
        self.navigation = NavigationAgent(bus)

        # Core agents
        self.planner    = PlannerAgent(bus)
        self.perception = PerceptionAgent(env, bus, fault_injector=fault_injector)
        self.fault_agent = FaultAgent(bus, seed=seed, no_learning=no_learning)
        self.adaptation = AdaptationAgent(bus)
        self.memory     = MemoryAgent(bus, session_log_dir=log_dir)

        self._step = 0
        self._goal: Optional[Dict] = None
        self._done = False
        self._success = False
        self._tracked_injected: set = set()

        bus.subscribe("FAULT_DETECTED", self._on_fault)

    def run_episode(self, task_text: str, scenario: Dict,
                    max_steps: int = 300, render: bool = False) -> Dict:
        """Run a full episode. Returns metrics dict."""
        obs = self.env.reset(scenario)
        self._step = 0
        self._done = False
        self._success = False
        self._tracked_injected = set()

        self.planner.reset()
        self.fault_agent.reset()
        self.adaptation.reset()
        self.camera.reset()
        self.movement.reset()
        self.power.reset()
        self.thermal.reset()
        self.navigation.reset()
        self.metrics._reset_episode()

        if hasattr(self.perception, 'fault_injector') and self.perception.fault_injector:
            self.perception.fault_injector.reset()
            if scenario:
                self.perception.fault_injector.load_from_scenario(scenario)

        self._goal = self.goal_parser.parse_and_validate(task_text, self.available_actions)
        self.planner.set_goal(self._goal)

        if self.verbose:
            print(f"\n[AETHER v3] Initializing...")
            print(f"[TaskManager] Goal: {self._goal['task']} → {self._goal['target']} | constraints: {self._goal['constraints']}")
            print(f"[Capabilities] {self.cap_graph.robot_name} | Actions: {len(self.available_actions)}")

        while self._step < max_steps and not self._done:
            self._step += 1
            self.bus.tick(self._step)

            # ── 1. Perceive ──────────────────────────────────────────────────
            state = self.perception.observe(self._step)
            obs_vec = np.array(state["observation"])

            # Track injected faults (once per activation)
            if hasattr(self.perception, 'fault_injector') and self.perception.fault_injector:
                for f in self.perception.fault_injector.get_active_faults():
                    key = (f["fault_type"], f["subsystem"])
                    if key not in self._tracked_injected:
                        self._tracked_injected.add(key)
                        normalized = _INJECTOR_TO_DETECTOR.get(f["fault_type"], f["fault_type"].upper())
                        self.metrics.record_fault_injected(normalized, f["subsystem"], self._step)

            # ── 2. Domain agent ticks ─────────────────────────────────────────
            power_state  = self.power.tick(obs_vec, self._step)
            thermal_state = self.thermal.tick(obs_vec, self._step)
            nav_state    = self.navigation.tick(obs_vec, state, self._step)
            camera_state = self.camera.tick(obs_vec, state, self._step)
            state["nav_confidence"] = nav_state.get("pose_confidence", 1.0)

            # Flush domain agent bus publications so FaultAgent can read them
            self.bus.tick(self._step)

            # ── 3. Fault detection (DRL-First Hybrid FDIR) ────────────────────
            fault_reports = self.fault_agent.process(obs_vec, self._step)
            for fr in fault_reports:
                # Only count rule-based and DRL detections in metrics.
                # Predictive early-warnings have low confidence and inflate FPR.
                if fr.detection_method != "PREDICTIVE":
                    self.metrics.record_fault_detected(fr.fault_type, fr.subsystem,
                                                        self._step, fr.detection_method)
                if self.verbose:
                    print(f"[Step {self._step:03d}] FAULT: {fr.fault_type} ({fr.detection_method}) "
                          f"sev={fr.severity:.2f} conf={fr.confidence:.2f}")

            # ── 4. Adaptation ─────────────────────────────────────────────────
            recovery_action = self.adaptation.adapt(fault_reports, self._step)
            if recovery_action and recovery_action != "stop":
                if self.verbose:
                    print(f"[Adaptation] Strategy: {recovery_action} | Latency: {self.adaptation.avg_latency:.1f}")
                for fr in fault_reports:
                    self.metrics.record_fault_recovered(fr.fault_type, fr.subsystem,
                                                         self._step, int(self.adaptation.avg_latency))
                    self.memory.record_experience(fr.fault_type, recovery_action, "SUCCESS", self._step)

            # ── 5. Plan ───────────────────────────────────────────────────────
            failed_sensors   = state.get("failed_sensors", [])
            failed_actuators = state.get("failed_actuators", [])
            available = self.cap_graph.degraded_actions(failed_sensors, failed_actuators)
            if not available:
                available = ["stop"]

            # Apply recovery effect to fault injector when a recovery action is chosen
            if recovery_action and hasattr(self.perception, 'fault_injector') and self.perception.fault_injector:
                self.perception.fault_injector.apply_recovery(recovery_action)

            if recovery_action and recovery_action in available:
                # Recovery action is a valid movement command — execute it directly
                plan = [recovery_action]
            else:
                # Recovery action is a background FDIR command (e.g. recalibrate_imu,
                # switch_backup_sensor) or None — let the planner keep navigating.
                # Speed/mode effects are already applied via the REPLAN bus message.
                plan = self.planner.plan(state, available)

            if self.verbose and self._step % 10 == 0:
                pos  = state.get("position", [0, 0])
                batt = power_state.get("battery", 1.0)
                tgt  = state.get("target", {})
                tgt_dist = tgt.get("dist", 0) if tgt else 0
                print(f"[Step {self._step:03d}] Pos: ({pos[0]:.1f},{pos[1]:.1f}) "
                      f"| Batt: {batt:.2f} | Target: {tgt_dist:.1f} units "
                      f"| Planner: {self.planner.planner_state}"
                      f"| NavConf: {nav_state.get('pose_confidence', 1.0):.2f}")

            if render:
                print(self.env.render())

            # ── 6. Execute via MovementAgent ──────────────────────────────────
            new_state, success = self.movement.execute(plan[0] if plan else "stop", state)

            # ── 7. Terminal conditions ────────────────────────────────────────
            if new_state.get("at_target"):
                self._done = True
                self._success = True
                if self.verbose:
                    print(f"[Step {self._step:03d}] GOAL REACHED | Success | Steps: {self._step} | SFRI: {self.metrics.compute_sfri():.1f}")

            if self.adaptation.in_safe_mode:
                if self.verbose:
                    print(f"[Step {self._step:03d}] SAFE MODE ENTERED — Mission terminated.")
                self._done = True
                self._success = False

            self.metrics.record_step(0.0)

        self.metrics.record_episode_end(self._success, self._step)
        self.memory.record_episode_end(self._success, self._step, self.metrics.to_dict())
        self.fault_agent.finish_episode()   # online learning update + periodic weight save
        result = self.metrics.to_dict()
        result["task"]    = self._goal["task"]
        result["scenario"] = scenario.get("name", "unknown")
        result["robot"]   = self.cap_graph.robot_name
        return result

    def _on_fault(self, msg) -> None:
        pass  # metrics are tracked directly above

    def _build_adapter(self, config_path: str):
        import json
        with open(config_path) as f:
            cfg = json.load(f)
        actuators = cfg.get("actuators", [])
        max_speed = cfg.get("limits", {}).get("max_speed", 1.0)
        if "thrusters" in actuators:
            from ..adapters.drone_adapter import DroneAdapter
            return DroneAdapter(self.env, max_speed=max_speed)
        from ..adapters.rover_adapter import RoverAdapter
        return RoverAdapter(self.env, max_speed=max_speed)
