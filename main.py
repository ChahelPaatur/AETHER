"""
AETHER v3 — Main entry point.
Usage:
  python main.py --task "navigate to target"
  python main.py --task "follow target while avoiding obstacles" --faults enabled --verbose
  python main.py --task "scan environment" --render --robot rover_v1
  python main.py --mode agent
"""
import argparse
import json
import logging
import os
import re
import sys
import time

from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format="[%(name)s] %(message)s", level=level)


def silence_http_logging() -> None:
    """Suppress noisy HTTP client logging from anthropic/httpx/httpcore."""
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def run_sim(args) -> None:
    """Existing simulation mode — unchanged."""
    from aether.simulation.environment import SimulationEnvironment
    from aether.simulation.scenarios import get_scenario, ScenarioGenerator
    from aether.faults.fault_injector import FaultInjector
    from aether.core.message_bus import MessageBus
    from aether.agents.task_manager import TaskManagerAgent

    env = SimulationEnvironment(seed=args.seed)
    bus = MessageBus()

    fault_prob = {"disabled": 0.0, "enabled": 0.015, "heavy": 0.05}.get(args.faults, 0.0)
    fault_injector = FaultInjector(fault_probability=fault_prob, seed=args.seed)

    config_path = os.path.join(os.path.dirname(__file__), "configs", f"{args.robot}.json")
    agent = TaskManagerAgent(
        env=env,
        config_path=config_path,
        bus=bus,
        log_dir="logs",
        seed=args.seed,
        fault_injector=fault_injector,
        verbose=args.verbose,
        no_learning=args.no_learning,
    )

    scenario = get_scenario(args.scenario)
    if scenario is None:
        scenario = ScenarioGenerator(args.seed).deterministic_scenario(0)

    result = agent.run_episode(
        task_text=args.task,
        scenario=scenario,
        max_steps=args.max_steps,
        render=args.render,
    )

    print(f"\n{'='*55}")
    print(f"AETHER v3 Run Complete")
    print(f"  Status           : {'SUCCESS' if result['success'] else 'FAILURE'}")
    print(f"  Steps            : {result['steps_to_completion']}")
    print(f"  SFRI             : {result['SFRI']:.2f}")
    print(f"  MTTD             : {result['MTTD']:.1f} steps")
    print(f"  MTTR             : {result['MTTR']:.1f} steps")
    print(f"  Detection Rate   : {result['detection_rate']*100:.1f}%")
    print(f"  Recovery Rate    : {result['recovery_rate']*100:.1f}%")
    print(f"  False Positives  : {result['false_positive_rate']*100:.1f}%")
    print(f"  Faults Injected  : {result['faults_injected']}")

    if args.plots:
        from aether.core.visualizer import Visualizer
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = os.path.join("logs", "plots", ts)
        Visualizer().generate_full_report({"runs": [result]}, plot_dir)
        print(f"\nPlots saved to: {plot_dir}/")


# ── Realworld Mode ──────────────────────────────────────────────────

_MOVEMENT_ACTIONS = {
    "move_forward", "move_backward", "turn_left", "turn_right",
    "stop", "reduce_speed",
}

# Realworld thermal noise filter: severity must exceed this AND persist
_THERMAL_SEVERITY_MIN = 0.15
_THERMAL_PERSIST_STEPS = 3


def run_realworld(args) -> None:
    """Realworld mode: live hardware perception with full FDIR pipeline."""
    from aether.simulation.real_perception import RealPerceptionAdapter
    from aether.core.message_bus import MessageBus
    from aether.core.metrics import MetricsTracker
    from aether.agents.perception_agent import PerceptionAgent
    from aether.agents.fault_agent import FaultAgent
    from aether.agents.adaptation_agent import AdaptationAgent
    from aether.agents.memory_agent import MemoryAgent
    from aether.agents.power_agent import PowerAgent
    from aether.agents.thermal_agent import ThermalAgent
    from aether.agents.navigation_agent import NavigationAgent
    from aether.agents.camera_agent import CameraAgent

    continuous = getattr(args, "continuous", False)
    max_steps = args.max_steps

    env = RealPerceptionAdapter(camera_index=0, max_steps=max_steps)
    bus = MessageBus()
    metrics = MetricsTracker()

    # Optional fault injection on real pipeline
    fault_injector = None
    if args.faults != "disabled":
        from aether.faults.fault_injector import FaultInjector
        fault_prob = {"enabled": 0.015, "heavy": 0.05}.get(args.faults, 0.0)
        fault_injector = FaultInjector(fault_probability=fault_prob, seed=args.seed)

    # Domain agents
    perception = PerceptionAgent(env, bus, fault_injector=fault_injector)
    fault_agent = FaultAgent(bus, seed=args.seed, no_learning=args.no_learning)
    adaptation = AdaptationAgent(bus)
    memory = MemoryAgent(bus, session_log_dir="logs")
    power = PowerAgent(bus)
    thermal = ThermalAgent(bus)
    navigation = NavigationAgent(bus)
    camera = CameraAgent(bus)

    # Reset
    scenario = {"name": "realworld", "max_steps": max_steps,
                "task": {"type": "monitor"}, "faults": [], "obstacles": []}
    env.reset(scenario)
    fault_agent.reset()
    adaptation.reset()
    camera.reset()
    power.reset()
    thermal.reset()
    navigation.reset()
    metrics._reset_episode()

    if fault_injector:
        fault_injector.reset()

    mode_label = "continuous" if continuous else f"{max_steps} steps"
    print("=" * 55)
    print("AETHER v3 — Realworld Mode")
    print("=" * 55)
    print(f"  Camera:     {'OK' if 'camera' not in env.failed_sensors else 'UNAVAILABLE'}")
    print(f"  psutil:     {'OK' if env._psutil else 'UNAVAILABLE'}")
    print(f"  Faults:     {args.faults}")
    print(f"  Mode:       {mode_label}")
    print(f"  Adapter:    MONITOR ONLY (no hardware actuators)")
    print(f"  Verbose:    {args.verbose}")
    print(f"\n  Running FDIR pipeline on live data. Press Ctrl+C to stop.\n")

    step = 0
    faults_total = 0
    thermal_consec = 0  # consecutive steps with THERMAL_ANOMALY above threshold
    try:
        while continuous or step < max_steps:
            step += 1
            bus.tick(step)

            # 1. Perceive from real hardware
            state = perception.observe(step)
            obs_vec = np.array(state["observation"])

            # 2. Domain agent ticks
            power_state = power.tick(obs_vec, step)
            thermal_state = thermal.tick(obs_vec, step)
            nav_state = navigation.tick(obs_vec, state, step)
            camera_state = camera.tick(obs_vec, state, step)
            bus.tick(step)

            # 3. Fault detection (DRL-First Hybrid FDIR)
            fault_reports = fault_agent.process(obs_vec, step)

            # Filter THERMAL_ANOMALY noise: must exceed severity threshold
            # AND persist for consecutive steps before reporting
            has_thermal = False
            filtered_reports = []
            for fr in fault_reports:
                if fr.fault_type == "THERMAL_ANOMALY":
                    if fr.severity < _THERMAL_SEVERITY_MIN:
                        continue  # below noise floor — drop silently
                    has_thermal = True
                    thermal_consec += 1
                    if thermal_consec < _THERMAL_PERSIST_STEPS:
                        continue  # not yet confirmed — suppress
                filtered_reports.append(fr)
            if not has_thermal:
                thermal_consec = 0

            for fr in filtered_reports:
                if fr.detection_method != "PREDICTIVE":
                    metrics.record_fault_detected(fr.fault_type, fr.subsystem,
                                                  step, fr.detection_method)
                    faults_total += 1
                    print(f"  [Step {step:03d}] FAULT: {fr.fault_type} "
                          f"({fr.detection_method}) sev={fr.severity:.2f} "
                          f"conf={fr.confidence:.2f}")

            # 4. Adaptation — monitor only, no movement commands
            recovery_action = adaptation.adapt(filtered_reports, step)
            if recovery_action and recovery_action != "stop":
                if recovery_action in _MOVEMENT_ACTIONS:
                    # No hardware adapter — suppress movement, log as monitor
                    if args.verbose:
                        print(f"  [Step {step:03d}] MONITOR ONLY: "
                              f"suppressed {recovery_action} (no actuator)")
                else:
                    print(f"  [Step {step:03d}] ADAPT: {recovery_action} "
                          f"| Latency: {adaptation.avg_latency:.1f}")
                for fr in filtered_reports:
                    metrics.record_fault_recovered(fr.fault_type, fr.subsystem,
                                                   step, int(adaptation.avg_latency))
                    memory.record_experience(fr.fault_type, recovery_action,
                                             "SUCCESS", step)

            metrics.record_step(0.0)

            # 5. Status display
            if args.verbose or step % 10 == 0:
                sd = env.get_state_dict()
                motion = "MOTION" if sd.get("motion_detected") else "static"
                print(f"  [Step {step:03d}] "
                      f"Batt: {sd['battery']*100:.0f}% | "
                      f"CPU: {sd.get('cpu_percent', 0):.0f}% | "
                      f"RAM: {sd.get('ram_percent', 0):.0f}% | "
                      f"Camera: {motion} | "
                      f"Faults: {faults_total}")

            if args.render:
                print(f"    {env.render()}")

    except KeyboardInterrupt:
        print(f"\n  Stopped by user at step {step}.")

    # Cleanup
    env.close()
    metrics.record_episode_end(True, step)
    memory.record_episode_end(True, step, metrics.to_dict())
    result = metrics.to_dict()

    print(f"\n{'='*55}")
    print(f"Realworld Session Complete")
    print(f"  Steps:          {step}")
    print(f"  Faults Detected: {faults_total}")
    print(f"  SFRI:           {result['SFRI']:.2f}")
    print(f"  Detection Rate: {result['detection_rate']*100:.1f}%")
    print(f"  Recovery Rate:  {result['recovery_rate']*100:.1f}%")
    print(f"  False Positives: {result['false_positive_rate']*100:.1f}%")
    print(f"{'='*55}")


# ── Agent Mode ──────────────────────────────────────────────────────

def _print_activity(label: str, msg: str, indent: int = 2) -> None:
    prefix = " " * indent
    print(f"{prefix}[{label}] {msg}")


def _print_fault_alert(fault_type: str, details: str) -> None:
    print(f"  !! FAULT: {fault_type} — {details}")


_AGENT_MEMORY_PATH = os.path.join("logs", "agent_memory.json")


def _load_agent_memory(n: int = 5) -> List[Dict]:
    """Load the last n entries from the persistent agent memory file."""
    if not os.path.exists(_AGENT_MEMORY_PATH):
        return []
    try:
        with open(_AGENT_MEMORY_PATH) as f:
            entries = json.load(f)
        return entries[-n:] if len(entries) > n else entries
    except (json.JSONDecodeError, OSError):
        return []


def _save_agent_memory_entry(entry: Dict) -> None:
    """Append a structured entry to the persistent agent memory file."""
    os.makedirs(os.path.dirname(_AGENT_MEMORY_PATH) or ".", exist_ok=True)
    existing = []
    if os.path.exists(_AGENT_MEMORY_PATH):
        try:
            with open(_AGENT_MEMORY_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []
    existing.append(entry)
    with open(_AGENT_MEMORY_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def _extract_key_facts(output) -> str:
    """Extract a short key-facts string from tool output for memory storage."""
    if output is None:
        return ""
    text = str(output)
    # Strip [KNOWLEDGE] / [local extraction] prefixes
    for prefix in ["[KNOWLEDGE]\n", "[KNOWLEDGE] ", "[local extraction]\n"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    # Take the first 200 chars as key facts
    text = text.strip()
    if len(text) > 200:
        return text[:197] + "..."
    return text


def run_agent(args) -> None:
    """Real agent mode: prompt-driven loop with live tool execution."""
    from aether.core.tool_registry import ToolRegistry, ToolResult, register_built_tools
    from aether.core.tool_discovery import ToolDiscovery
    from aether.core.llm_planner import LLMPlanner, build_tool_descriptions
    from aether.agents.correction_agent import CorrectionAgent
    from aether.core.goal_parser import GoalParser
    from aether.core.planner import Planner
    from aether.core.navigation_engine import NavigationEngine
    from aether.core.tool_builder import ToolBuilder
    from aether.core.auto_installer import AutoInstaller
    from aether.core.auto_updater import AutoUpdater

    auto_install = getattr(args, "auto_install", False)
    no_install = getattr(args, "no_install", False)
    auto_update = getattr(args, "auto_update", False)
    no_update = getattr(args, "no_update", False)

    # ── Step 1: Silence HTTP logging ──────────────────────────────────
    silence_http_logging()

    # ── Step 2: Banner already printed in main() ──────────────────────

    # ── Step 3: AutoUpdater ───────────────────────────────────────────
    updater = AutoUpdater()
    AutoUpdater.ensure_version_file(os.path.dirname(os.path.abspath(__file__)))
    updater.run(auto_update=auto_update, no_update=no_update)

    # ── Step 4: Capability Discovery ──────────────────────────────────
    discovery = ToolDiscovery()
    discovery.discover()
    discovery.print_summary()

    # ── Step 5: AutoInstaller ─────────────────────────────────────────
    installer = AutoInstaller(discovery.manifest)
    packages_installed = installer.run(auto_install=auto_install, no_install=no_install)

    # ── Step 6: Re-run discovery if anything was installed ────────────
    if packages_installed:
        _print_activity("DISCOVERY", "Re-running discovery after installs...")
        discovery = ToolDiscovery()
        discovery.discover()
        discovery.print_summary()

    # ── Step 6b: Hardware Calibration ─────────────────────────────────
    from aether.core.calibration import CalibrationWizard, load_calibration
    calibrate = getattr(args, "calibrate", False)
    recalibrate = getattr(args, "recalibrate", False)
    auto_cal = getattr(args, "auto_calibrate", False)

    calibration_profile = None
    wizard = CalibrationWizard(
        discovery.manifest,
        auto_mode=auto_cal,
    )
    if calibrate or recalibrate:
        if recalibrate or wizard.needs_calibration(force=True):
            calibration_profile = wizard.run()
    else:
        # Auto-detect: load existing or prompt if new hardware found
        calibration_profile = load_calibration()
        if calibration_profile:
            _print_activity("CAL", f"Loaded: {calibration_profile['robot_name']} "
                            f"({calibration_profile['robot_type']})")
        elif wizard.needs_calibration():
            _print_activity("CAL", "New hardware detected — running calibration wizard")
            calibration_profile = wizard.run()

    # Enrich manifest with calibration data
    if calibration_profile:
        discovery.manifest["calibration"] = calibration_profile
        cal_actions = CalibrationWizard.generate_actions(calibration_profile)
        discovery.manifest["calibrated_actions"] = [a["name"] for a in cal_actions]
        discovery.manifest["robot_purpose"] = calibration_profile.get("robot_purpose", "")

    registry = ToolRegistry()
    parser = GoalParser()
    planner = Planner()

    # ── Step 7: Build Tools + Navigation Engine ───────────────────────
    builder = ToolBuilder(discovery.manifest)
    built_tools = builder.build_all()
    _print_activity("TOOLS", builder.build_summary())

    nav_engine = NavigationEngine(discovery.manifest, tools=built_tools)
    nav_actions = nav_engine.available_actions()
    _print_activity("NAV", f"Level {nav_engine.level} ({nav_engine.level_name}) — "
                    f"{len(nav_actions)} navigation actions")

    # Merge calibrated actions into navigation actions
    if calibration_profile:
        cal_action_names = discovery.manifest.get("calibrated_actions", [])
        nav_actions = sorted(set(nav_actions + cal_action_names))
        _print_activity("CAL", f"Robot mode: {calibration_profile['robot_type']} — "
                        f"{len(cal_action_names)} calibrated actions")

    # Pass capability manifest to planner so it never plans with unavailable tools
    manifest = discovery.manifest
    manifest["tool_descriptions"] = build_tool_descriptions(registry)
    manifest["navigation_actions"] = nav_actions
    manifest["navigation_level"] = nav_engine.level
    # Merge navigation actions into available_tools for the LLM planner
    all_available = sorted(set(manifest.get("available_tools", []) + nav_actions))
    manifest["available_tools"] = all_available

    # Register ToolBuilder tools + NavigationEngine actions into ToolRegistry
    # (after manifest enrichment so system_metrics includes full capabilities)
    register_built_tools(registry, built_tools, nav_engine, manifest=manifest)
    planner.update_weights({"capabilities": manifest})

    # ── LLM Planner ───────────────────────────────────────────────────
    llm_planner = LLMPlanner()
    if llm_planner.available:
        _print_activity("LLM", "LLM Planner active (claude-sonnet-4-20250514)")
    else:
        _print_activity("LLM", "LLM Planner unavailable — using keyword fallback")

    # ── Correction Agent ──────────────────────────────────────────────
    corrector = CorrectionAgent(available_tools=registry.available_tools())
    if corrector.available:
        _print_activity("CORRECT", "CorrectionAgent active — post-step verification enabled")
    else:
        _print_activity("CORRECT", "CorrectionAgent inactive (no API key) — skipping post-step checks")

    # Load domain definitions and inject into summarize_text
    defs_path = os.path.join(os.path.dirname(__file__), "context", "aether_definitions.txt")
    if os.path.exists(defs_path):
        with open(defs_path) as f:
            defs_text = f.read()
        summarizer = registry.get("summarize_text")
        if summarizer and hasattr(summarizer, "set_system_context"):
            summarizer.set_system_context(defs_text)
            _print_activity("CTX", f"Loaded {len(defs_text)} chars from {defs_path}")

    available = registry.available_tools()
    all_tools = registry.list_tools()

    # Load persistent memory
    recent_memory = _load_agent_memory(5)

    print()
    print("=" * 55)
    print("AETHER v3 — Agent Mode")
    print("=" * 55)
    print(f"Available tools ({len(available)}):")
    for t in all_tools:
        status = "OK" if t["available"] else "UNAVAILABLE"
        print(f"  {t['name']:<20} [{status}]  {t['description']}")
    if recent_memory:
        print(f"\nRecent memory ({len(recent_memory)} entries):")
        for mem in recent_memory:
            status = "OK" if mem.get("outcome") == "SUCCESS" else "FAIL"
            print(f"  [{status}] {mem.get('objective', '?')[:60]}")
    print(f"\nType an objective (or 'quit' to exit).\n")

    # Fault tracking
    faults_detected = 0
    total_actions = 0

    while True:
        try:
            objective = input("objective> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting agent mode.")
            break

        if not objective or objective.lower() in ("quit", "exit", "q"):
            print("Exiting agent mode.")
            break

        _print_activity("GOAL", f"Parsing objective: {objective}")

        # Use GoalParser for structure, but fall back to raw text for agent-mode planning
        parsed = parser.parse_and_validate(objective, available)
        achievable = parsed.get("validation", {}).get("achievable", False)
        if not achievable:
            # GoalParser is sim-oriented; use raw text directly for agent mode
            parsed = {"task": objective, "target": objective, "subtasks": [],
                      "constraints": [], "raw": objective}

        task_info = parsed.get("task", objective)
        subtasks = parsed.get("subtasks", [])

        _print_activity("PLAN", f"Task: {task_info}")
        if subtasks:
            _print_activity("PLAN", f"Subtasks: {', '.join(subtasks)}")

        # Build action plan: try LLM planner first, fall back to keywords
        plan = None
        plan_source = "keyword"
        if llm_planner.available:
            _print_activity("PLAN", "Sending to LLM planner...")
            plan = llm_planner.plan(objective, manifest, recent_memory)
            if plan:
                plan_source = "LLM"
        if plan is None:
            plan = _build_agent_plan(parsed, available, recent_memory)
            plan_source = "keyword"
        _print_activity("PLAN", f"[{plan_source}] {' → '.join(a['tool'] for a in plan)}")

        print(f"\n  --- Executing {len(plan)} action(s) ---")
        step_faults = 0
        prev_output = None  # Chain: previous step's output feeds next step's input
        sim_accumulator = []  # Collect outputs from consecutive run_simulation steps
        for i, action in enumerate(plan, 1):
            tool_name = action["tool"]
            params = action.get("params", {})

            # Pipe previous output into this step's input parameter
            pipe_key = action.get("pipe_input")
            if pipe_key:
                # If we accumulated multiple sim results, merge them as prev_output
                if sim_accumulator and prev_output is None:
                    prev_output = _format_sim_accumulator(sim_accumulator)
                    sim_accumulator = []
                if prev_output:
                    params[pipe_key] = str(prev_output)
                else:
                    _print_activity("SKIP", f"[{i}/{len(plan)}] {tool_name} (no input from previous step)")
                    continue

            # When transitioning from sim steps to non-sim, flush accumulator
            if tool_name != "run_simulation" and sim_accumulator:
                prev_output = _format_sim_accumulator(sim_accumulator)
                sim_accumulator = []

            _print_activity("EXEC", f"[{i}/{len(plan)}] {tool_name}({_short_params(params)})")
            total_actions += 1

            result = registry.execute(tool_name, params)

            # Fallback: if primary tool fails and a fallback is defined, try it
            if not result.success and action.get("fallback_tool"):
                fb_tool = action["fallback_tool"]
                fb_params = action.get("fallback_params") or {}

                # Safe fallback for write_file: mkdir -p then retry write, never
                # pass file content to execute_shell
                if tool_name == "write_file" and fb_tool == "execute_shell":
                    dirpath = os.path.dirname(params.get("path", ""))
                    if dirpath:
                        _print_activity("FALLBACK", f"write_file failed → mkdir -p {dirpath}")
                        registry.execute("execute_shell",
                                         {"command": f"mkdir -p {dirpath}"})
                        result = registry.execute("write_file", params)
                        total_actions += 2
                    else:
                        _print_activity("FALLBACK", f"write_file failed — no directory to create")
                else:
                    _print_activity("FALLBACK", f"{tool_name} failed → trying {fb_tool}")
                    if pipe_key and prev_output:
                        fb_params[pipe_key] = str(prev_output)
                    result = registry.execute(fb_tool, fb_params)
                    total_actions += 1
                    tool_name = fb_tool  # for logging below

            # CorrectionAgent: evaluate result against expected output
            if corrector.available and action.get("expected_output"):
                eval_result = corrector.evaluate_and_correct(action, result, registry)
                result = eval_result["final_result"]
                if eval_result["corrected"]:
                    total_actions += eval_result["attempts"]
                    _print_activity("CORRECT",
                                    f"Applied {eval_result['attempts']} correction(s) → "
                                    f"{'OK' if result.success else 'FAILED'}")

            # FaultAgent monitoring: failed/timeout/unexpected → TASK_FAILURE
            if not result.success:
                step_faults += 1
                faults_detected += 1
                _print_fault_alert("TASK_FAILURE",
                                   f"{tool_name} failed: {result.error} "
                                   f"({result.duration_ms:.0f}ms)")
                prev_output = None
            # Per-tool timeout thresholds: API-backed tools get 30s, others 10s
            elif result.duration_ms > (30000 if tool_name in ("web_search", "summarize_text", "run_simulation") else 10000):
                step_faults += 1
                faults_detected += 1
                _print_fault_alert("TIMEOUT_WARNING",
                                   f"{tool_name} took {result.duration_ms:.0f}ms")
                if tool_name == "run_simulation":
                    sim_accumulator.append(("timeout", params.get("scenario", "?"), result.output))
                prev_output = result.output
            else:
                summary = _summarize_output(result.output)
                label = "KNOWLEDGE" if (tool_name == "web_search"
                                        and str(result.output).startswith("[KNOWLEDGE]")) else "OK"
                _print_activity(label, f"{tool_name} → {summary} ({result.duration_ms:.0f}ms)")
                if tool_name == "run_simulation":
                    sim_accumulator.append(("ok", params.get("scenario", "?"), result.output))
                prev_output = result.output

        # Flush any remaining sim results
        if sim_accumulator:
            prev_output = _format_sim_accumulator(sim_accumulator)

        # Fix 2: auto-chain summarize + write after simulation if objective demands it
        task_lower = objective.lower()
        plan_tools = {a["tool"] for a in plan}
        obj_wants_analyze = any(w in task_lower for w in ["analyze", "summarize", "report"])
        obj_wants_save = any(w in task_lower for w in ["save", "write", "store"])
        ran_sim = "run_simulation" in plan_tools

        if ran_sim and prev_output and obj_wants_analyze and "summarize_text" not in plan_tools:
            _print_activity("CHAIN", "Auto-appending summarize_text for analysis")
            prompt = f"Analyze these simulation results and write a comparison report:\n\n{prev_output}"
            sum_result = registry.execute("summarize_text", {"text": prompt})
            total_actions += 1
            if sum_result.success:
                _print_activity("OK", f"summarize_text → {_summarize_output(sum_result.output)} ({sum_result.duration_ms:.0f}ms)")
                prev_output = sum_result.output
            else:
                step_faults += 1
                faults_detected += 1
                _print_fault_alert("TASK_FAILURE", f"summarize_text failed: {sum_result.error}")

        if ran_sim and prev_output and obj_wants_save and "write_file" not in plan_tools:
            filename = _extract_filename(objective)
            _print_activity("CHAIN", f"Auto-appending write_file → {filename}")
            wr_result = registry.execute("write_file", {"path": filename, "content": str(prev_output)})
            total_actions += 1
            if wr_result.success:
                _print_activity("OK", f"write_file → {_summarize_output(wr_result.output)} ({wr_result.duration_ms:.0f}ms)")
                prev_output = wr_result.output
            else:
                step_faults += 1
                faults_detected += 1
                _print_fault_alert("TASK_FAILURE", f"write_file failed: {wr_result.error}")

        # Episode summary
        outcome = "SUCCESS" if step_faults == 0 else "DEGRADED"
        print(f"\n  --- Objective Complete ---")
        print(f"  Actions:  {len(plan)}")
        print(f"  Faults:   {step_faults}")
        print(f"  Status:   {outcome}" + (f" ({step_faults} fault(s) during execution)" if step_faults else ""))

        # Persist to agent memory
        memory_entry = {
            "timestamp": time.time(),
            "objective": objective,
            "tool_chain": [a["tool"] for a in plan],
            "outcome": outcome,
            "key_facts": _extract_key_facts(prev_output),
        }
        _save_agent_memory_entry(memory_entry)
        recent_memory = _load_agent_memory(5)
        _print_activity("MEM", f"Saved to agent memory ({len(recent_memory)} entries total)")
        print()

    # Session summary
    print(f"\n{'='*55}")
    print(f"Agent Session Summary")
    print(f"  Total actions:  {total_actions}")
    print(f"  Total faults:   {faults_detected}")
    fpr = faults_detected / max(total_actions, 1)
    print(f"  Fault rate:     {fpr*100:.1f}%")
    print(f"{'='*55}")


_FILEPATH_RE = re.compile(r'[\w./\-]+\.(?:json|csv|txt|py|md|log|html)')


def _extract_filepath(text: str) -> str:
    """Extract the FIRST file path from objective text using regex.

    Scans for any substring matching path/to/file.ext pattern.
    Used for read_file where we want the input file.
    """
    m = _FILEPATH_RE.search(text)
    return m.group(0) if m else ""


def _extract_filename(text: str) -> str:
    """Extract the LAST file path from objective text using regex.

    Scans for all path/to/file.ext matches and returns the last one.
    Used for write_file where the output path typically appears last.
    """
    matches = _FILEPATH_RE.findall(text)
    if matches:
        return matches[-1]
    return "output.txt"


_VALID_SCENARIOS = {"simple", "obstacles", "imu_fault", "battery",
                    "compound", "fault_heavy", "multi_task"}
_VALID_FAULT_MODES = {"disabled", "enabled", "heavy"}


def _extract_sim_params(raw: str) -> Dict:
    """Extract scenario(s), fault_mode, runs count, and max_steps from objective."""
    lower = raw.lower()
    params: Dict[str, Any] = {}

    # Find ALL mentioned scenario names (word-boundary match to avoid false positives)
    found_scenarios = []
    for s in _VALID_SCENARIOS:
        # Match "imu_fault" or "imu fault" as whole words
        pattern = r'\b' + re.escape(s) + r'\b'
        pattern_spaced = r'\b' + re.escape(s.replace("_", " ")) + r'\b'
        if re.search(pattern, lower) or re.search(pattern_spaced, lower):
            found_scenarios.append(s)
    if len(found_scenarios) > 1:
        params["scenarios"] = found_scenarios
    elif len(found_scenarios) == 1:
        params["scenario"] = found_scenarios[0]
    else:
        params["scenario"] = "simple"

    # Find fault mode
    for fm in _VALID_FAULT_MODES:
        if fm in lower:
            params["fault_mode"] = fm
            break
    if "fault_mode" not in params:
        if any(w in lower for w in ["no fault", "without fault", "fault off"]):
            params["fault_mode"] = "disabled"
        else:
            params["fault_mode"] = "enabled"

    # Repetition count: "5 simulations", "10 runs", "×5", "x5"
    m = re.search(r'(\d+)\s*(?:simulation|run|episode|time)s?\b', lower)
    if m:
        runs = int(m.group(1))
        if runs > 1:
            params["runs"] = runs
    if "runs" not in params:
        m = re.search(r'[×x]\s*(\d+)', lower)
        if m:
            params["runs"] = int(m.group(1))

    # Optional max_steps
    m = re.search(r'(\d+)\s*(?:steps?|max.?steps?)', lower)
    if m:
        params["max_steps"] = int(m.group(1))

    return params


def _extract_after(raw: str, task: str, keywords: list) -> str:
    """Strip leading verb/keyword to extract the argument portion."""
    for kw in keywords:
        idx = task.find(kw)
        if idx >= 0:
            rest = raw[idx + len(kw):].strip()
            return rest if rest else raw
    return raw


def _extract_search_query(raw: str, task: str) -> str:
    """Extract just the search topic, stripping trailing action clauses.

    Strategy: strip verb prefix, then cut at the first comma or action clause.
    If result is too short or still contains task words, use text before first comma.
    """
    import re
    _TASK_WORDS = {"summary", "paragraph", "save", "write", "file", "confirm",
                   "store", "output", "create"}

    query = _extract_after(raw, task, ["search for ", "search ", "research ",
                                        "look up ", "find "])
    # Cut at comma, "and summarize/save/write", "save to", "then"
    query = re.split(r',\s*|\s+and\s+(?:summarize|save|confirm|store|write|then)'
                     r'|\s+save\s+to|\s+store\s+in|\s+output\s+to|\s+then\s',
                     query, maxsplit=1, flags=re.I)[0].strip()

    # Validate: if still contains task words or too short, use text before first comma
    words = set(query.lower().split())
    if len(query) < 10 or words & _TASK_WORDS:
        if ',' in raw:
            query = raw[:raw.index(',')].strip()
            # Strip leading verb
            for prefix in ["research ", "search for ", "search ", "look up ", "find "]:
                if query.lower().startswith(prefix):
                    query = query[len(prefix):].strip()
                    break

    return query.strip(' ,')


def _rebuild_plan_from_chain(chain: List[str], parsed: Dict,
                            available: list) -> list:
    """Reconstruct a plan from a stored tool chain, re-extracting params."""
    raw = parsed.get("raw", parsed.get("task", ""))
    task = raw.lower()
    plan = []
    for idx, tool_name in enumerate(chain):
        if tool_name not in available:
            continue
        if tool_name == "web_search":
            query = _extract_search_query(raw, task)
            plan.append({"tool": "web_search", "params": {"query": query}})
        elif tool_name == "read_file":
            path = _extract_filepath(raw)
            if not path:
                path = raw
            plan.append({"tool": "read_file", "params": {"path": path}})
        elif tool_name == "write_file":
            filename = _extract_filename(raw)
            pipe = "content" if idx > 0 else None
            entry = {"tool": "write_file", "params": {"path": filename, "content": ""}}
            if pipe:
                entry["pipe_input"] = pipe
            plan.append(entry)
        elif tool_name == "summarize_text":
            pipe = "text" if idx > 0 else None
            entry = {"tool": "summarize_text", "params": {"text": ""}}
            if pipe:
                entry["pipe_input"] = pipe
            plan.append(entry)
        elif tool_name == "execute_shell":
            cmd = _extract_after(raw, task, ["run ", "execute ", "shell ",
                                              "command ", "install ", "build "])
            plan.append({"tool": "execute_shell", "params": {"command": cmd}})
        elif tool_name == "call_api":
            url = _extract_after(raw, task, ["fetch ", "request ", "call api ",
                                              "call ", "api "])
            plan.append({"tool": "call_api", "params": {"url": url, "method": "GET"}})
        elif tool_name == "run_simulation":
            plan.append({"tool": "run_simulation", "params": _extract_sim_params(raw)})
    return plan if plan else [{"tool": "web_search", "params": {"query": raw}}]


def _build_agent_plan(parsed: Dict, available: list,
                      memory: Optional[List[Dict]] = None) -> list:
    """Convert a parsed goal into an ordered list of tool calls.

    Multi-step chain logic — detects compound objectives and builds pipelines
    where each step's output feeds the next step's input via pipe_input key.

    If memory contains a previous successful run of the same objective,
    reuse that tool chain instead of re-deriving it.

    Chain patterns:
      search + save + confirm → [web_search, summarize_text, write_file, execute_shell(verify)]
      search + summarize      → [web_search, summarize_text]
      summarize + save        → [summarize_text, write_file]
      read + summarize        → [read_file, summarize_text]
      Single-intent           → [single tool]
    """
    raw = parsed.get("raw", parsed.get("task", ""))
    task = raw.lower()
    plan = []

    # Check memory for a previously successful identical objective
    if memory:
        for mem in reversed(memory):
            if (mem.get("objective", "").lower() == task
                    and mem.get("outcome") == "SUCCESS"
                    and mem.get("tool_chain")):
                cached_chain = mem["tool_chain"]
                # Verify cached chain satisfies current objective's requirements
                needs_summarize = any(w in task for w in ["analyze", "summarize",
                                                           "explain", "report"])
                needs_write = any(w in task for w in ["save", "write", "store",
                                                       "output to ", "report"])
                chain_has_summarize = "summarize_text" in cached_chain
                chain_has_write = "write_file" in cached_chain
                if needs_summarize and not chain_has_summarize:
                    _print_activity("MEM", "Cached chain missing summarize — rebuilding")
                    break
                if needs_write and not chain_has_write:
                    _print_activity("MEM", "Cached chain missing write — rebuilding")
                    break
                _print_activity("MEM", f"Reusing tool chain from previous run")
                return _rebuild_plan_from_chain(cached_chain, parsed, available)

    # ── Detect intent flags ──
    has_search = any(w in task for w in ["search", "find information", "research",
                                          "latest", "recent", "look up", "google",
                                          "what is", "who is", "how to"])
    has_summarize = any(w in task for w in ["summarize", "analyze", "explain",
                                             "tldr", "summary"])
    has_save = any(w in task for w in ["save", "write to ", "store", "output to "])
    has_confirm = any(w in task for w in ["confirm", "verify", "check"])
    has_list = any(w in task for w in ["list file", "list dir", "current dir",
                                        "show file", "show dir"])
    has_run = any(w in task for w in ["run ", "execute ", "command ", "shell ",
                                       "install ", "build "])
    has_read = any(w in task for w in ["read ", "open ", "cat ", "view "])
    has_write = any(w in task for w in ["write ", "save "]) and not has_search
    has_api = any(w in task for w in ["api ", "fetch ", "request ", "call api"])
    has_sim = any(w in task for w in ["simulat", "run sim", "run episode",
                                       "aether sim", "run scenario"])

    # ── Simulation (highest priority — before generic "run") ──
    if has_sim and "run_simulation" in available:
        sim_params = _extract_sim_params(raw)
        scenarios = sim_params.pop("scenarios", [])
        if not scenarios:
            scenarios = [sim_params.pop("scenario", "simple")]
        else:
            sim_params.pop("scenario", None)

        # Multi-scenario: one run_simulation step per scenario
        if len(scenarios) > 1:
            for sc in scenarios:
                step_params = {**sim_params, "scenario": sc}
                plan.append({"tool": "run_simulation", "params": step_params})
        else:
            plan.append({"tool": "run_simulation",
                          "params": {**sim_params, "scenario": scenarios[0]}})

        # Chain: sim → summarize → write → verify when objective asks for analysis + save
        if has_summarize or "analyze" in task or "report" in task:
            plan.append({"tool": "summarize_text", "params": {"text": ""},
                          "pipe_input": "text"})
        if has_save or "report" in task or "save" in task:
            filename = _extract_filename(raw)
            plan.append({"tool": "write_file",
                          "params": {"path": filename, "content": ""},
                          "pipe_input": "content"})
        if has_confirm or "verify" in task or ("report" in task and has_save):
            filename = _extract_filename(raw)
            plan.append({"tool": "execute_shell",
                          "params": {"command": f"cat {filename}"}})

        return plan

    # ── Multi-step chains (checked first) ──

    # search + save + confirm → [web_search, summarize_text, write_file, execute_shell(verify)]
    if has_search and has_save and has_confirm:
        query = _extract_search_query(raw, task)
        filename = _extract_filename(raw)
        plan.append({"tool": "web_search", "params": {"query": query}})
        plan.append({"tool": "summarize_text", "params": {"text": ""}, "pipe_input": "text"})
        plan.append({"tool": "write_file", "params": {"path": filename, "content": ""},
                      "pipe_input": "content"})
        plan.append({"tool": "execute_shell",
                      "params": {"command": f"cat {filename}"},
                      "pipe_input": None})
        return plan

    # search + summarize + save → [web_search, summarize_text, write_file]
    if has_search and has_summarize and has_save:
        query = _extract_search_query(raw, task)
        filename = _extract_filename(raw)
        plan.append({"tool": "web_search", "params": {"query": query}})
        plan.append({"tool": "summarize_text", "params": {"text": ""}, "pipe_input": "text"})
        plan.append({"tool": "write_file", "params": {"path": filename, "content": ""},
                      "pipe_input": "content"})
        return plan

    # search + summarize → [web_search, summarize_text]
    if has_search and has_summarize:
        query = _extract_search_query(raw, task)
        plan.append({"tool": "web_search", "params": {"query": query}})
        plan.append({"tool": "summarize_text", "params": {"text": ""}, "pipe_input": "text"})
        return plan

    # search + save → [web_search, write_file]
    if has_search and has_save:
        query = _extract_search_query(raw, task)
        filename = _extract_filename(raw)
        plan.append({"tool": "web_search", "params": {"query": query}})
        plan.append({"tool": "write_file", "params": {"path": filename, "content": ""},
                      "pipe_input": "content"})
        return plan

    # read + summarize → [read_file, summarize_text]
    if has_read and has_summarize:
        path = _extract_filepath(raw)
        if not path:
            path = _extract_after(raw, task, ["read file ", "read ", "open ", "cat ", "view ",
                                               "summarize ", "analyze ", "explain "])
        plan.append({"tool": "read_file", "params": {"path": path}})
        plan.append({"tool": "summarize_text", "params": {"text": ""}, "pipe_input": "text"})
        return plan

    # summarize + save → [summarize_text, write_file]
    if has_summarize and has_save:
        text = _extract_after(raw, task, ["summarize ", "analyze ", "explain "])
        filename = _extract_filename(raw)
        plan.append({"tool": "summarize_text", "params": {"text": text}})
        plan.append({"tool": "write_file", "params": {"path": filename, "content": ""},
                      "pipe_input": "content"})
        return plan

    # ── Single-intent plans ──

    # List files / current directory
    if has_list:
        target = _extract_after(raw, task, ["list files in ", "list files ",
                                             "list directory ", "show files in ",
                                             "show files ", "current directory"])
        cmd = f"ls {target}" if target != raw else "ls ."
        plan.append({"tool": "execute_shell", "params": {"command": cmd}})

    # Run / execute command
    elif has_run:
        cmd = _extract_after(raw, task, ["run ", "execute ", "shell ",
                                          "command ", "install ", "build "])
        plan.append({"tool": "execute_shell", "params": {"command": cmd}})

    # Read file
    elif has_read:
        path = _extract_filepath(raw)
        if not path:
            path = _extract_after(raw, task, ["read file ", "read ", "open ",
                                               "cat ", "show ", "view "])
        plan.append({"tool": "read_file", "params": {"path": path}})

    # Write / save
    elif has_write:
        path = _extract_filename(raw)  # last filepath match = output destination
        content = parsed.get("content", "")
        plan.append({"tool": "write_file", "params": {"path": path, "content": content}})

    # Summarize / analyze / explain
    elif has_summarize:
        text = _extract_after(raw, task, ["summarize ", "analyze ", "explain ",
                                           "tldr ", "summary of "])
        plan.append({"tool": "summarize_text", "params": {"text": text}})

    # Web search
    elif has_search:
        query = _extract_after(raw, task, ["search for ", "search ",
                                            "find information about ",
                                            "find information ", "research ",
                                            "look up ", "google ",
                                            "what is ", "who is ", "how to "])
        plan.append({"tool": "web_search", "params": {"query": query}})

    # API call
    elif has_api:
        url = _extract_after(raw, task, ["fetch ", "request ", "call api ",
                                          "call ", "api "])
        plan.append({"tool": "call_api", "params": {"url": url, "method": "GET"}})

    # Fallback: web search
    if not plan and "web_search" in available:
        plan.append({"tool": "web_search", "params": {"query": raw}})

    return plan


def _format_sim_accumulator(entries: list) -> str:
    """Format accumulated run_simulation outputs into a text report for piping."""
    import json
    parts = []
    for status, scenario, output in entries:
        parts.append(f"=== Scenario: {scenario} ===")
        if isinstance(output, dict):
            # Single run or aggregated result
            if "mean" in output:
                parts.append(f"Runs: {output.get('runs', '?')}")
                parts.append("Averaged metrics:")
                for k, v in output["mean"].items():
                    parts.append(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")
            else:
                for k, v in output.items():
                    if isinstance(v, (int, float)):
                        parts.append(f"  {k}: {v}")
        else:
            parts.append(str(output)[:500])
        parts.append("")
    return "\n".join(parts)


def _short_params(params: Dict) -> str:
    """Compact parameter display for activity log."""
    parts = []
    for k, v in params.items():
        val = str(v)
        if len(val) > 40:
            val = val[:37] + "..."
        parts.append(f"{k}={val}")
    return ", ".join(parts)


def _summarize_output(output) -> str:
    """One-line summary of tool output for the activity log."""
    if output is None:
        return "(no output)"
    s = str(output)
    if len(s) > 80:
        return s[:77] + "..."
    return s


def _execute_objective(objective: str, registry, llm_planner, corrector,
                       parser_obj, manifest, recent_memory,
                       available) -> Dict:
    """Execute a single objective and return structured result.

    Used by both agent mode (interactive) and server mode (HTTP).
    Returns dict with: status, actions, faults, result, plan_source, tool_chain.
    """
    start_time = time.time()

    # Parse objective
    parsed = parser_obj.parse_and_validate(objective, available)
    achievable = parsed.get("validation", {}).get("achievable", False)
    if not achievable:
        parsed = {"task": objective, "target": objective, "subtasks": [],
                  "constraints": [], "raw": objective}

    # Build plan: LLM first, then keyword fallback
    plan = None
    plan_source = "keyword"
    if llm_planner.available:
        plan = llm_planner.plan(objective, manifest, recent_memory)
        if plan:
            plan_source = "LLM"
    if plan is None:
        plan = _build_agent_plan(parsed, available, recent_memory)
        plan_source = "keyword"

    # Execute plan steps
    step_faults = 0
    prev_output = None
    sim_accumulator = []
    action_log = []

    for i, action in enumerate(plan, 1):
        tool_name = action["tool"]
        params = action.get("params", {})

        # Pipe previous output
        pipe_key = action.get("pipe_input")
        if pipe_key:
            if sim_accumulator and prev_output is None:
                prev_output = _format_sim_accumulator(sim_accumulator)
                sim_accumulator = []
            if prev_output:
                params[pipe_key] = str(prev_output)
            else:
                action_log.append({"tool": tool_name, "status": "skipped",
                                   "reason": "no input from previous step"})
                continue

        if tool_name != "run_simulation" and sim_accumulator:
            prev_output = _format_sim_accumulator(sim_accumulator)
            sim_accumulator = []

        result = registry.execute(tool_name, params)

        # Fallback — safe write_file handling
        if not result.success and action.get("fallback_tool"):
            fb_tool = action["fallback_tool"]
            fb_params = action.get("fallback_params") or {}
            if tool_name == "write_file" and fb_tool == "execute_shell":
                dirpath = os.path.dirname(params.get("path", ""))
                if dirpath:
                    registry.execute("execute_shell",
                                     {"command": f"mkdir -p {dirpath}"})
                    result = registry.execute("write_file", params)
            else:
                if pipe_key and prev_output:
                    fb_params[pipe_key] = str(prev_output)
                result = registry.execute(fb_tool, fb_params)
                tool_name = fb_tool

        # Correction agent
        if corrector.available and action.get("expected_output"):
            eval_result = corrector.evaluate_and_correct(action, result, registry)
            result = eval_result["final_result"]

        entry = {
            "tool": tool_name,
            "success": result.success,
            "duration_ms": round(result.duration_ms, 1),
        }
        if not result.success:
            step_faults += 1
            entry["error"] = str(result.error)
        else:
            output_str = _summarize_output(result.output)
            entry["output"] = output_str
            if tool_name == "run_simulation":
                sim_accumulator.append(("ok", params.get("scenario", "?"),
                                        result.output))
            prev_output = result.output

        action_log.append(entry)

    # Flush remaining sim results
    if sim_accumulator:
        prev_output = _format_sim_accumulator(sim_accumulator)

    elapsed = time.time() - start_time
    outcome = "SUCCESS" if step_faults == 0 else "DEGRADED"

    # Persist to memory
    memory_entry = {
        "timestamp": time.time(),
        "objective": objective,
        "tool_chain": [a["tool"] for a in plan],
        "outcome": outcome,
        "key_facts": _extract_key_facts(prev_output),
    }
    _save_agent_memory_entry(memory_entry)

    return {
        "status": outcome,
        "plan_source": plan_source,
        "actions": action_log,
        "faults": step_faults,
        "result": str(prev_output)[:2000] if prev_output else None,
        "time_taken": round(elapsed, 2),
        "tool_chain": [a["tool"] for a in plan],
    }


def run_server(args) -> None:
    """HTTP server mode: POST /objective and GET /health endpoints."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from aether.core.tool_registry import ToolRegistry, register_built_tools
    from aether.core.tool_discovery import ToolDiscovery
    from aether.core.llm_planner import LLMPlanner, build_tool_descriptions
    from aether.agents.correction_agent import CorrectionAgent
    from aether.core.goal_parser import GoalParser
    from aether.core.navigation_engine import NavigationEngine
    from aether.core.tool_builder import ToolBuilder

    # ── Capability Discovery ──
    discovery = ToolDiscovery()
    discovery.discover()
    discovery.print_summary()

    registry = ToolRegistry()
    goal_parser = GoalParser()

    # ── Build Tools + Navigation Engine ──
    builder = ToolBuilder(discovery.manifest)
    built_tools = builder.build_all()
    nav_engine = NavigationEngine(discovery.manifest, tools=built_tools)

    manifest = discovery.manifest
    manifest["tool_descriptions"] = build_tool_descriptions(registry)
    manifest["navigation_actions"] = nav_engine.available_actions()
    manifest["navigation_level"] = nav_engine.level
    manifest["available_tools"] = sorted(
        set(manifest.get("available_tools", []) + nav_engine.available_actions()))

    # Register ToolBuilder tools + NavigationEngine actions into ToolRegistry
    # (after manifest enrichment so system_metrics includes full capabilities)
    register_built_tools(registry, built_tools, nav_engine, manifest=manifest)

    llm_planner = LLMPlanner()
    corrector = CorrectionAgent(available_tools=registry.available_tools())
    available = registry.available_tools()

    # Load domain definitions
    defs_path = os.path.join(os.path.dirname(__file__), "context",
                             "aether_definitions.txt")
    if os.path.exists(defs_path):
        with open(defs_path) as f:
            defs_text = f.read()
        summarizer = registry.get("summarize_text")
        if summarizer and hasattr(summarizer, "set_system_context"):
            summarizer.set_system_context(defs_text)

    # Server start time for uptime
    server_start = time.time()

    class AETHERHandler(BaseHTTPRequestHandler):
        """HTTP handler for AETHER server endpoints."""

        def do_POST(self):
            if self.path == "/objective":
                self._handle_objective()
            else:
                self._send_json(404, {"error": f"Not found: {self.path}"})

        def do_GET(self):
            if self.path == "/health":
                self._handle_health()
            else:
                self._send_json(404, {"error": f"Not found: {self.path}"})

        def _handle_objective(self):
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                data = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, ValueError) as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            objective = data.get("objective", "").strip()
            if not objective:
                self._send_json(400, {"error": "Missing 'objective' field"})
                return

            print(f"\n  [SERVER] Received objective: {objective}")
            recent_memory = _load_agent_memory(5)

            try:
                result = _execute_objective(
                    objective, registry, llm_planner, corrector,
                    goal_parser, manifest, recent_memory, available,
                )
                print(f"  [SERVER] Completed: {result['status']} "
                      f"({result['time_taken']}s)")
                self._send_json(200, result)
            except Exception as e:
                print(f"  [SERVER] Error: {e}")
                self._send_json(500, {"error": str(e)})

        def _handle_health(self):
            recent = _load_agent_memory(3)

            # Last SFRI from realworld session logs
            last_sfri = None
            sfri_log = os.path.join("logs", "metrics.json")
            if os.path.exists(sfri_log):
                try:
                    with open(sfri_log) as f:
                        metrics_data = json.load(f)
                    if isinstance(metrics_data, list) and metrics_data:
                        last_sfri = metrics_data[-1].get("SFRI")
                    elif isinstance(metrics_data, dict):
                        last_sfri = metrics_data.get("SFRI")
                except (json.JSONDecodeError, OSError):
                    pass

            health = {
                "status": "healthy",
                "uptime_seconds": round(time.time() - server_start, 1),
                "capabilities": {
                    "score": discovery.score,
                    "available_tools": len(available),
                    "llm_planner": llm_planner.available,
                    "correction_agent": corrector.available,
                },
                "last_sfri": last_sfri,
                "recent_objectives": [
                    {
                        "objective": m.get("objective", "?"),
                        "outcome": m.get("outcome", "?"),
                        "tools": m.get("tool_chain", []),
                    }
                    for m in recent
                ],
            }
            self._send_json(200, health)

        def _send_json(self, code: int, data: dict):
            body = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *log_args):
            # Suppress default stderr logs — we print our own
            pass

    port = args.port
    server = HTTPServer(("0.0.0.0", port), AETHERHandler)

    print("=" * 55)
    print("AETHER v3 — Server Mode")
    print("=" * 55)
    print(f"  Port:             {port}")
    print(f"  LLM Planner:      {'active' if llm_planner.available else 'keyword fallback'}")
    print(f"  Correction Agent: {'active' if corrector.available else 'inactive'}")
    print(f"  Available Tools:  {len(available)}")
    print(f"  Capability Score: {discovery.score}/100")
    print()
    print(f"  POST /objective   — Submit {{\"objective\": \"...\"}}")
    print(f"  GET  /health      — System status + recent objectives")
    print()
    print(f"  Listening on http://0.0.0.0:{port}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  Server stopped.")
        server.server_close()


def main():
    from aether.core.banner import print_banner
    print_banner()

    parser = argparse.ArgumentParser(description="AETHER v3 Agent")
    parser.add_argument("--mode", type=str, default="sim",
                        choices=["sim", "agent", "realworld", "server"],
                        help="sim = simulation (default), agent = real tool execution, "
                             "realworld = live hardware with webcam + system metrics, "
                             "server = HTTP server on --port")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for server mode (default: 8080)")
    parser.add_argument("--task", type=str, default="navigate to target",
                        help="Natural language task description (sim mode)")
    parser.add_argument("--robot", type=str, default="rover_v1",
                        choices=["rover_v1", "drone_v1"])
    parser.add_argument("--faults", type=str, default="disabled",
                        choices=["disabled", "enabled", "heavy"])
    parser.add_argument("--scenario", type=str, default="simple",
                        help="Scenario name (simple/obstacles/imu_fault/battery/compound/fault_heavy)")
    parser.add_argument("--render", action="store_true", help="ASCII render each step")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plots", action="store_true", help="Generate plots after run")
    parser.add_argument("--no-learning", action="store_true",
                        help="Disable online learning (use fixed weights for controlled runs)")
    parser.add_argument("--continuous", action="store_true",
                        help="Run indefinitely in realworld mode (Ctrl+C to stop)")
    parser.add_argument("--auto-install", action="store_true",
                        help="Install missing libraries without asking")
    parser.add_argument("--auto-update", action="store_true",
                        help="Update without asking")
    parser.add_argument("--no-install", action="store_true",
                        help="Skip install prompt entirely")
    parser.add_argument("--no-update", action="store_true",
                        help="Skip update check")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run full hardware calibration wizard")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force re-calibration even if profile exists")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Auto-calibrate using camera only, no questions")
    args = parser.parse_args()

    setup_logging(args.verbose)
    silence_http_logging()

    # ── Calibration: handle before any mode dispatch ──────────────────
    if args.calibrate or args.recalibrate:
        from aether.core.tool_discovery import ToolDiscovery
        from aether.core.calibration import CalibrationWizard

        discovery = ToolDiscovery()
        discovery.discover()
        discovery.print_summary()

        auto_mode = getattr(args, "auto_calibrate", False)
        wizard = CalibrationWizard(
            discovery.manifest,
            auto_mode=auto_mode,
        )
        profile = wizard.run()
        print(f"\n[Calibration] Saved profile: {profile.get('robot_name')}")
        print(f"[Calibration] Type: {profile.get('robot_type')}")
        print(f"[Calibration] Capabilities: {', '.join(profile.get('capabilities', []))}")
        return  # exit — do not run any simulation or agent mode

    if args.mode == "agent":
        run_agent(args)
    elif args.mode == "realworld":
        run_realworld(args)
    elif args.mode == "server":
        run_server(args)
    else:
        run_sim(args)


if __name__ == "__main__":
    main()
