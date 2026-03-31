"""
AETHER v3 Experiment Runner.
Usage:
  python run_experiments.py --runs 100
  python run_experiments.py --runs 1000 --parallel --workers 8
  python run_experiments.py --runs 100 --plots
  python run_experiments.py --plot-only logs/experiment_XYZ.json
  python run_experiments.py --stress-agent power
  python run_experiments.py --stress-agent navigation --runs 50
"""
import argparse
import csv
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Domain agent → fault catalog keys that target its subsystem
STRESS_FAULT_MAP = {
    "camera":     ["sensor_dropout_camera", "intermittent"],
    "movement":   ["actuator_degrade"],
    "power":      ["battery_drain", "bus_voltage_drop"],
    "thermal":    ["thermal_spike"],
    "navigation": ["imu_drift", "sensor_noise_imu"],
}


def _run_single(args_tuple) -> Dict:
    """Worker function — must be top-level for multiprocessing pickle."""
    (run_id, task, scenario_name, robot, fault_mode, seed, max_steps, stress_agent, no_learning) = args_tuple
    try:
        from aether.simulation.environment import SimulationEnvironment
        from aether.simulation.scenarios import get_scenario, ScenarioGenerator
        from aether.faults.fault_injector import FaultInjector
        from aether.core.message_bus import MessageBus
        from aether.agents.task_manager import TaskManagerAgent

        env = SimulationEnvironment(seed=seed)
        bus = MessageBus()
        # Log scenario being loaded for diagnostics
        if run_id < 5 or run_id % 500 == 0:
            print(f"[Run {run_id}] Loading scenario='{scenario_name}' fault_mode='{fault_mode}' seed={seed}", flush=True)

        if stress_agent:
            fault_injector = FaultInjector(fault_probability=0.0, seed=seed)
            targeted_faults = STRESS_FAULT_MAP.get(stress_agent, [])
            for fault_type in targeted_faults:
                fault_injector.schedule(
                    fault_type=fault_type,
                    start_step=5,
                    subsystem=stress_agent,
                    severity=0.8,
                )
        else:
            fault_prob = {"disabled": 0.0, "enabled": 0.015, "heavy": 0.05}.get(fault_mode, 0.0)
            fault_injector = FaultInjector(fault_probability=fault_prob, seed=seed)

        config_path = os.path.join(os.path.dirname(__file__), "configs", f"{robot}.json")
        agent = TaskManagerAgent(env=env, config_path=config_path, bus=bus,
                                  log_dir="logs", seed=seed, fault_injector=fault_injector,
                                  no_learning=no_learning)

        scenario = get_scenario(scenario_name)
        if scenario is None:
            print(f"[Run {run_id}] WARNING: scenario '{scenario_name}' not found, using random", flush=True)
            scenario = ScenarioGenerator(seed).random_scenario("medium")

        if not scenario.get("name"):
            print(f"[Run {run_id}] WARNING: scenario has no 'name' field: {list(scenario.keys())}", flush=True)

        effective_max_steps = scenario.get("max_steps", max_steps)
        result = agent.run_episode(task_text=task, scenario=scenario, max_steps=effective_max_steps)
        result["run_id"]      = run_id
        result["fault_mode"]  = fault_mode
        result["task_text"]   = task
        result["stress_agent"] = stress_agent or ""
        return result
    except Exception as e:
        import traceback
        print(f"[Run {run_id}] ERROR in scenario '{scenario_name}': {e}", flush=True)
        traceback.print_exc()
        return {
            "run_id": run_id, "task": task, "scenario": scenario_name, "robot": robot,
            "fault_mode": fault_mode, "stress_agent": stress_agent or "",
            "success": False, "steps_to_completion": 0,
            "SFRI": 0.0, "MTTD": 0.0, "MTTR": 0.0, "detection_rate": 0.0,
            "recovery_rate": 0.0, "false_positive_rate": 0.0,
            "faults_injected": 0, "faults_detected": 0, "faults_recovered": 0,
            "false_positives": 0, "adaptation_latency": 0.0,
            "error": str(e),
        }


TASK_VARIANTS = [
    "navigate to target",
    "follow target while avoiding obstacles",
    "navigate to waypoint B",
]
SCENARIO_VARIANTS = ["simple", "obstacles", "imu_fault", "battery", "compound"]
FAULT_MODES = ["disabled", "enabled", "heavy"]


def run_experiments(
    num_runs: int,
    parallel: bool = False,
    workers: int = 4,
    output_path: Optional[str] = None,
    plot: bool = False,
    max_steps: int = 300,
    stress_agent: Optional[str] = None,
    no_learning: bool = False,
) -> Dict:
    rng = np.random.default_rng(42)
    jobs = []
    for i in range(num_runs):
        task       = TASK_VARIANTS[i % len(TASK_VARIANTS)]
        scenario   = SCENARIO_VARIANTS[i % len(SCENARIO_VARIANTS)]
        fault_mode = "heavy" if stress_agent else FAULT_MODES[i % len(FAULT_MODES)]
        seed       = int(rng.integers(0, 2**31))
        jobs.append((i, task, scenario, "rover_v1", fault_mode, seed, max_steps, stress_agent, no_learning))

    label = f"stress:{stress_agent}" if stress_agent else ("parallel, " + str(workers) + " workers" if parallel else "sequential")
    print(f"Running {num_runs} simulations ({label})...")
    if stress_agent:
        faults = ", ".join(STRESS_FAULT_MAP.get(stress_agent, []))
        print(f"Stress target: {stress_agent.upper()} subsystem | Faults: {faults}")

    t_start = time.time()
    results = []

    if parallel and num_runs > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_run_single, jobs)):
                results.append(result)
                status = "." if result.get("success") else "F"
                sfri = result.get("SFRI", 0.0)
                print(f"[{i+1:5d}/{num_runs}] {status} {result.get('task_text','')[:30]:<30} "
                      f"| {result.get('fault_mode',''):<8} | SFRI: {sfri:5.1f}", flush=True)
    else:
        for i, job in enumerate(jobs):
            result = _run_single(job)
            results.append(result)
            status = "." if result.get("success") else "F"
            sfri = result.get("SFRI", 0.0)
            if i % max(1, num_runs // 20) == 0 or i == num_runs - 1:
                pct = (i + 1) / num_runs * 100
                bar = "█" * int(pct // 5) + "░" * (20 - int(pct // 5))
                print(f"\r|{bar}| {i+1}/{num_runs} — SFRI: {sfri:.1f}", end="", flush=True)
    print()

    elapsed = time.time() - t_start
    from aether.core.metrics import MetricsTracker
    metrics_agg = MetricsTracker.aggregate(results)
    fault_breakdown = _compute_fault_breakdown(results)

    print(f"\n{'='*55}")
    if stress_agent:
        print(f"=== STRESS TEST: {stress_agent.upper()} AGENT ({elapsed:.1f}s) ===")
    else:
        print(f"=== EXPERIMENT RESULTS ({elapsed:.1f}s) ===")
    print(f"{'='*55}")
    _print_metric(metrics_agg, "SFRI",               "SFRI Score")
    _print_metric(metrics_agg, "detection_rate",     "Detection Rate",    pct=True)
    _print_metric(metrics_agg, "recovery_rate",      "Recovery Rate",     pct=True)
    _print_metric(metrics_agg, "MTTD",               "MTTD (steps)")
    _print_metric(metrics_agg, "MTTR",               "MTTR (steps)")
    _print_metric(metrics_agg, "false_positive_rate","False Positive Rate", pct=True)
    _print_metric(metrics_agg, "adaptation_latency", "Adaptation Latency (steps)")
    error_count = sum(1 for r in results if r.get("error"))
    success_rate = sum(1 for r in results if r.get("success")) / len(results) * 100
    steps_vals   = [r["steps_to_completion"] for r in results if r.get("steps_to_completion", 0) > 0]
    mean_steps   = sum(steps_vals) / len(steps_vals) if steps_vals else 0
    print(f"\n  Success Rate:    {success_rate:.1f}%")
    print(f"  Mean Steps:      {mean_steps:.1f}")
    if error_count:
        print(f"  ERRORS:          {error_count}/{len(results)} runs failed with exceptions!")
        error_msgs = {}
        for r in results:
            if r.get("error"):
                msg = r["error"][:80]
                error_msgs[msg] = error_msgs.get(msg, 0) + 1
        for msg, count in sorted(error_msgs.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{count}x] {msg}")

    if fault_breakdown:
        label = "Domain Resilience by Subsystem" if stress_agent else "Fault Breakdown"
        print(f"\n{label}:")
        for ft, info in fault_breakdown.items():
            det = info["detected"] / max(info["injected"], 1) * 100
            rec = info["recovered"] / max(info["detected"], 1) * 100
            print(f"  {ft:<25} detected {det:.1f}% | recovered {rec:.1f}%")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"stress_{stress_agent}_{ts}" if stress_agent else f"experiment_{ts}"
    out = output_path or os.path.join("logs", f"{prefix}.json")
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    output = {
        "timestamp": ts, "num_runs": num_runs, "metrics": metrics_agg,
        "success_rate": success_rate / 100, "mean_steps": mean_steps,
        "fault_breakdown": fault_breakdown, "runs": results,
    }
    if stress_agent:
        output["stress_agent"] = stress_agent
        output["stress_faults"] = STRESS_FAULT_MAP.get(stress_agent, [])

    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults → {out}")

    csv_path = out.replace(".json", ".csv")
    if results:
        fieldnames = [k for k in results[0] if isinstance(results[0][k], (int, float, bool, str))]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"CSV     → {csv_path}")

    if plot:
        try:
            from aether.core.visualizer import Visualizer
            plot_dir = os.path.join("logs", "plots", ts)
            Visualizer().generate_full_report(output, plot_dir)
            print(f"Plots   → {plot_dir}/")
        except ImportError:
            print("matplotlib unavailable — skipping plots.")

    return output


def _print_metric(agg: Dict, key: str, label: str, pct: bool = False) -> None:
    if key not in agg:
        return
    mean = agg[key]["mean"] * (100 if pct else 1)
    std  = agg[key]["std"]  * (100 if pct else 1)
    suffix = "%" if pct else ""
    print(f"  {label:<25} {mean:.2f}{suffix} ± {std:.2f}{suffix}")


def _compute_fault_breakdown(results: List[Dict]) -> Dict:
    breakdown = {}
    for r in results:
        fi  = r.get("faults_injected", 0)
        fd  = r.get("faults_detected", 0)
        fr  = r.get("faults_recovered", 0)
        key = r.get("stress_agent") or r.get("fault_mode", "unknown")
        if key not in breakdown:
            breakdown[key] = {"injected": 0, "detected": 0, "recovered": 0}
        breakdown[key]["injected"]  += fi
        breakdown[key]["detected"]  += fd
        breakdown[key]["recovered"] += fr
    return breakdown


def _run_auto_calibrate(no_learning: bool = False) -> None:
    """
    Run 20 warmup episodes on simple + obstacles scenarios to prime the FaultAgent
    weights and give the MemoryAgent real successful experience before main batch.
    """
    print("[AutoCalibrate] Running 20 warmup episodes (simple × 10, obstacles × 10)...")
    cal_scenarios = ["simple"] * 10 + ["obstacles"] * 10
    successes = 0
    for i, scenario_name in enumerate(cal_scenarios):
        seed = 9000 + i
        result = _run_single((
            i, "navigate to target", scenario_name,
            "rover_v1", "disabled", seed, 300, None, no_learning
        ))
        if result.get("success"):
            successes += 1
        print(f"\r[AutoCalibrate] Warmup {i+1:2d}/20 — "
              f"{'SUCCESS' if result.get('success') else 'FAIL   '} "
              f"(SFRI {result.get('SFRI', 0):.1f})", end="", flush=True)
    print(f"\n[AutoCalibrate] Warmup complete. "
          f"Strategy scores initialized from {successes}/20 successful runs.")


def main():
    parser = argparse.ArgumentParser(description="AETHER v3 Experiment Runner")
    parser.add_argument("--runs",           type=int,  default=20)
    parser.add_argument("--parallel",       action="store_true")
    parser.add_argument("--workers",        type=int,  default=4)
    parser.add_argument("--output",         type=str,  default=None)
    parser.add_argument("--plots",          action="store_true")
    parser.add_argument("--max-steps",      type=int,  default=300)
    parser.add_argument("--stress-agent",   type=str,  default=None,
                        choices=["camera", "movement", "power", "thermal", "navigation"],
                        help="Run 100 sims with heavy fault injection targeting one domain agent")
    parser.add_argument("--no-learning",    action="store_true",
                        help="Disable online learning (fixed weights, for controlled experiments)")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Run 20 warmup episodes before main batch to prime weights and strategy scores")
    parser.add_argument("--plot-only",      type=str,  default=None,
                        help="Path to existing JSON results file — replot without running")
    args = parser.parse_args()

    if args.plot_only:
        with open(args.plot_only) as f:
            data = json.load(f)
        from aether.core.visualizer import Visualizer
        plot_dir = os.path.join("logs", "plots", datetime.now().strftime("%Y%m%d_%H%M%S"))
        Visualizer().generate_full_report(data, plot_dir)
        print(f"Plots saved to: {plot_dir}/")
        return

    if args.auto_calibrate:
        _run_auto_calibrate(no_learning=args.no_learning)

    num_runs = 100 if args.stress_agent else args.runs

    run_experiments(
        num_runs=num_runs,
        parallel=args.parallel,
        workers=args.workers,
        output_path=args.output,
        plot=args.plots,
        max_steps=args.max_steps,
        stress_agent=args.stress_agent,
        no_learning=args.no_learning,
    )


if __name__ == "__main__":
    main()
