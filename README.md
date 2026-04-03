<p align="center">
  <img src="https://img.shields.io/pypi/v/aether-robotics?style=flat-square&color=blue" alt="PyPI">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT">
  <img src="https://img.shields.io/github/stars/ChahelPaatur/AETHER?style=flat-square" alt="GitHub Stars">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Raspberry%20Pi%20%7C%20Jetson-lightgrey?style=flat-square" alt="Platforms">
  <img src="https://img.shields.io/badge/version-3.0-red?style=flat-square" alt="v3.0">
</p>

```
    ████████████████████████████████
   █░██████████████████████████████░█
  █░████╔═══════════════════╗████░█
  █░████║  ▄████▄   ▄████▄  ║████░█
  █░████║ ████████████████ ║████░█
  █░████║ ██░██████████░██ ║████░█
  █░████║ █▀▀████████▀▀█ ║████░█
  █░████║  ▀████▀   ▀████▀  ║████░█
  █░████║▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄║████░█
  █░████║  ━━━━━━━━━━━━━━━  ║████░█
  █░████║  ▌ AETHER  v3 ▐  ║████░█
  █░████║  ━━━━━━━━━━━━━━━  ║████░█
  █░████╚═══════════════════╝████░█
   █░██████████████████████████████░█
    ████████████████████████████████
```

<h1 align="center">AETHER v3</h1>
<h3 align="center">Adaptive Embodied Task Hierarchy for Executable Robotics</h3>
<p align="center"><i>DRL-First Hybrid FDIR &middot; Multi-Agent &middot; Self-Correcting</i></p>

---

AETHER is a multi-agent robotics framework that detects, diagnoses, and recovers from hardware faults in real time using a Deep Reinforcement Learning-first approach. It auto-discovers whatever hardware is connected — webcam, GPIO motors, flight controller — builds a capability manifest, and constructs a complete autonomy stack from planning through execution. A PPO neural network serves as the primary fault detector, backed by rule-based safety checks and temporal validation, achieving perfect detection and recovery rates on real hardware across thousands of operational steps.

---

## Quick Start

```bash
# 1. Install
pip install aether-robotics

# 2. Calibrate your hardware
aether --calibrate

# 3. Run in agent mode
aether --mode agent
```

---

## What It Does

### DRL-First Hybrid FDIR

Traditional fault detection relies on hand-written threshold rules that break when conditions change. AETHER inverts this: a PPO neural network (15-dim observation &rarr; 64 &rarr; 64 &rarr; 8 fault classes) is the **primary** detector, with rule-based checks as a safety backup. The network self-bootstraps from scratch using the rule detector as a teacher, then surpasses it through online learning.

The detection pipeline runs every step:

1. **PPO Network** infers fault class probabilities from the 15-dimensional observation vector (battery, IMU, temperature, obstacle distances, mission progress)
2. **Temporal Validation** filters transient spikes to prevent false positives
3. **Confidence Arbitration** (threshold &tau;=0.12) decides whether to alert
4. **Critical Bypass** (&sigma;&ge;0.80) escalates severe faults directly, skipping temporal filtering

Fault classes: `SENSOR_FAILURE`, `ACTUATOR_DEGRADATION`, `POWER_CRITICAL`, `THERMAL_ANOMALY`, `IMU_DRIFT`, `INTERMITTENT_FAULT`, `SAFE_MODE`

### Auto-Configuration

AETHER discovers its own hardware at startup — no config files required. `ToolDiscovery` probes for cameras (OpenCV, picamera2), GPIO pins (RPi.GPIO, gpiozero), flight controllers (MAVLink over serial/USB), I2C sensors, network interfaces, and AI models (YOLOv8, Claude API). The result is a capability manifest that drives everything downstream: `ToolBuilder` constructs only the tools that will work, `NavigationEngine` selects the correct autonomy level, and `LLMPlanner` avoids planning with unavailable hardware.

Three capability levels adapt automatically:

| Level | Hardware | Capabilities |
|-------|----------|-------------|
| **1** | Camera only | Visual scan, object detection, scene description |
| **2** | Camera + GPIO motors | Level 1 + navigation, obstacle avoidance, color tracking |
| **3** | Camera + MAVLink FC | Level 2 + takeoff, landing, waypoint navigation, RTL |

---

## Benchmark Results

| Platform | SFRI | Detection Rate | Recovery Rate | FPR |
|----------|-----:|---------------:|--------------:|----:|
| Laptop (simulation, 5,000 runs) | **60.07** | 74.7% | 80–90% | 2–5% |
| Raspberry Pi (real hardware, 6,023 steps) | **69.99** | 100% | 100% | 0.0% |
| SpeedyBee Drone | TBD | TBD | TBD | TBD |

> **SFRI** (Stability Fault Recovery Index) = 35&times;DR + 25&times;(1 &minus; MTTR/max_steps) + 10&times;RR &minus; 30&times;FPR
> Range: 0&ndash;70. Higher is better.

Real hardware outperforms simulation because the physical system encounters genuine sensor noise that the PPO network learns to distinguish from actual faults, while simulation injects idealized fault signatures that can mislead the temporal validator.

---

## Supported Hardware

| Platform | Details |
|----------|---------|
| :computer: **Laptop + USB webcam** | Level 1 autonomy — development & testing |
| :strawberry: **Raspberry Pi + Pi Camera** | Level 1 — headless visual perception |
| :strawberry: **Raspberry Pi + GPIO motors** | Level 2 — ground vehicle navigation |
| :helicopter: **Raspberry Pi + SpeedyBee FC** | Level 3 — autonomous drone flight |

Additional: I2C sensors, serial UART, ultrasonic rangefinders, IMU, temperature probes, LiDAR, battery monitoring.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        AETHER v3 PIPELINE                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ ToolDiscovery │───▶│ ToolBuilder  │───▶│  ToolRegistry    │   │
│  │ probe hw/sw   │    │ build tools  │    │  register all    │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│         │                                          │             │
│         ▼                                          ▼             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  Calibration  │    │  GoalParser  │───▶│   LLMPlanner     │   │
│  │  Wizard       │    │  NL → struct │    │  Claude / kw     │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │             │
│                                                    ▼             │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                   EXECUTION LOOP                          │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐   │   │
│  │  │ Navigation  │  │  Perception  │  │  Correction    │   │   │
│  │  │ Engine      │  │  Agent       │  │  Agent         │   │   │
│  │  │ L1/L2/L3    │  │  15-dim obs  │  │  verify steps  │   │   │
│  │  └──────┬──────┘  └──────┬───────┘  └────────────────┘   │   │
│  │         │                │                                │   │
│  │         ▼                ▼                                │   │
│  │  ┌─────────────────────────────────────────────────────┐  │   │
│  │  │              FAULT AGENT (DRL-First)                │  │   │
│  │  │  PPO Network ──▶ Temporal Validation ──▶ Response   │  │   │
│  │  │  (15→64→64→8)    confidence filter      adaptation  │  │   │
│  │  │       ▲               ▲                     │       │  │   │
│  │  │       │               │                     ▼       │  │   │
│  │  │  Rule Backup    Memory Agent         Recovery       │  │   │
│  │  │  (safety net)   (experience)         Action         │  │   │
│  │  └─────────────────────────────────────────────────────┘  │   │
│  └───────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  MetricsTracker  ──▶  Visualizer  ──▶  logs/plots/      │    │
│  │  SFRI · MTTD · MTTR · DR · RR · FPR · reward curves    │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Calibration

The `CalibrationWizard` runs when new hardware is detected that hasn't been profiled before. It walks through 7 phases:

1. **Component Identification** — asks robot type (ground, aerial, arm, aquatic, custom)
2. **Motor Mapping** — tests each channel at low power, records what physically moved
3. **Camera-Assisted Verification** — uses optical flow to confirm motor-to-function assignments
4. **Environment Mapping** — drives a grid pattern to build a 100x100 occupancy grid
5. **Safety Limits** — sets per-component speed/angle/power limits
6. **Capability Generation** — saves a JSON robot profile to `configs/`
7. **Action Generation** — builds the correct action set for the detected robot type

Run with `--auto-calibrate` for headless (no interactive prompts) calibration using camera feedback only.

---

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--mode {sim,agent,realworld,server}` | `sim` | Operating mode |
| `--task TEXT` | `"navigate to target"` | Natural language task objective |
| `--robot {rover_v1,drone_v1}` | `rover_v1` | Robot platform configuration |
| `--scenario {simple,obstacles,imu_fault,battery,compound,fault_heavy}` | `simple` | Simulation scenario |
| `--faults {disabled,enabled,heavy}` | `disabled` | Fault injection level |
| `--max-steps N` | `300` | Maximum steps per episode |
| `--seed N` | `42` | Random seed |
| `--port N` | `8080` | HTTP server port (server mode) |
| `--render` | off | ASCII render each simulation step |
| `--plots` | off | Generate matplotlib plots after run |
| `--verbose` | off | Debug logging |
| `--continuous` | off | Run indefinitely in realworld mode |
| `--no-learning` | off | Disable online PPO learning (fixed weights) |
| `--calibrate` | off | Run hardware calibration wizard |
| `--recalibrate` | off | Force re-calibration over existing profile |
| `--auto-calibrate` | off | Camera-only auto calibration (no prompts) |
| `--auto-install` | off | Install missing packages without asking |
| `--auto-update` | off | Update without asking |
| `--no-install` | off | Skip install prompts |
| `--no-update` | off | Skip update check |

---

## Citation

If you use AETHER in your research, please cite:

```bibtex
@software{aether2026,
  title     = {AETHER: Adaptive Embodied Task Hierarchy for Executable Robotics},
  author    = {Paatur, Chahel},
  year      = {2026},
  version   = {3.0},
  url       = {https://github.com/ChahelPaatur/AETHER},
  note      = {DRL-First Hybrid FDIR with multi-agent auto-configuration},
}
```
