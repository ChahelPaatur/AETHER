<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Raspberry%20Pi%20%7C%20Jetson-lightgrey?style=flat-square" alt="Platforms">
  <img src="https://img.shields.io/badge/version-3.0-red?style=flat-square" alt="v3.0">
</p>

<pre align="center">
    ████████████████████████████████
   █<span style="color:red">██████████████████████████████</span>█
  █<span style="color:red">████</span>╔═══════════════════╗<span style="color:red">████</span>█
  █<span style="color:red">████</span>║  <b>▄████▄   ▄████▄</b>  ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║ <b>████████████████</b> ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║ ██<span style="color:red">██████████</span>██ ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║ █▀▀████████▀▀█ ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║  <b>▀████▀   ▀████▀</b>  ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║  ━━━━━━━━━━━━━━━  ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║  <b>▌ AETHER  v3 ▐</b>  ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>║  ━━━━━━━━━━━━━━━  ║<span style="color:red">████</span>█
  █<span style="color:red">████</span>╚═══════════════════╝<span style="color:red">████</span>█
   █<span style="color:red">██████████████████████████████</span>█
    ████████████████████████████████
</pre>

<h1 align="center">AETHER v3</h1>
<h3 align="center">Adaptive Embodied Task Hierarchy for Executable Robotics</h3>
<p align="center"><i>DRL-First Hybrid FDIR &middot; Multi-Agent &middot; Self-Correcting</i></p>

---

AETHER is a multi-agent robotics framework that detects, diagnoses, and recovers from hardware faults in real time using a Deep Reinforcement Learning-first approach. It auto-discovers whatever hardware is connected — webcam, GPIO motors, flight controller — builds a capability manifest, and constructs a complete autonomy stack from planning through execution. A PPO neural network serves as the primary fault detector, backed by rule-based safety checks and temporal validation, achieving perfect detection and recovery rates on real hardware across thousands of operational steps.

---

## Quick Start

```bash
# 1. Install dependencies (auto-detects your platform)
bash install.sh

# 2. Run the simulation with fault injection and performance plots
python main.py --mode sim --faults enabled --scenario compound --plots

# 3. Run on real hardware (webcam + system telemetry)
python main.py --mode realworld --continuous
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

| Metric | Simulation (5,000 runs) | Real Hardware (6,023 steps) |
|--------|------------------------:|---------------------------:|
| **SFRI** | 60.07 | **69.99** |
| Detection Rate | 85–95% | **100%** |
| Recovery Rate | 80–90% | **100%** |
| False Positive Rate | 2–5% | **0.0%** |
| MTTD (steps) | 3–8 | **< 2** |
| MTTR (steps) | 5–15 | **< 5** |

> **SFRI** (Stability Fault Recovery Index) = 35&times;DR + 25&times;(1 &minus; MTTR/max_steps) + 10&times;RR &minus; 30&times;FPR
> Range: 0&ndash;70. Higher is better.

Real hardware outperforms simulation because the physical system encounters genuine sensor noise that the PPO network learns to distinguish from actual faults, while simulation injects idealized fault signatures that can mislead the temporal validator.

---

## Supported Hardware

| Platform | Camera | Motors | Flight Controller | Notes |
|----------|--------|--------|-------------------|-------|
| **Laptop + USB webcam** | OpenCV (cv2) | &mdash; | &mdash; | Level 1 autonomy, development & testing |
| **Raspberry Pi + Pi Camera** | picamera2 / cv2 | &mdash; | &mdash; | Level 1, headless visual perception |
| **Raspberry Pi + GPIO motors** | picamera2 / cv2 | RPi.GPIO / gpiozero | &mdash; | Level 2, ground vehicle navigation |
| **Raspberry Pi + SpeedyBee FC** | picamera2 / cv2 | &mdash; | MAVLink (pymavlink) | Level 3, autonomous drone flight |

Additional supported interfaces: I2C sensors (smbus2), serial UART, ultrasonic rangefinders, IMU, temperature probes, LiDAR, battery monitoring.

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

## Project Structure

```
aether/
├── core/               # Discovery, planning, execution, metrics
│   ├── tool_discovery   # Hardware/software capability probing
│   ├── tool_builder     # Construct tools from manifest
│   ├── tool_registry    # Register executable tools
│   ├── navigation_engine# 3-level hardware-agnostic navigation
│   ├── llm_planner      # Claude-based task planning
│   ├── calibration      # Interactive hardware calibration
│   ├── metrics          # SFRI, MTTD, MTTR tracking
│   └── visualizer       # Plot generation
├── agents/              # Domain-specific agents
│   ├── fault_agent      # DRL-First Hybrid FDIR (PPO)
│   ├── perception_agent # 15-dim observation construction
│   ├── adaptation_agent # Fault recovery actions
│   ├── camera_agent     # Visual processing
│   └── ...              # power, thermal, navigation, memory
├── simulation/          # Physics environment, scenarios
├── faults/              # Fault injection & detection
├── adapters/            # Hardware abstraction (rover, drone)
configs/                 # Robot profiles (rover_v1, drone_v1)
weights/                 # Pre-trained PPO network weights
tests/                   # Test suite
```

---

## Citation

If you use AETHER in your research, please cite:

```bibtex
@software{aether2026,
  title     = {AETHER: Adaptive Embodied Task Hierarchy for Executable Robotics},
  author    = {Paatur, Chahel},
  year      = {2026},
  version   = {3.0},
  note      = {DRL-First Hybrid FDIR with multi-agent auto-configuration},
}
```
