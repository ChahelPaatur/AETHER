#!/bin/bash
set -e

echo "=================================================="
echo "AETHER v3 — Installation"
echo "=================================================="

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "armv7l" ]; then
    PLATFORM="raspberry_pi"
    echo "Platform: Raspberry Pi ($ARCH)"
elif [ "$OS" = "Darwin" ]; then
    PLATFORM="macos"
    echo "Platform: macOS ($ARCH)"
else
    PLATFORM="linux"
    echo "Platform: Linux ($ARCH)"
fi

# Create virtual environment if not exists
if [ ! -d "aether_env" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv aether_env
fi

# Activate
source aether_env/bin/activate 2>/dev/null || source aether_env/Scripts/activate 2>/dev/null
echo "Virtual environment: $(which python3)"

# Install base dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install anthropic psutil opencv-python numpy requests flask -q

# Platform specific
if [ "$PLATFORM" = "raspberry_pi" ]; then
    echo "Installing Pi-specific packages..."
    pip install picamera2 RPi.GPIO gpiozero smbus2 -q 2>/dev/null || echo "  Some Pi packages skipped (install manually if needed)"
fi

# Ensure logs directory exists
mkdir -p logs/captures

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs) 2>/dev/null
    fi
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    read -p "Enter your Anthropic API key (or press Enter to skip): " API_KEY
    if [ -n "$API_KEY" ]; then
        echo "ANTHROPIC_API_KEY=$API_KEY" > .env
        export ANTHROPIC_API_KEY=$API_KEY
        echo "API key saved to .env"
    else
        echo "Skipped — set ANTHROPIC_API_KEY later for LLM features"
    fi
fi

# Run tool discovery
echo ""
echo "Running capability discovery..."
python3 -c "
import sys; sys.path.insert(0, '.')
from aether.core.tool_discovery import ToolDiscovery
td = ToolDiscovery()
td.discover()
m = td.manifest
hw = m.get('hardware', {})
sw = m.get('software', {})
net = m.get('network', {})
score = m.get('capability_score', 0)
hw_list = [k for k, v in hw.items() if isinstance(v, dict) and v.get('available')]
sw_list = [k for k, v in sw.items() if v]
print(f'  Capability score: {score}/100')
print(f'  Hardware: {', '.join(hw_list) if hw_list else 'none detected'}')
print(f'  Software: {len(sw_list)} packages')
print(f'  Internet: {\"yes\" if net.get(\"internet\") else \"no\"}')
"

# Health checks
echo ""
echo "Running health checks..."
PASSED=0
FAILED=0

# Check 1: Core imports
if python3 -c "import anthropic, psutil, cv2, numpy" 2>/dev/null; then
    echo "  [PASS] Core imports (anthropic, psutil, cv2, numpy)"
    PASSED=$((PASSED+1))
else
    echo "  [FAIL] Core imports — run: pip install anthropic psutil opencv-python numpy"
    FAILED=$((FAILED+1))
fi

# Check 2: Camera
if python3 -c "
import cv2
cap = cv2.VideoCapture(0)
ok, _ = cap.read()
cap.release()
assert ok
" 2>/dev/null; then
    echo "  [PASS] Camera accessible"
    PASSED=$((PASSED+1))
else
    echo "  [SKIP] Camera not connected (level 1 nav requires camera)"
fi

# Check 3: API key
if python3 -c "
import os
key = os.environ.get('ANTHROPIC_API_KEY', '')
if not key:
    with open('.env') as f:
        for line in f:
            if line.startswith('ANTHROPIC_API_KEY='):
                key = line.split('=', 1)[1].strip()
assert len(key) > 10
" 2>/dev/null; then
    echo "  [PASS] Anthropic API key configured"
    PASSED=$((PASSED+1))
else
    echo "  [FAIL] API key missing — LLM planner and correction agent disabled"
    FAILED=$((FAILED+1))
fi

# Check 4: File write
if python3 -c "
import os
os.makedirs('logs', exist_ok=True)
open('logs/.write_test', 'w').write('ok')
os.remove('logs/.write_test')
" 2>/dev/null; then
    echo "  [PASS] File write permissions"
    PASSED=$((PASSED+1))
else
    echo "  [FAIL] Cannot write to logs/ directory"
    FAILED=$((FAILED+1))
fi

# Check 5: Tool discovery
if python3 -c "
import sys; sys.path.insert(0, '.')
from aether.core.tool_discovery import ToolDiscovery
td = ToolDiscovery()
td.discover()
assert td.manifest.get('capability_score', 0) > 0
" 2>/dev/null; then
    echo "  [PASS] Tool discovery engine"
    PASSED=$((PASSED+1))
else
    echo "  [FAIL] Tool discovery failed"
    FAILED=$((FAILED+1))
fi

# Check 6: Simulation
if python3 -c "
import sys; sys.path.insert(0, '.')
from aether.simulation.environment import SimulationEnvironment
env = SimulationEnvironment(seed=42)
obs = env.reset()
assert len(obs) == 15
" 2>/dev/null; then
    echo "  [PASS] Simulation environment"
    PASSED=$((PASSED+1))
else
    echo "  [SKIP] Simulation environment not available"
fi

# Create launch script
cat > launch_aether.sh << 'LAUNCH'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source aether_env/bin/activate 2>/dev/null || source aether_env/Scripts/activate 2>/dev/null
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null
fi
python3 main.py --mode agent "$@"
LAUNCH
chmod +x launch_aether.sh

# Summary
echo ""
echo "=================================================="
echo "  Checks: $PASSED passed, $FAILED failed"
if [ $FAILED -eq 0 ]; then
    echo "  AETHER READY"
    echo ""
    echo "  Run:  bash launch_aether.sh"
    echo "  Or:   bash launch_aether.sh --mode server --port 8080"
else
    echo "  $FAILED check(s) failed — review above"
    echo ""
    echo "  AETHER will still run but some features may be degraded."
    echo "  Run:  bash launch_aether.sh"
fi
echo "=================================================="
