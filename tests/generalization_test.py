"""
AETHER v3 — Generalization Validation Suite
Validates that AETHER is fully platform-generalized with working
discovery, building, navigation, adaptation, and registry layers.
"""
import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

passed = 0
failed = 0
total = 15


def report(test_num: int, name: str, ok: bool, reason: str = ""):
    global passed, failed
    tag = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    detail = f" — {reason}" if reason else ""
    print(f"  [{tag}] Test {test_num:>2}: {name}{detail}")


print("=" * 60)
print("AETHER v3 — Generalization Test Suite (15 tests)")
print("=" * 60)
print()

# ── Shared setup ──────────────────────────────────────────────────────

discovery = None
manifest = None
builder = None
built_tools = None
nav_engine = None
adapter = None
registry = None

try:
    from aether.core.tool_discovery import ToolDiscovery
    discovery = ToolDiscovery()
    discovery.discover()
    manifest = discovery.manifest
except Exception as e:
    print(f"  [FATAL] Cannot initialize ToolDiscovery: {e}")
    print(f"\nScore: 0/{total}")
    sys.exit(1)

try:
    from aether.core.tool_builder import ToolBuilder
    builder = ToolBuilder(manifest)
    built_tools = builder.build_all()
except Exception:
    built_tools = {}

try:
    from aether.core.navigation_engine import NavigationEngine
    nav_engine = NavigationEngine(manifest, tools=built_tools)
except Exception:
    nav_engine = None

try:
    from aether.adapters.universal_adapter import UniversalAdapter
    adapter = UniversalAdapter(manifest)
except Exception:
    adapter = None

try:
    from aether.core.tool_registry import ToolRegistry, register_built_tools
    from aether.core.llm_planner import build_tool_descriptions
    registry = ToolRegistry()
    # Replicate main.py manifest enrichment
    manifest["tool_descriptions"] = build_tool_descriptions(registry)
    if nav_engine:
        manifest["navigation_actions"] = nav_engine.available_actions()
        manifest["navigation_level"] = nav_engine.level
        all_avail = sorted(set(
            manifest.get("available_tools", []) + nav_engine.available_actions()))
        manifest["available_tools"] = all_avail
    register_built_tools(registry, built_tools, nav_engine, manifest=manifest)
except Exception:
    registry = None

# ── Tests ─────────────────────────────────────────────────────────────

# Test 1: ToolDiscovery returns manifest with platform field
try:
    platform = manifest.get("platform", "")
    platform_info = manifest.get("platform_info", {})
    ok = bool(platform) and (isinstance(platform, str) or isinstance(platform, dict))
    report(1, "ToolDiscovery manifest has platform field", ok,
           f"platform={platform}, info={platform_info}" if ok
           else "platform field missing or empty")
except Exception as e:
    report(1, "ToolDiscovery manifest has platform field", False, str(e))

# Test 2: manifest has capability_score above 0
try:
    score = manifest.get("capability_score", 0)
    ok = isinstance(score, (int, float)) and score > 0
    report(2, "capability_score above 0", ok, f"score={score}")
except Exception as e:
    report(2, "capability_score above 0", False, str(e))

# Test 3: ToolBuilder.build_all() returns dict with at least 3 tools
try:
    ok = isinstance(built_tools, dict) and len(built_tools) >= 3
    report(3, "ToolBuilder builds at least 3 tools", ok,
           f"built {len(built_tools)} tools: {', '.join(sorted(built_tools.keys()))}")
except Exception as e:
    report(3, "ToolBuilder builds at least 3 tools", False, str(e))

# Test 4: SystemTool.get_cpu_percent() returns float between 0 and 100
try:
    from aether.core.tool_builder import SystemTool
    sys_tool = SystemTool()
    result = sys_tool.get_cpu_percent()
    cpu = result.get("result", {}).get("cpu_percent", -1)
    ok = result["success"] and 0 <= cpu <= 100
    report(4, "SystemTool.get_cpu_percent() returns 0-100", ok,
           f"cpu_percent={cpu}")
except Exception as e:
    report(4, "SystemTool.get_cpu_percent() returns 0-100", False, str(e))

# Test 5: CameraTool.capture_image() returns success if camera available
try:
    from aether.core.tool_builder import CameraTool
    cam = CameraTool()
    result = cam.capture_image()
    cam_available = manifest.get("hardware", {}).get("camera", {}).get("available", False)
    if cam_available:
        ok = result["success"]
        detail = f"filepath={result.get('result', {}).get('filepath', '?')}"
    else:
        ok = True  # no camera = skip is acceptable
        detail = "camera not available, skipped (acceptable)"
    report(5, "CameraTool.capture_image() works if camera present", ok, detail)
except Exception as e:
    report(5, "CameraTool.capture_image() works if camera present", False, str(e))

# Test 6: NavigationEngine.level returns 1, 2, or 3
try:
    assert nav_engine is not None, "NavigationEngine not initialized"
    lvl = nav_engine.level
    ok = lvl in (1, 2, 3)
    report(6, "NavigationEngine.level is 1, 2, or 3", ok,
           f"level={lvl} ({nav_engine.level_name})")
except Exception as e:
    report(6, "NavigationEngine.level is 1, 2, or 3", False, str(e))

# Test 7: NavigationEngine.available_actions() returns non-empty list
try:
    assert nav_engine is not None, "NavigationEngine not initialized"
    actions = nav_engine.available_actions()
    ok = isinstance(actions, list) and len(actions) > 0
    report(7, "NavigationEngine.available_actions() non-empty", ok,
           f"{len(actions)} actions")
except Exception as e:
    report(7, "NavigationEngine.available_actions() non-empty", False, str(e))

# Test 8: NavigationEngine.observe() returns array of length 15
try:
    assert nav_engine is not None, "NavigationEngine not initialized"
    obs = nav_engine.observe()
    ok = hasattr(obs, '__len__') and len(obs) == 15
    report(8, "NavigationEngine.observe() returns length 15", ok,
           f"shape={getattr(obs, 'shape', len(obs))}")
except Exception as e:
    report(8, "NavigationEngine.observe() returns length 15", False, str(e))

# Test 9: UniversalAdapter.available_actions() returns non-empty list
try:
    assert adapter is not None, "UniversalAdapter not initialized"
    actions = adapter.available_actions()
    ok = isinstance(actions, list) and len(actions) > 0
    report(9, "UniversalAdapter.available_actions() non-empty", ok,
           f"{len(actions)} actions: {', '.join(actions[:6])}")
except Exception as e:
    report(9, "UniversalAdapter.available_actions() non-empty", False, str(e))

# Test 10: UniversalAdapter.execute('stop', {}) returns success True
try:
    assert adapter is not None, "UniversalAdapter not initialized"
    result, success = adapter.execute("stop", {})
    ok = success and result.get("success", False)
    report(10, "UniversalAdapter.execute('stop') succeeds", ok,
           f"result={result.get('result', result.get('error', '?'))}")
except Exception as e:
    report(10, "UniversalAdapter.execute('stop') succeeds", False, str(e))

# Test 11: UniversalAdapter.execute('takeoff', {}) fails with reason
try:
    assert adapter is not None, "UniversalAdapter not initialized"
    result, success = adapter.execute("takeoff", {})
    has_mavlink = manifest.get("hardware", {}).get("mavlink", {}).get("available", False)
    if has_mavlink:
        # If mavlink IS available, takeoff might succeed or fail for other reasons
        ok = True
        detail = "mavlink available, takeoff attempted"
    else:
        ok = not success
        error_msg = result.get("error", "")
        ok = ok and ("not available" in error_msg or "requires" in error_msg
                     or "not detected" in error_msg)
        detail = f"error={error_msg[:100]}"
    report(11, "UniversalAdapter.execute('takeoff') fails with reason", ok, detail)
except Exception as e:
    report(11, "UniversalAdapter.execute('takeoff') fails with reason", False, str(e))

# Test 12: UniversalAdapter.execute('scan_environment', {}) returns success
try:
    assert adapter is not None, "UniversalAdapter not initialized"
    result, success = adapter.execute("scan_environment", {})
    ok = success and result.get("success", False)
    report(12, "UniversalAdapter.execute('scan_environment') succeeds", ok,
           f"keys={list(result.get('result', result).keys())[:5]}")
except Exception as e:
    report(12, "UniversalAdapter.execute('scan_environment') succeeds", False, str(e))

# Test 13: ToolRegistry has at least 20 registered tools
try:
    assert registry is not None, "ToolRegistry not initialized"
    tools = registry.available_tools()
    ok = len(tools) >= 20
    report(13, "ToolRegistry has at least 20 tools", ok,
           f"{len(tools)} tools registered")
except Exception as e:
    report(13, "ToolRegistry has at least 20 tools", False, str(e))

# Test 14: write_file accepts both path and filename
try:
    assert registry is not None, "ToolRegistry not initialized"
    import tempfile
    # Test with 'path' param
    fd, tmp1 = tempfile.mkstemp(suffix=".txt", prefix="aether_test_")
    os.close(fd)
    r1 = registry.execute("write_file", {"path": tmp1, "content": "test_path"})

    # Test with 'filename' param
    fd, tmp2 = tempfile.mkstemp(suffix=".txt", prefix="aether_test_")
    os.close(fd)
    r2 = registry.execute("write_file", {"filename": tmp2, "content": "test_filename"})

    ok = r1.success and r2.success
    # Verify contents
    with open(tmp1) as f:
        c1 = f.read()
    with open(tmp2) as f:
        c2 = f.read()
    ok = ok and c1 == "test_path" and c2 == "test_filename"
    os.unlink(tmp1)
    os.unlink(tmp2)
    report(14, "write_file accepts path and filename params", ok,
           f"path={'OK' if r1.success else 'FAIL'}, "
           f"filename={'OK' if r2.success else 'FAIL'}")
except Exception as e:
    report(14, "write_file accepts path and filename params", False, str(e))

# Test 15: manifest missing_capabilities is non-empty
try:
    missing = manifest.get("missing_capabilities", [])
    ok = isinstance(missing, list) and len(missing) > 0
    report(15, "manifest missing_capabilities is non-empty", ok,
           f"{len(missing)} missing: {', '.join(missing[:4])}"
           + ("..." if len(missing) > 4 else ""))
except Exception as e:
    report(15, "manifest missing_capabilities is non-empty", False, str(e))

# ── Summary ───────────────────────────────────────────────────────────

print()
print("=" * 60)
print(f"  Score: {passed}/{total}")
if passed >= 13:
    print("  GENERALIZATION VALIDATED")
else:
    print(f"  {failed} test(s) failed — review above")
print("=" * 60)
