"""
LLMPlanner: decomposes objectives into tool steps via the Anthropic API.
Falls back to None when API is unavailable, letting the caller use keyword planning.
"""
import json
import os
import re
from typing import Any, Dict, List, Optional


_SYSTEM_PROMPT = (
    "You are a planning agent for AETHER, an autonomous spacecraft fault detection "
    "and recovery system. Given a capability manifest listing all available tools "
    "and a user objective, decompose the objective into ordered steps.\n\n"
    "Each step must have:\n"
    "  step_number: integer starting at 1\n"
    "  tool: name of the tool to use (MUST be in the available tools list)\n"
    "  inputs: dict of parameter key-value pairs for the tool\n"
    "  expected_output: brief description of what this step produces\n"
    "  pipe_from: step_number whose output feeds this step's primary input "
    "(null if no piping)\n"
    "  pipe_key: the input parameter name to pipe into (null if no piping)\n"
    "  fallback_tool: alternative tool if primary fails (null if none)\n"
    "  fallback_inputs: alternative inputs dict (null if none)\n\n"
    "Rules:\n"
    "- Only use tools from the available_tools list\n"
    "- For simulation objectives, use run_simulation with the correct scenario "
    "and fault_mode parameters\n"
    "- Valid scenarios: simple, obstacles, imu_fault, battery, compound, "
    "fault_heavy, multi_task\n"
    "- Valid fault modes: disabled, enabled, heavy\n"
    "- When the objective asks to analyze or summarize results, chain "
    "summarize_text after the data-producing step\n"
    "- When the objective asks to save or write, chain write_file at the end\n"
    "- When using write_file, ALWAYS use 'path' as the parameter name, NEVER "
    "'filename'. Example: write_file(path=\"output.txt\", content=\"text here\")\n"
    "- When the objective asks to verify or confirm a written file, chain "
    "execute_shell with a cat command\n"
    "- For repetition (e.g. 'run 10 simulations'), set the 'runs' parameter "
    "on run_simulation instead of creating 10 separate steps\n"
    "- SELF-INSPECTION RULE: If the objective asks about this system's own "
    "hardware, capabilities, tools, missing hardware, what it can do, or what "
    "would enable new features — NEVER use web_search. Instead use "
    "system_metrics to read the local capability manifest directly. The "
    "manifest already contains all hardware/software/network capabilities. "
    "Chain summarize_text after system_metrics to produce the answer. Only "
    "use web_search for external research topics unrelated to this system.\n"
    "- Navigation actions (visual_scan, track_object, navigate_to_color, "
    "avoid_obstacle, takeoff, land, etc.) are only available if listed in "
    "available_tools. Check navigation_level in the manifest: level 1 = camera "
    "only, level 2 = camera + motors, level 3 = flight controller. Never plan "
    "motor or flight actions at a lower level.\n"
    "- Navigation action inputs use: color, direction, speed, altitude, pin, "
    "duration, timeout, marker_color, left_pin, right_pin, lat, lon, alt\n"
    "- Return valid JSON only — an object with a single key 'steps' containing "
    "the array. No markdown, no explanation, no code fences."
)

# Token-efficient model for planning
_DEFAULT_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 2048


class LLMPlanner:
    """
    Calls the Anthropic API to decompose objectives into executable steps.
    Returns None if the API is unavailable, signaling the caller to use
    the keyword-based fallback.
    """

    def __init__(self, model: str = _DEFAULT_MODEL):
        self.model = model
        self._client = None
        self._available = self._init_client()

    def _init_client(self) -> bool:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return False
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def plan(self, objective: str, manifest: Dict,
             memory: Optional[List[Dict]] = None) -> Optional[List[Dict]]:
        """
        Decompose an objective into ordered tool steps via LLM.

        Returns a list of step dicts on success, or None if the API call
        fails (caller should fall back to keyword planning).
        """
        if not self._available:
            return None

        # Build the user message with manifest and memory context
        available_tools = manifest.get("available_tools", [])
        tool_descriptions = manifest.get("tool_descriptions", {})

        user_parts = []
        user_parts.append("## Available Tools")
        user_parts.append(json.dumps(available_tools, indent=2))

        if tool_descriptions:
            user_parts.append("\n## Tool Descriptions")
            for name, desc in tool_descriptions.items():
                user_parts.append(f"- {name}: {desc}")

        nav_level = manifest.get("navigation_level")
        nav_actions = manifest.get("navigation_actions", [])
        if nav_actions:
            level_names = {0: "system-only", 1: "camera",
                           2: "camera+motors", 3: "flight"}
            user_parts.append(f"\n## Navigation Engine")
            user_parts.append(f"Level: {nav_level} ({level_names.get(nav_level, '?')})")
            user_parts.append(f"Navigation actions: {', '.join(nav_actions)}")

        if memory:
            user_parts.append("\n## Recent Memory (previous successful objectives)")
            for mem in memory[-3:]:
                user_parts.append(
                    f"- Objective: {mem.get('objective', '?')}\n"
                    f"  Chain: {' -> '.join(mem.get('tool_chain', []))}\n"
                    f"  Outcome: {mem.get('outcome', '?')}"
                )

        user_parts.append(f"\n## Objective\n{objective}")

        user_message = "\n".join(user_parts)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text = response.content[0].text
            steps = self._parse_response(raw_text, available_tools)
            if steps:
                return steps
            return None
        except Exception as e:
            print(f"  [LLM Planner] API error: {e} — falling back to keyword planner")
            return None

    def _parse_response(self, raw: str,
                        available_tools: List[str]) -> Optional[List[Dict]]:
        """Parse LLM JSON response into validated step list."""
        # Strip markdown code fences if present
        text = raw.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            m = re.search(r'\{[\s\S]*\}', text)
            if not m:
                print("  [LLM Planner] Could not parse JSON from response")
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                print("  [LLM Planner] Could not parse JSON from response")
                return None

        steps_raw = data.get("steps", data if isinstance(data, list) else [])
        if not isinstance(steps_raw, list) or not steps_raw:
            return None

        # Validate and normalize each step
        validated = []
        for step in steps_raw:
            tool = step.get("tool", "")
            if tool not in available_tools:
                print(f"  [LLM Planner] Skipping unavailable tool: {tool}")
                continue

            normalized = {
                "step_number": step.get("step_number", len(validated) + 1),
                "tool": tool,
                "params": step.get("inputs", {}),
                "expected_output": step.get("expected_output", ""),
                "fallback_tool": step.get("fallback_tool"),
                "fallback_params": step.get("fallback_inputs"),
            }

            # Set up piping from previous step
            pipe_from = step.get("pipe_from")
            pipe_key = step.get("pipe_key")
            if pipe_from and pipe_key:
                normalized["pipe_input"] = pipe_key

            # Validate fallback tool
            fb = normalized.get("fallback_tool")
            if fb and fb not in available_tools:
                normalized["fallback_tool"] = None
                normalized["fallback_params"] = None

            validated.append(normalized)

        return validated if validated else None


def build_tool_descriptions(registry) -> Dict[str, str]:
    """Extract tool name→description mapping from ToolRegistry for the manifest."""
    descriptions = {}
    for info in registry.list_tools():
        if info.get("available"):
            descriptions[info["name"]] = info["description"]
    return descriptions
