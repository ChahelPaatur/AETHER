"""
CorrectionAgent: post-step verifier that uses the Anthropic API to evaluate
whether a step produced the expected result and suggests corrections.
Retries up to 3 times before reporting failure to the user.
"""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

_CORRECTIONS_LOG = os.path.join("logs", "corrections.json")

_SYSTEM_PROMPT = (
    "You are a correction agent for AETHER, an autonomous spacecraft system. "
    "You receive the planned step, expected output, and actual result. "
    "Evaluate whether the step succeeded. If it failed, diagnose the issue "
    "and suggest a corrective action using one of the available tools.\n\n"
    "CRITICAL RULES:\n"
    "- If write_file failed due to a missing directory, suggest correction_tool "
    "= 'write_file' with the SAME path and content. write_file creates "
    "directories automatically on retry.\n"
    "- NEVER suggest execute_shell with file content as the command. "
    "execute_shell must only receive short shell commands (ls, mkdir, cat, etc).\n"
    "- If you suggest execute_shell, the command must be a valid shell command "
    "under 200 characters, never raw file content.\n"
    "- If a tool fails due to a missing dependency (e.g. 'cv2 not installed'), "
    "do NOT suggest pip install. Instead, suggest an alternative tool that "
    "provides similar functionality (e.g. capture_image uses picamera2 as "
    "fallback, measure_brightness and detect_motion work without cv2).\n\n"
    "Return valid JSON only — no markdown, no explanation, no code fences:\n"
    "{\n"
    '  "success": true/false,\n'
    '  "issue": "description of what went wrong (empty string if success)",\n'
    '  "correction_tool": "tool name to try (null if success or no fix possible)",\n'
    '  "correction_inputs": {"param": "value"} (null if success)\n'
    "}"
)

# Fallback tool mappings: when a tool fails due to missing dependency,
# try these alternatives instead of attempting pip install
_TOOL_FALLBACKS = {
    "capture_image": ["visual_scan", "measure_brightness", "report_surroundings"],
    "visual_scan": ["capture_image", "measure_brightness", "report_surroundings"],
    "detect_motion": ["measure_brightness", "capture_image"],
    "detect_color": ["measure_brightness", "capture_image"],
    "detect_objects": ["visual_scan", "capture_image"],
}

# Error patterns that indicate a missing dependency (do NOT pip install)
_DEPENDENCY_ERRORS = [
    "cv2 not installed",
    "not installed",
    "no module named",
    "import error",
]

MAX_RETRIES = 3


class CorrectionAgent:
    """
    Runs after every execution step. Calls the Anthropic API to evaluate
    whether the step met expectations. If not, executes corrections up to
    MAX_RETRIES times.
    """

    def __init__(self, available_tools: List[str]):
        self._available_tools = available_tools
        self._client = None
        self._available = self._init_client()
        self._log: List[Dict] = []
        self._load_log()

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

    def evaluate_and_correct(
        self,
        step: Dict,
        result: Any,
        registry,
    ) -> Dict:
        """
        Evaluate a completed step and apply corrections if needed.

        Args:
            step: The planned step dict (tool, params, expected_output, etc.)
            result: The ToolResult from executing the step.
            registry: ToolRegistry for executing correction tools.

        Returns:
            Dict with keys:
                final_result: The final ToolResult (original or corrected)
                corrected: bool — whether a correction was applied
                attempts: int — number of correction attempts
                corrections: list of correction dicts applied
        """
        if not self._available:
            return self._passthrough(result)

        tool_name = step.get("tool", "")
        expected = step.get("expected_output", "")
        params = step.get("params", {})

        # If no expected_output defined, skip LLM evaluation —
        # just use the result's own success flag
        if not expected:
            return self._passthrough(result)

        actual = self._format_actual(result)
        corrections_applied = []
        current_result = result
        attempt = 0

        # Pre-check: if the error is a missing dependency, try fallback tools
        # instead of going through the LLM correction loop
        if not result.success:
            error_lower = (result.error or "").lower()
            is_dep_error = any(pat in error_lower for pat in _DEPENDENCY_ERRORS)
            if is_dep_error:
                fallbacks = _TOOL_FALLBACKS.get(tool_name, [])
                for fb in fallbacks:
                    if fb in self._available_tools:
                        print(f"    [CORRECT] Missing dependency for {tool_name} "
                              f"— switching to {fb}")
                        fb_result = registry.execute(fb, params)
                        if fb_result.success:
                            return {
                                "final_result": fb_result,
                                "corrected": True,
                                "attempts": 1,
                                "corrections": [{
                                    "attempt": 1,
                                    "issue": f"dependency missing: {result.error}",
                                    "correction_tool": fb,
                                    "correction_inputs": params,
                                    "result_success": True,
                                }],
                            }
                # No fallback worked — still try the LLM loop below

        while attempt < MAX_RETRIES:
            verdict = self._call_llm(tool_name, params, expected, actual)
            if verdict is None:
                # API failed — trust the result as-is
                break

            if verdict.get("success", True):
                # Step is fine
                if attempt > 0:
                    self._log_correction(step, corrections_applied, final_success=True)
                return {
                    "final_result": current_result,
                    "corrected": attempt > 0,
                    "attempts": attempt,
                    "corrections": corrections_applied,
                }

            # Step failed — try correction
            attempt += 1
            issue = verdict.get("issue", "Unknown issue")
            corr_tool = verdict.get("correction_tool")
            corr_inputs = verdict.get("correction_inputs") or {}

            correction_entry = {
                "attempt": attempt,
                "issue": issue,
                "correction_tool": corr_tool,
                "correction_inputs": corr_inputs,
            }

            print(f"    [CORRECT] Attempt {attempt}/{MAX_RETRIES}: {issue}")

            if not corr_tool or corr_tool not in self._available_tools:
                print(f"    [CORRECT] No valid correction tool suggested — stopping")
                correction_entry["result"] = "no_valid_tool"
                corrections_applied.append(correction_entry)
                break

            # Guard: block execute_shell when command looks like file content
            corr_inputs = self._sanitize_correction(
                corr_tool, corr_inputs, step)
            if corr_inputs is None:
                print(f"    [CORRECT] Blocked unsafe correction — stopping")
                correction_entry["result"] = "blocked_unsafe"
                corrections_applied.append(correction_entry)
                break

            print(f"    [CORRECT] Trying {corr_tool}({self._short(corr_inputs)})")
            current_result = registry.execute(corr_tool, corr_inputs)
            actual = self._format_actual(current_result)

            correction_entry["result_success"] = current_result.success
            corrections_applied.append(correction_entry)

            if current_result.success:
                # Re-evaluate with the new result in next loop iteration
                tool_name = corr_tool
                params = corr_inputs
                continue

        # Exhausted retries
        if corrections_applied:
            self._log_correction(step, corrections_applied,
                                 final_success=current_result.success)
            if not current_result.success:
                self._report_failure(step, corrections_applied)

        return {
            "final_result": current_result,
            "corrected": len(corrections_applied) > 0,
            "attempts": attempt,
            "corrections": corrections_applied,
        }

    def _call_llm(self, tool: str, params: Dict, expected: str,
                  actual: str) -> Optional[Dict]:
        """Call the Anthropic API to evaluate step success."""
        user_msg = (
            f"Tool: {tool}\n"
            f"Inputs: {json.dumps(params, default=str)[:500]}\n"
            f"Expected: {expected}\n"
            f"Got: {actual}\n\n"
            f"Available tools for corrections: {', '.join(self._available_tools)}\n\n"
            f"Did this succeed? If not what went wrong and what should be tried instead?"
        )

        try:
            response = self._client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            return self._parse_verdict(response.content[0].text)
        except Exception as e:
            print(f"    [CORRECT] API error: {e}")
            return None

    def _parse_verdict(self, raw: str) -> Optional[Dict]:
        """Parse the LLM JSON verdict."""
        text = raw.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r'\{[\s\S]*\}', text)
            if not m:
                return None
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                return None

        # Validate required fields
        if "success" not in data:
            return None

        return {
            "success": bool(data.get("success")),
            "issue": str(data.get("issue", "")),
            "correction_tool": data.get("correction_tool"),
            "correction_inputs": data.get("correction_inputs"),
        }

    def _report_failure(self, step: Dict, corrections: List[Dict]) -> None:
        """Print a clear failure report to the user."""
        tool = step.get("tool", "?")
        expected = step.get("expected_output", "?")
        print(f"\n    {'='*50}")
        print(f"    CORRECTION FAILED after {len(corrections)} attempt(s)")
        print(f"    Original tool: {tool}")
        print(f"    Expected: {expected}")
        for c in corrections:
            print(f"    Attempt {c['attempt']}: {c['issue']}")
            if c.get("correction_tool"):
                print(f"      Tried: {c['correction_tool']} → "
                      f"{'OK' if c.get('result_success') else 'FAILED'}")
        print(f"    {'='*50}\n")

    def _log_correction(self, step: Dict, corrections: List[Dict],
                        final_success: bool) -> None:
        """Append correction record to logs/corrections.json."""
        entry = {
            "timestamp": time.time(),
            "step_tool": step.get("tool", ""),
            "step_params": {k: str(v)[:100] for k, v in step.get("params", {}).items()},
            "expected": step.get("expected_output", ""),
            "attempts": len(corrections),
            "final_success": final_success,
            "corrections": corrections,
        }
        self._log.append(entry)
        self._save_log()

    def _load_log(self) -> None:
        if os.path.exists(_CORRECTIONS_LOG):
            try:
                with open(_CORRECTIONS_LOG) as f:
                    self._log = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._log = []

    def _save_log(self) -> None:
        os.makedirs(os.path.dirname(_CORRECTIONS_LOG) or ".", exist_ok=True)
        try:
            with open(_CORRECTIONS_LOG, "w") as f:
                json.dump(self._log, f, indent=2)
        except OSError:
            pass

    @staticmethod
    def _format_actual(result) -> str:
        """Format a ToolResult into a concise string for the LLM."""
        if not result.success:
            return f"FAILED: {result.error} (took {result.duration_ms:.0f}ms)"
        output = str(result.output) if result.output is not None else "(no output)"
        if len(output) > 800:
            output = output[:797] + "..."
        return f"SUCCESS: {output} (took {result.duration_ms:.0f}ms)"

    @staticmethod
    def _short(d: Dict) -> str:
        parts = []
        for k, v in d.items():
            val = str(v)
            if len(val) > 30:
                val = val[:27] + "..."
            parts.append(f"{k}={val}")
        return ", ".join(parts)

    @staticmethod
    def _sanitize_correction(corr_tool: str, corr_inputs: Dict,
                             original_step: Dict) -> Optional[Dict]:
        """Validate correction inputs. Returns sanitized inputs or None to block.

        Key guard: when write_file fails, the LLM sometimes suggests
        execute_shell with the file content as the command. This detects
        that case and instead returns a mkdir + write_file retry.
        """
        if corr_tool == "execute_shell":
            cmd = corr_inputs.get("command", "")
            # Block if "command" is suspiciously long (file content, not a shell cmd)
            if len(cmd) > 200:
                return None
            # Block if command contains newlines (likely file content pasted in)
            if "\n" in cmd and not cmd.strip().startswith("#"):
                return None

        # If the original step was write_file and the correction is execute_shell,
        # rewrite to mkdir -p on the directory, then let the retry loop re-attempt
        if (original_step.get("tool") == "write_file"
                and corr_tool == "execute_shell"):
            original_path = original_step.get("params", {}).get("path", "")
            if original_path:
                import os
                dirpath = os.path.dirname(original_path)
                if dirpath:
                    return {"command": f"mkdir -p {dirpath}"}
            # No valid path to mkdir — block
            return None

        return corr_inputs

    @staticmethod
    def _passthrough(result) -> Dict:
        return {
            "final_result": result,
            "corrected": False,
            "attempts": 0,
            "corrections": [],
        }
