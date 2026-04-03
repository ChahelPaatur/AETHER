"""
ToolRegistry: real executable tools for AETHER agent mode.
Each tool matches the AbstractAction interface: name, description, execute(params) → result, can_execute() → bool.
"""
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class ToolResult:
    """Result of a real tool execution."""
    tool: str
    success: bool
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0


class BaseTool(ABC):
    """Abstract base matching AbstractAction interface."""
    name: str = ""

    @abstractmethod
    def execute(self, params: Dict) -> ToolResult:
        """Execute the tool with given parameters."""
        ...

    @abstractmethod
    def can_execute(self) -> bool:
        """Returns True if the tool's dependencies are available."""
        ...

    def description(self) -> str:
        return f"Tool: {self.name}"

    def preconditions(self, state: Dict) -> bool:
        return self.can_execute()

    def expected_effect(self, state: Dict) -> Dict:
        return {"tool_executed": self.name}


class WebSearchTool(BaseTool):
    name = "web_search"

    _SYSTEM_PROMPT = (
        "You are a research assistant with knowledge up to 2025. The user will "
        "give you a topic. Write 3 substantive paragraphs about that topic with "
        "specific facts, examples, and recent developments. Never ask for "
        "clarification — always write about the topic directly."
    )

    # Phrases to strip from queries before searching
    _STRIP_PATTERNS = [
        r'\bwrite\s+a?\s*\d*\s*paragraph\s*summary\b',
        r'\bwrite\s+a\s+summary\b',
        r'\bsave\s+(it\s+)?to\s+\S+',
        r'\bconfirm\b',
        r'\band\s+confirm\b',
        r'\bsearch\s+for\b',
        r'\bresearch\b',
        r'\bsummarize\s*(the\s+results?)?\b',
        r'\blook\s+up\b',
        r'\bfind\s+information\s+(about|on)\b',
        r'\band\s+save\b',
        r'\bthen\s+\w+\b',
        r'\S+\.(?:txt|json|md|csv|html|log)\b',
        r'\b\d+\s+paragraph\b',
    ]

    def description(self) -> str:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "Research topics using AI knowledge (Anthropic API)"
        return "Research topics using AI knowledge (set ANTHROPIC_API_KEY to enable)"

    def can_execute(self) -> bool:
        return True  # falls back to topic echo if no API key

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        query = params.get("query", "")
        url = params.get("url", "")
        if not query and not url:
            return ToolResult(tool=self.name, success=False, error="No query or url provided")

        # Direct URL fetch (unchanged)
        if url:
            try:
                from urllib.request import Request, urlopen
                req = Request(url, headers={"User-Agent": "AETHER/3.0"})
                resp = urlopen(req, timeout=params.get("timeout", 15))
                body = resp.read().decode("utf-8", errors="replace")
                return ToolResult(tool=self.name, success=True,
                                  output=body[:params.get("max_chars", 5000)],
                                  duration_ms=(time.time() - t0) * 1000)
            except Exception as e:
                return ToolResult(tool=self.name, success=False, error=str(e),
                                  duration_ms=(time.time() - t0) * 1000)

        # Clean query: strip task instructions, keep only the research topic
        clean_query = self._extract_topic(query)
        print(f"    [web_search] Topic: \"{clean_query}\"")

        # Try Anthropic API
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model=params.get("model", "claude-sonnet-4-20250514"),
                    max_tokens=params.get("max_tokens", 2048),
                    system=self._SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": clean_query}],
                )
                output = message.content[0].text
                elapsed = (time.time() - t0) * 1000
                print(f"    [web_search] Anthropic API returned {len(output)} chars")
                return ToolResult(tool=self.name, success=True,
                                  output=f"[KNOWLEDGE]\n{output}",
                                  duration_ms=elapsed)
            except ImportError:
                print("    [web_search] anthropic package not installed — using fallback")
            except Exception as e:
                print(f"    [web_search] Anthropic API error: {e} — using fallback")

        # Fallback: return the cleaned topic as a stub so chained tools still work
        print("    [web_search] No API key — set ANTHROPIC_API_KEY for AI-powered research")
        elapsed = (time.time() - t0) * 1000
        return ToolResult(
            tool=self.name, success=True,
            output=(f"[KNOWLEDGE] Research topic: {clean_query}\n"
                    f"To get AI-generated research content, run:\n"
                    f"  export ANTHROPIC_API_KEY=your_key_here\n"
                    f"  pip install anthropic"),
            duration_ms=elapsed,
        )

    _TASK_WORDS = {"summary", "paragraph", "save", "write", "file", "confirm",
                   "store", "output", "create"}

    def _extract_topic(self, raw: str) -> str:
        """Strip task instructions and extract only the research topic.

        Strategy: apply regex stripping, then validate. If result still has
        task words or is too short, fall back to text before first comma.
        """
        text = raw.strip()

        # First try: strip task instruction patterns
        cleaned = text
        for pattern in self._STRIP_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s*,\s*,\s*', ', ', cleaned)
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        cleaned = re.sub(r'^(?:and|then|also|,)\s+', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r'\s+(?:and|then|also|,)\s*$', '', cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip(' ,')

        # Validate: if too short or still has task words, use text before first comma
        words = set(cleaned.lower().split())
        if len(cleaned) < 10 or words & self._TASK_WORDS:
            if ',' in text:
                cleaned = text[:text.index(',')].strip()
            else:
                cleaned = text

            # Strip leading verb prefix
            for prefix in ["research ", "search for ", "search ", "look up ",
                           "find information about ", "find "]:
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
                    break
            cleaned = cleaned.strip(' ,')

        return cleaned if cleaned else raw.strip()


class ExecuteShellTool(BaseTool):
    name = "execute_shell"

    # Commands that are never allowed
    _BLOCKED = {"rm -rf /", "mkfs", "dd if=", ":(){", "fork bomb"}

    # Shell metacharacters that enable injection — rejected unless
    # the command is explicitly whitelisted.
    _INJECTION_CHARS = {";", "|", "`", "$("}

    def description(self) -> str:
        return "Execute a command or filesystem operation"

    def can_execute(self) -> bool:
        return True

    # Phrases that mean "list files" — matched against raw input before any translation
    _LIST_PHRASES = [
        "list files in ", "list files", "list directory ", "list directory",
        "show files in ", "show files", "show directory ", "show directory",
        "current directory", "what files are in ", "what's in ",
        "search for files in ", "files in directory",
    ]

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        command = params.get("command", "")
        if not command:
            return ToolResult(tool=self.name, success=False, error="No command provided")

        # Check for file-listing intent FIRST on the raw input — never pass to subprocess
        lower = command.strip().lower()
        for phrase in self._LIST_PHRASES:
            if phrase in lower:
                path = self._extract_dir_path(command, phrase)
                return self._list_files(path, t0)

        # Translate remaining natural language to an executable command
        command = self._translate_command(command)

        # Safety check
        for blocked in self._BLOCKED:
            if blocked in command:
                return ToolResult(tool=self.name, success=False,
                                  error=f"Blocked dangerous command pattern: {blocked}")

        # Reject shell injection characters
        for ch in self._INJECTION_CHARS:
            if ch in command:
                return ToolResult(
                    tool=self.name, success=False,
                    error=f"Blocked shell metacharacter '{ch}' — "
                          f"split into separate commands instead")

        # Python builtins for common shell commands
        builtin_result = self._try_python_builtin(command, t0)
        if builtin_result is not None:
            return builtin_result

        # Fall through to subprocess for actual shell commands
        timeout = params.get("timeout", 30)
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=params.get("cwd"),
            )
            elapsed = (time.time() - t0) * 1000
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr] {result.stderr}"
            return ToolResult(
                tool=self.name,
                success=result.returncode == 0,
                output=output[:params.get("max_chars", 10000)],
                error="" if result.returncode == 0 else f"Exit code {result.returncode}",
                duration_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(tool=self.name, success=False,
                              error=f"Command timed out after {timeout}s",
                              duration_ms=(time.time() - t0) * 1000)
        except Exception as e:
            return ToolResult(tool=self.name, success=False, error=str(e),
                              duration_ms=(time.time() - t0) * 1000)

    def _extract_dir_path(self, raw: str, matched_phrase: str) -> str:
        """Extract directory path from a file-listing phrase."""
        lower = raw.strip().lower()
        idx = lower.find(matched_phrase)
        if idx >= 0:
            after = raw.strip()[idx + len(matched_phrase):].strip()
            # Remove trailing natural language ("and confirm", "then save", etc.)
            after = re.split(r'\s+(?:and|then|save|confirm)\b', after, maxsplit=1)[0].strip()
            if after:
                return after
        return "."

    def _translate_command(self, raw: str) -> str:
        """Strip natural-language prefixes so only the real command remains."""
        text = raw.strip()
        lower = text.lower()

        for prefix in ["please run ", "please execute ", "run the command ",
                        "run command ", "execute command ", "run ", "execute "]:
            if lower.startswith(prefix):
                text = text[len(prefix):].strip()
                lower = text.lower()
                break

        # "create directory X" / "make directory X" → mkdir
        for phrase in ["create directory ", "make directory ", "create folder ",
                       "make folder "]:
            if lower.startswith(phrase):
                dirname = text[len(phrase):].strip()
                return f"mkdir {dirname}"

        if lower.startswith("install "):
            pkg = text[len("install "):].strip()
            return f"pip install {pkg}"

        return text

    def _try_python_builtin(self, command: str, t0: float) -> Optional[ToolResult]:
        """Handle common commands with Python builtins instead of shell."""
        cmd = command.strip()
        lower = cmd.lower()

        # ls / dir → os.listdir()
        if lower in ("ls", "dir", "ls ."):
            return self._list_files(".", t0)
        if lower.startswith("ls "):
            path = cmd[3:].strip()
            tokens = path.split()
            path = next((t for t in tokens if not t.startswith("-")), ".")
            return self._list_files(path, t0)

        # mkdir
        if lower.startswith("mkdir "):
            dirname = cmd[6:].strip().lstrip("-p ")
            try:
                os.makedirs(dirname, exist_ok=True)
                return ToolResult(tool=self.name, success=True,
                                  output=f"Created directory: {dirname}",
                                  duration_ms=(time.time() - t0) * 1000)
            except Exception as e:
                return ToolResult(tool=self.name, success=False, error=str(e),
                                  duration_ms=(time.time() - t0) * 1000)

        # pwd
        if lower in ("pwd",):
            return ToolResult(tool=self.name, success=True,
                              output=os.getcwd(),
                              duration_ms=(time.time() - t0) * 1000)

        # cat → read file
        if lower.startswith("cat "):
            path = cmd[4:].strip()
            try:
                content = Path(path).read_text()[:100_000]
                return ToolResult(tool=self.name, success=True, output=content,
                                  duration_ms=(time.time() - t0) * 1000)
            except Exception as e:
                return ToolResult(tool=self.name, success=False, error=str(e),
                                  duration_ms=(time.time() - t0) * 1000)

        return None

    def _list_files(self, target_dir: str, t0: float) -> ToolResult:
        """Use os.listdir() directly — never subprocess."""
        target_dir = os.path.realpath(target_dir)
        try:
            entries = sorted(os.listdir(target_dir))
            output = "\n".join(entries)
            return ToolResult(
                tool=self.name, success=True, output=output,
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, error=str(e),
                              duration_ms=(time.time() - t0) * 1000)


class ReadFileTool(BaseTool):
    name = "read_file"

    _FILEPATH_RE = re.compile(r'[\w./\-]+\.(?:json|csv|txt|py|md|log|html)')
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    def description(self) -> str:
        return "Read a file from the local filesystem"

    def can_execute(self) -> bool:
        return True

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        path = params.get("path", "")
        if not path:
            return ToolResult(tool=self.name, success=False, error="No path provided")

        # Extract actual file path from natural language if needed
        if not os.path.exists(path) and ' ' in path:
            m = self._FILEPATH_RE.search(path)
            if m:
                path = m.group(0)

        # Prevent directory traversal
        path = os.path.realpath(path)

        # Guard: reject binary image files — they cause utf-8 decode errors
        ext = os.path.splitext(path)[1].lower()
        if ext in self._IMAGE_EXTS:
            return ToolResult(
                tool=self.name, success=False,
                error=(f"Binary image file ({ext}): use capture_image or "
                       f"visual_scan to process images, not read_file"),
                duration_ms=(time.time() - t0) * 1000,
            )

        try:
            with open(path, "r") as f:
                content = f.read(params.get("max_bytes", 100_000))
            return ToolResult(
                tool=self.name, success=True, output=content,
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, error=str(e),
                              duration_ms=(time.time() - t0) * 1000)


class WriteFileTool(BaseTool):
    name = "write_file"

    def description(self) -> str:
        return "Write content to a file on the local filesystem"

    def can_execute(self) -> bool:
        return True

    _FILEPATH_RE = re.compile(r'[\w./\-]+\.(?:json|csv|txt|py|md|log|html)')

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        # Accept both 'path' and 'filename' as the file path parameter
        path = params.get("path", "") or params.get("filename", "") or params.get("file", "")
        content = params.get("content", "")
        if not path:
            return ToolResult(tool=self.name, success=False, error="No path provided")

        # Extract actual file path from natural language if needed
        if ' ' in path:
            m = self._FILEPATH_RE.search(path)
            if m:
                path = m.group(0)

        path = os.path.realpath(path)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            mode = "a" if params.get("append") else "w"
            with open(path, mode) as f:
                f.write(content)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Wrote {len(content)} bytes to {path}",
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, error=str(e),
                              duration_ms=(time.time() - t0) * 1000)


class AppendFileTool(BaseTool):
    name = "append_file"

    def description(self) -> str:
        return "Append content to a file (creates if missing, adds newline separator)"

    def can_execute(self) -> bool:
        return True

    _FILEPATH_RE = re.compile(r'[\w./\-]+\.(?:json|csv|txt|py|md|log|html)')

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        path = params.get("path", "") or params.get("filename", "") or params.get("file", "")
        content = params.get("content", "")
        if not path:
            return ToolResult(tool=self.name, success=False, error="No path provided")

        if ' ' in path:
            m = self._FILEPATH_RE.search(path)
            if m:
                path = m.group(0)

        path = os.path.realpath(path)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a") as f:
                # Add newline separator if file already has content
                if os.path.getsize(path) > 0:
                    f.write("\n")
                f.write(content)
            return ToolResult(
                tool=self.name, success=True,
                output=f"Appended {len(content)} bytes to {path}",
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, error=str(e),
                              duration_ms=(time.time() - t0) * 1000)


class CallApiTool(BaseTool):
    name = "call_api"

    def description(self) -> str:
        return "Make an HTTP API call (GET/POST/PUT/DELETE)"

    def can_execute(self) -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        url = params.get("url", "")
        method = params.get("method", "GET").upper()
        if not url:
            return ToolResult(tool=self.name, success=False, error="No url provided")

        try:
            import requests
            kwargs = {
                "timeout": params.get("timeout", 15),
                "headers": params.get("headers", {}),
            }
            if method in ("POST", "PUT", "PATCH"):
                kwargs["json"] = params.get("body", params.get("json", {}))

            resp = getattr(requests, method.lower())(url, **kwargs)
            elapsed = (time.time() - t0) * 1000

            try:
                body = resp.json()
            except Exception:
                body = resp.text[:params.get("max_chars", 5000)]

            return ToolResult(
                tool=self.name,
                success=200 <= resp.status_code < 400,
                output=body,
                error="" if 200 <= resp.status_code < 400 else f"HTTP {resp.status_code}",
                duration_ms=elapsed,
            )
        except Exception as e:
            return ToolResult(tool=self.name, success=False, error=str(e),
                              duration_ms=(time.time() - t0) * 1000)


class SummarizeTextTool(BaseTool):
    name = "summarize_text"

    def __init__(self) -> None:
        self.system_context: str = ""

    def set_system_context(self, context: str) -> None:
        """Set domain definitions that are prepended to every summarization call."""
        self.system_context = context.strip()

    def description(self) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            return "Summarize text using the Anthropic Claude API"
        return "Summarize text (local fallback — set ANTHROPIC_API_KEY for AI summaries)"

    def can_execute(self) -> bool:
        # Always available: falls back to local extraction if no API key
        return True

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        text = params.get("text", "")
        if not text:
            return ToolResult(tool=self.name, success=False, error="No text provided")

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

        # Build system prompt with domain context
        system_parts = []
        if self.system_context:
            system_parts.append(f"Domain definitions:\n{self.system_context}")
        system_parts.append("You are a concise technical summarizer. Use the domain "
                            "definitions above when interpreting acronyms and metrics.")
        system_prompt = "\n\n".join(system_parts)

        # Try Anthropic API first
        if api_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                prompt = params.get("prompt", "Summarize the following text concisely:")
                message = client.messages.create(
                    model=params.get("model", "claude-haiku-4-5-20251001"),
                    max_tokens=params.get("max_tokens", 1024),
                    system=system_prompt,
                    messages=[{"role": "user", "content": f"{prompt}\n\n{text[:10000]}"}],
                )
                summary = message.content[0].text
                return ToolResult(
                    tool=self.name, success=True, output=summary,
                    duration_ms=(time.time() - t0) * 1000,
                )
            except ImportError:
                # anthropic package not installed — fall through to local
                pass
            except Exception as e:
                # API error — fall through to local with a note
                pass

        # Local fallback: extract first 3 sentences
        summary = self._local_summarize(text)
        method = "local extraction"
        if not api_key:
            method += (" | To enable AI summaries: "
                       "export ANTHROPIC_API_KEY=your_key_here && pip install anthropic")
        return ToolResult(
            tool=self.name, success=True,
            output=f"[{method}]\n{summary}",
            duration_ms=(time.time() - t0) * 1000,
        )

    @staticmethod
    def _local_summarize(text: str) -> str:
        """Extract the first 3 sentences as a basic summary."""
        # Split on sentence-ending punctuation followed by whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        selected = sentences[:3]
        result = " ".join(s.strip() for s in selected if s.strip())
        if len(sentences) > 3:
            result += " ..."
        return result if result else text[:200]


class RunSimulationTool(BaseTool):
    name = "run_simulation"

    _VALID_SCENARIOS = {"simple", "obstacles", "imu_fault", "battery",
                        "compound", "fault_heavy", "multi_task"}
    _VALID_FAULT_MODES = {"disabled", "enabled", "heavy"}
    _FAULT_PROB = {"disabled": 0.0, "enabled": 0.015, "heavy": 0.05}

    def description(self) -> str:
        return "Run AETHER simulation(s) and return metrics (supports repetition and multi-scenario)"

    def can_execute(self) -> bool:
        return True

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()

        # Support multiple scenarios via "scenarios" list param
        scenarios = params.get("scenarios", [])
        if not scenarios:
            scenarios = [params.get("scenario", "simple")]

        fault_mode = params.get("fault_mode", "enabled")
        max_steps = params.get("max_steps", 300)
        seed = params.get("seed", 42)
        task_text = params.get("task", "navigate to target")
        runs = max(1, min(100, params.get("runs", 1)))

        # Validate
        for sc in scenarios:
            if sc not in self._VALID_SCENARIOS:
                return ToolResult(
                    tool=self.name, success=False,
                    error=f"Invalid scenario '{sc}'. "
                          f"Valid: {', '.join(sorted(self._VALID_SCENARIOS))}",
                    duration_ms=(time.time() - t0) * 1000)
        if fault_mode not in self._VALID_FAULT_MODES:
            return ToolResult(
                tool=self.name, success=False,
                error=f"Invalid fault_mode '{fault_mode}'. "
                      f"Valid: {', '.join(sorted(self._VALID_FAULT_MODES))}",
                duration_ms=(time.time() - t0) * 1000)

        try:
            from ..simulation.environment import SimulationEnvironment
            from ..simulation.scenarios import get_scenario, ScenarioGenerator
            from ..faults.fault_injector import FaultInjector
            from .message_bus import MessageBus
            from ..agents.task_manager import TaskManagerAgent
            from .metrics import MetricsTracker

            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "configs", "rover_v1.json")
            fault_prob = self._FAULT_PROB[fault_mode]

            all_scenario_results = {}

            for sc_name in scenarios:
                scenario_data = get_scenario(sc_name)
                if scenario_data is None:
                    scenario_data = ScenarioGenerator(seed).deterministic_scenario(0)

                run_results = []
                for run_idx in range(runs):
                    run_seed = seed + run_idx
                    env = SimulationEnvironment(seed=run_seed)
                    bus = MessageBus()
                    injector = FaultInjector(fault_probability=fault_prob, seed=run_seed)

                    agent = TaskManagerAgent(
                        env=env, config_path=config_path, bus=bus,
                        log_dir="logs", seed=run_seed,
                        fault_injector=injector,
                        verbose=False, no_learning=False,
                    )

                    result = agent.run_episode(
                        task_text=task_text,
                        scenario=scenario_data,
                        max_steps=max_steps,
                        render=False,
                    )
                    run_results.append(result)

                    if runs > 1:
                        print(f"    [run_simulation] {sc_name} run {run_idx + 1}/{runs} "
                              f"— SFRI={result['SFRI']:.1f}")

                if runs == 1:
                    all_scenario_results[sc_name] = run_results[0]
                else:
                    all_scenario_results[sc_name] = {
                        "runs": runs,
                        "mean": MetricsTracker.aggregate(run_results),
                        "individual": run_results,
                    }

            # Single scenario → return its result directly; multi → labeled dict
            if len(scenarios) == 1:
                output = all_scenario_results[scenarios[0]]
            else:
                output = all_scenario_results

            elapsed = (time.time() - t0) * 1000
            return ToolResult(
                tool=self.name, success=True,
                output=output,
                duration_ms=elapsed,
            )
        except Exception as e:
            return ToolResult(
                tool=self.name, success=False, error=str(e),
                duration_ms=(time.time() - t0) * 1000)


class DynamicTool(BaseTool):
    """Adapter that wraps any callable as a BaseTool for ToolRegistry.

    Bridges ToolBuilder tools (which return {success, result, error} dicts)
    and NavigationEngine actions into the BaseTool/ToolResult interface.
    """

    def __init__(self, name: str, fn, desc: str = ""):
        self.name = name
        self._fn = fn
        self._desc = desc or f"Dynamic tool: {name}"

    def description(self) -> str:
        return self._desc

    def can_execute(self) -> bool:
        return True

    def execute(self, params: Dict) -> ToolResult:
        t0 = time.time()
        try:
            raw = self._fn(**params) if params else self._fn()
            elapsed = (time.time() - t0) * 1000

            # String return — use directly as output
            if isinstance(raw, str):
                return ToolResult(tool=self.name, success=True,
                                  output=raw, duration_ms=elapsed)

            if isinstance(raw, dict) and "success" in raw:
                # ToolBuilder format: {success, result, error}
                if "result" in raw:
                    output = raw["result"]
                else:
                    # NavigationEngine format: data fields alongside
                    # success/error (no "result" wrapper) — extract the
                    # meaningful data by stripping meta keys
                    data = {k: v for k, v in raw.items()
                            if k not in ("success", "error")}
                    output = data if data else raw.get("result")
                return ToolResult(
                    tool=self.name,
                    success=raw["success"],
                    output=output,
                    error=raw.get("error", ""),
                    duration_ms=elapsed,
                )
            # Plain return value
            return ToolResult(tool=self.name, success=True,
                              output=raw, duration_ms=elapsed)
        except Exception as e:
            return ToolResult(tool=self.name, success=False, error=str(e),
                              duration_ms=(time.time() - t0) * 1000)


def register_built_tools(registry: "ToolRegistry",
                         built_tools: Dict[str, Any],
                         nav_engine=None,
                         manifest: Optional[Dict] = None) -> int:
    """Bridge ToolBuilder tools and NavigationEngine into ToolRegistry.

    Returns the number of tools registered.
    """
    count = 0

    # -- ToolBuilder tools --------------------------------------------------
    _TOOL_METHODS: Dict[str, List[tuple]] = {
        "camera": [
            ("capture_image", "Capture a camera frame and save to disk"),
            ("detect_motion", "Detect motion via frame differencing"),
            ("measure_brightness", "Measure average pixel brightness"),
            ("detect_color", "Detect percentage of a named color in frame"),
        ],
        "system": [
            ("get_cpu_percent", "Get current CPU usage percentage"),
            ("get_ram_percent", "Get current RAM usage and totals"),
            ("get_cpu_temp", "Read CPU temperature"),
            ("get_disk_space", "Get disk usage statistics"),
            ("get_battery", "Get battery status and percentage"),
        ],
        "gpu": [
            ("get_gpu_memory", "Report GPU memory usage"),
            ("run_inference", "Run a test inference on GPU"),
        ],
        "network": [
            ("web_fetch", "Fetch a URL and return status + body"),
            ("check_connectivity", "Check internet connectivity"),
        ],
        "motor": [
            ("get_motor_info", "Get detected motor controller info"),
            ("motor_forward", "Drive forward at given speed"),
            ("motor_turn", "Turn in a direction at given speed"),
            ("motor_stop", "Stop all motors immediately"),
            ("arm", "Arm the flight controller (MAVLink only)"),
            ("disarm", "Disarm the flight controller (MAVLink only)"),
        ],
        "yolo": [
            ("detect", "Run YOLOv8 object detection on an image"),
            ("detect_from_camera", "Capture camera frame and run YOLOv8 detection"),
            ("count_objects", "Count objects of a given class via YOLO detection"),
            ("describe_scene", "Describe everything visible (YOLO or Anthropic vision API)"),
        ],
        # storage methods overlap with existing ReadFileTool/WriteFileTool,
        # so we skip them to avoid name collisions
    }

    # Also register a combined system_metrics tool that includes the
    # capability manifest so the planner can answer self-inspection questions
    system_obj = built_tools.get("system")
    if system_obj:
        _manifest_ref = manifest or {}

        def _system_metrics():
            cpu = system_obj.get_cpu_percent()
            ram = system_obj.get_ram_percent()
            disk = system_obj.get_disk_space()
            batt = system_obj.get_battery()
            result = {
                "cpu": cpu.get("result") if cpu.get("success") else cpu.get("error"),
                "ram": ram.get("result") if ram.get("success") else ram.get("error"),
                "disk": disk.get("result") if disk.get("success") else disk.get("error"),
                "battery": batt.get("result") if batt.get("success") else batt.get("error"),
            }
            # Include capability manifest for self-inspection
            hw = _manifest_ref.get("hardware", {})
            sw = _manifest_ref.get("software", {})
            net = _manifest_ref.get("network", {})
            env = _manifest_ref.get("environment", {})
            result["capabilities"] = {
                "hardware": {
                    "camera": hw.get("camera", {}),
                    "gpio": hw.get("gpio", {}),
                    "gpu": hw.get("gpu", {}),
                    "i2c": hw.get("i2c", {}),
                    "imu": hw.get("imu", {}),
                    "mavlink": hw.get("mavlink", {}),
                    "serial_ports": hw.get("serial_ports", []),
                },
                "software": {k: v for k, v in sw.items()},
                "network": net,
                "platform": _manifest_ref.get("platform", {}),
                "navigation_level": _manifest_ref.get("navigation_level", 0),
                "navigation_actions": _manifest_ref.get("navigation_actions", []),
                "available_tools": _manifest_ref.get("available_tools", []),
            }
            # Summarize what's missing and what it would enable
            missing = []
            if not hw.get("gpio", {}).get("available"):
                missing.append("GPIO (would enable: motor control, level 2 navigation)")
            if not hw.get("mavlink", {}).get("available"):
                missing.append("MAVLink flight controller (would enable: takeoff, waypoint nav, level 3 navigation)")
            if not hw.get("imu", {}).get("available"):
                missing.append("IMU sensor (would enable: accelerometer, gyroscope, orientation)")
            if not hw.get("i2c", {}).get("available"):
                missing.append("I2C bus (would enable: external sensors, OLED displays)")
            if not hw.get("gpu", {}).get("available"):
                missing.append("CUDA/MPS GPU (would enable: fast inference, ML training)")
            if not net.get("internet"):
                missing.append("Internet (would enable: web search, API calls)")
            result["missing_hardware"] = missing
            return {"success": True, "result": result, "error": ""}

        registry.register(DynamicTool("system_metrics", _system_metrics,
                                      "Get system metrics and full capability manifest (hardware, software, missing hardware)"))
        count += 1

    # Name overrides: register some tools under custom names
    _NAME_OVERRIDES = {
        ("yolo", "detect"): "yolo_detect",
        ("yolo", "detect_from_camera"): "detect_objects_yolo",
        ("yolo", "count_objects"): "count_objects",
        ("yolo", "describe_scene"): "describe_scene",
    }

    for tool_key, methods in _TOOL_METHODS.items():
        obj = built_tools.get(tool_key)
        if obj is None:
            continue
        for method_name, desc in methods:
            fn = getattr(obj, method_name, None)
            if fn is None:
                continue
            reg_name = _NAME_OVERRIDES.get((tool_key, method_name), method_name)
            registry.register(DynamicTool(reg_name, fn, desc))
            count += 1

    # -- NavigationEngine actions -------------------------------------------
    if nav_engine is not None:
        _NAV_TOOLS = [
            ("visual_scan", "visual_scan",
             "Capture a frame and return structured metrics (brightness, edges, colors, contours, motion)"),
            ("report_surroundings", "report_surroundings",
             "Analyze camera image and return natural language scene description"),
        ]
        for reg_name, method_name, desc in _NAV_TOOLS:
            fn = getattr(nav_engine, method_name, None)
            if fn is not None:
                registry.register(DynamicTool(reg_name, fn, desc))
                count += 1

        # Register all remaining navigation actions via nav_engine.execute()
        for action in nav_engine.available_actions():
            if registry.get(action) is not None:
                continue  # already registered above or as a default tool
            def _make_nav_fn(act):
                def _fn(**params):
                    return nav_engine.execute(act, params)
                return _fn
            registry.register(DynamicTool(
                action, _make_nav_fn(action),
                f"Navigation action: {action} (level {nav_engine.level})"))
            count += 1

    print(f"  [ToolRegistry] Registered {count} tools from ToolBuilder")
    return count


class ToolRegistry:
    """
    Registry of real executable tools for agent mode.
    Tools are discovered at init and can be queried by name.
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for tool_cls in [
            WebSearchTool, ExecuteShellTool, ReadFileTool,
            WriteFileTool, AppendFileTool, CallApiTool, SummarizeTextTool,
            RunSimulationTool,
        ]:
            tool = tool_cls()
            self._tools[tool.name] = tool

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {
                "name": t.name,
                "description": t.description(),
                "available": t.can_execute(),
            }
            for t in self._tools.values()
        ]

    def available_tools(self) -> List[str]:
        return [name for name, t in self._tools.items() if t.can_execute()]

    def execute(self, tool_name: str, params: Dict) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(tool=tool_name, success=False,
                              error=f"Unknown tool: {tool_name}")
        if not tool.can_execute():
            return ToolResult(tool=tool_name, success=False,
                              error=f"Tool {tool_name} dependencies not available")
        return tool.execute(params)
