"""Security tests: shell injection, rate limiting, API key auth."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.tool_registry import ExecuteShellTool


class TestShellInjection:
    """Verify shell metacharacters are blocked."""

    def test_semicolon_blocked(self):
        tool = ExecuteShellTool()
        result = tool.execute({"command": "echo hello; rm -rf /"})
        assert result.success is False
        assert "metacharacter" in result.error.lower() or "blocked" in result.error.lower()

    def test_pipe_blocked(self):
        tool = ExecuteShellTool()
        result = tool.execute({"command": "cat /etc/passwd | grep root"})
        assert result.success is False

    def test_backtick_blocked(self):
        tool = ExecuteShellTool()
        result = tool.execute({"command": "echo `whoami`"})
        assert result.success is False

    def test_subshell_blocked(self):
        tool = ExecuteShellTool()
        result = tool.execute({"command": "echo $(id)"})
        assert result.success is False

    def test_safe_command_allowed(self):
        tool = ExecuteShellTool()
        result = tool.execute({"command": "pwd"})
        assert result.success is True

    def test_existing_blocked_patterns_still_work(self):
        tool = ExecuteShellTool()
        result = tool.execute({"command": "rm -rf /"})
        assert result.success is False

    def test_list_files_still_works(self):
        tool = ExecuteShellTool()
        result = tool.execute({"command": "list files in ."})
        assert result.success is True


class TestGitignoreSecrets:
    """Verify .gitignore covers secret files."""

    def test_env_in_gitignore(self):
        gitignore = os.path.join(
            os.path.dirname(__file__), "..", ".gitignore")
        with open(gitignore) as f:
            content = f.read()
        assert ".env" in content

    def test_secrets_json_in_gitignore(self):
        gitignore = os.path.join(
            os.path.dirname(__file__), "..", ".gitignore")
        with open(gitignore) as f:
            content = f.read()
        assert "secrets.json" in content


class TestNoHardcodedSecrets:
    """Scan source files for hardcoded API keys."""

    def test_no_hardcoded_keys_in_core(self):
        core_dir = os.path.join(
            os.path.dirname(__file__), "..", "aether", "core")
        import re
        key_patterns = [
            re.compile(r'sk-[a-zA-Z0-9]{20,}'),           # Anthropic/OpenAI
            re.compile(r'api_key\s*=\s*["\'][a-zA-Z0-9]'), # hardcoded key
            re.compile(r'password\s*=\s*["\'][^"\']+["\']'), # passwords
        ]

        violations = []
        for fname in os.listdir(core_dir):
            if not fname.endswith(".py"):
                continue
            path = os.path.join(core_dir, fname)
            with open(path) as f:
                for i, line in enumerate(f, 1):
                    for pat in key_patterns:
                        if pat.search(line):
                            violations.append(f"{fname}:{i}")

        assert violations == [], f"Potential hardcoded secrets: {violations}"
