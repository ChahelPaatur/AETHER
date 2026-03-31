"""Tests for AutoUpdater GitHub release checking."""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from aether.core.auto_updater import AutoUpdater, _parse_version


class TestParseVersion:
    def test_simple(self):
        assert _parse_version("3.0") == (3, 0)

    def test_v_prefix(self):
        assert _parse_version("v3.1.2") == (3, 1, 2)

    def test_uppercase_v(self):
        assert _parse_version("V4.0") == (4, 0)

    def test_empty_fallback(self):
        assert _parse_version("") == (0,)

    def test_comparison(self):
        assert _parse_version("v3.1") > _parse_version("3.0")
        assert _parse_version("3.0") == _parse_version("v3.0")
        assert _parse_version("3.0.1") > _parse_version("3.0")


class TestGetCurrentVersion:
    def test_reads_version_file(self, tmp_path):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        assert updater.get_current_version() == "3.0"

    def test_fallback_when_no_file(self, tmp_path):
        updater = AutoUpdater(project_root=str(tmp_path))
        assert updater.get_current_version() == "3.0"


class TestFetchLatestRelease:
    def test_returns_none_without_requests(self, tmp_path):
        updater = AutoUpdater(project_root=str(tmp_path))
        with patch.dict(sys.modules, {"requests": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert updater._fetch_latest_release() is None

    def test_returns_json_on_200(self, tmp_path):
        updater = AutoUpdater(project_root=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"tag_name": "v4.0", "body": "notes"}

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp

        with patch.dict(sys.modules, {"requests": mock_requests}):
            result = updater._fetch_latest_release()
            assert result == {"tag_name": "v4.0", "body": "notes"}

    def test_returns_none_on_404(self, tmp_path):
        updater = AutoUpdater(project_root=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        mock_requests = MagicMock()
        mock_requests.get.return_value = mock_resp

        with patch.dict(sys.modules, {"requests": mock_requests}):
            assert updater._fetch_latest_release() is None

    def test_returns_none_on_timeout(self, tmp_path):
        updater = AutoUpdater(project_root=str(tmp_path))
        mock_requests = MagicMock()
        mock_requests.get.side_effect = Exception("timeout")

        with patch.dict(sys.modules, {"requests": mock_requests}):
            assert updater._fetch_latest_release() is None


class TestCheckForUpdate:
    def test_newer_version_returns_tag(self, tmp_path):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater._fetch_latest_release = MagicMock(return_value={
            "tag_name": "v3.1",
            "body": "Bug fixes and improvements",
        })
        result = updater.check_for_update()
        assert result == "v3.1"
        assert "Bug fixes" in updater._release_notes

    def test_same_version_returns_none(self, tmp_path):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater._fetch_latest_release = MagicMock(return_value={
            "tag_name": "v3.0",
            "body": "",
        })
        assert updater.check_for_update() is None

    def test_older_version_returns_none(self, tmp_path):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater._fetch_latest_release = MagicMock(return_value={
            "tag_name": "v2.5",
            "body": "",
        })
        assert updater.check_for_update() is None

    def test_api_failure_returns_none(self, tmp_path):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater._fetch_latest_release = MagicMock(return_value=None)
        assert updater.check_for_update() is None

    def test_empty_tag_returns_none(self, tmp_path):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater._fetch_latest_release = MagicMock(return_value={
            "tag_name": "",
            "body": "",
        })
        assert updater.check_for_update() is None


class TestRun:
    def test_no_update_skips(self, tmp_path):
        updater = AutoUpdater(project_root=str(tmp_path))
        assert updater.run(no_update=True) is False

    def test_up_to_date(self, tmp_path, capsys):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater.check_for_update = MagicMock(return_value=None)
        assert updater.run() is False
        assert "up to date" in capsys.readouterr().out

    def test_update_declined(self, tmp_path, capsys):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater.check_for_update = MagicMock(return_value="v4.0")
        updater._release_notes = "New features"

        with patch("builtins.input", return_value="n"):
            assert updater.run() is False
        out = capsys.readouterr().out
        assert "v4.0" in out
        assert "Skipped" in out

    def test_update_prints_release_notes(self, tmp_path, capsys):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater.check_for_update = MagicMock(return_value="v4.0")
        updater._release_notes = "Fixed critical bug\nAdded new feature"

        with patch("builtins.input", return_value="n"):
            updater.run()
        out = capsys.readouterr().out
        assert "Fixed critical bug" in out
        assert "Added new feature" in out

    def test_auto_update_calls_apply(self, tmp_path):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater.check_for_update = MagicMock(return_value="v4.0")
        updater._release_notes = ""
        updater._apply_update = MagicMock(return_value=True)

        result = updater.run(auto_update=True)
        updater._apply_update.assert_called_once()

    def test_eof_declines_update(self, tmp_path, capsys):
        vf = tmp_path / "VERSION"
        vf.write_text("3.0\n")
        updater = AutoUpdater(project_root=str(tmp_path))
        updater.check_for_update = MagicMock(return_value="v4.0")
        updater._release_notes = ""

        with patch("builtins.input", side_effect=EOFError):
            assert updater.run() is False
        assert "Skipped" in capsys.readouterr().out


class TestApplyUpdate:
    def test_git_pull_origin_main(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        updater = AutoUpdater(project_root=str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            with patch("os.execv") as mock_execv:
                updater._apply_update()
                mock_run.assert_called_once()
                args = mock_run.call_args
                assert args[0][0] == ["git", "pull", "origin", "main"]
                mock_execv.assert_called_once()

    def test_git_pull_failure(self, tmp_path, capsys):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        updater = AutoUpdater(project_root=str(tmp_path))

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "merge conflict"

        with patch("subprocess.run", return_value=mock_result):
            assert updater._apply_update() is False
        assert "failed" in capsys.readouterr().out

    def test_no_git_directory(self, tmp_path, capsys):
        updater = AutoUpdater(project_root=str(tmp_path))
        assert updater._apply_update() is False
        assert "No .git directory" in capsys.readouterr().out
