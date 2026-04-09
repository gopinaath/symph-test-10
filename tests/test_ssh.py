"""Tests for SSH utilities."""

import os
import pytest
from symphony.ssh import parse_ssh_target, build_ssh_command, remote_shell_command, SSHTarget


class TestParseSSHTarget:
    def test_host_only(self):
        t = parse_ssh_target("example.com")
        assert t == SSHTarget(user=None, host="example.com", port=None)

    def test_host_port(self):
        t = parse_ssh_target("example.com:2222")
        assert t == SSHTarget(user=None, host="example.com", port=2222)

    def test_user_host(self):
        t = parse_ssh_target("root@example.com")
        assert t == SSHTarget(user="root", host="example.com", port=None)

    def test_user_host_port(self):
        t = parse_ssh_target("root@127.0.0.1:2200")
        assert t == SSHTarget(user="root", host="127.0.0.1", port=2200)

    def test_bracketed_ipv6_with_port(self):
        t = parse_ssh_target("root@[::1]:2200")
        assert t == SSHTarget(user="root", host="[::1]", port=2200)

    def test_unbracketed_ipv6(self):
        t = parse_ssh_target("::1:2200")
        # multiple colons without brackets - treated as host, not split
        assert t.host == "::1:2200"
        assert t.port is None


class TestBuildSSHCommand:
    def test_basic_command(self):
        cmd = build_ssh_command("example.com", "ls -la")
        assert cmd[0].endswith("ssh")
        assert "-T" in cmd
        assert "example.com" in cmd

    def test_with_port(self):
        cmd = build_ssh_command("example.com:2222", "ls")
        assert "-p" in cmd
        idx = cmd.index("-p")
        assert cmd[idx + 1] == "2222"

    def test_with_user(self):
        cmd = build_ssh_command("root@example.com", "ls")
        assert "root@example.com" in cmd

    def test_with_ssh_config(self):
        cmd = build_ssh_command("example.com", "ls", ssh_config="/path/to/config")
        assert "-F" in cmd
        idx = cmd.index("-F")
        assert cmd[idx + 1] == "/path/to/config"

    def test_ssh_config_from_env(self, monkeypatch):
        monkeypatch.setenv("SYMPHONY_SSH_CONFIG", "/env/config")
        cmd = build_ssh_command("example.com", "ls")
        assert "-F" in cmd
        idx = cmd.index("-F")
        assert cmd[idx + 1] == "/env/config"

    def test_ssh_not_found(self, monkeypatch):
        monkeypatch.setenv("PATH", "")
        with pytest.raises(FileNotFoundError, match="ssh not found"):
            build_ssh_command("example.com", "ls")


class TestRemoteShellCommand:
    def test_basic_wrapping(self):
        result = remote_shell_command("echo hello")
        assert result == "bash -lc 'echo hello'"

    def test_escapes_single_quotes(self):
        result = remote_shell_command("echo 'hello world'")
        assert "'\\''" in result


class TestLogFile:
    def test_default_log_file_uses_cwd(self):
        from symphony.log_file import default_log_file
        result = default_log_file()
        assert result.endswith(os.path.join("log", "symphony.log"))
        assert os.getcwd() in result

    def test_default_log_file_custom_root(self):
        from symphony.log_file import default_log_file
        result = default_log_file("/custom/root")
        assert result == "/custom/root/log/symphony.log"


class TestCLI:
    def test_missing_ack_flag_returns_error(self):
        import subprocess
        result = subprocess.run(
            ["python", "-m", "symphony.cli"],
            capture_output=True, text=True, cwd="/home/jerry/dev/scratch/karpathy/symph-test-10",
        )
        assert result.returncode == 1
        assert "guardrails" in result.stderr.lower() or "guardrails" in result.stdout.lower()

    def test_missing_workflow_returns_error(self, tmp_path):
        import subprocess
        result = subprocess.run(
            [
                "python", "-m", "symphony.cli",
                "--i-understand-that-this-will-be-running-without-the-usual-guardrails",
                str(tmp_path / "nonexistent.md"),
            ],
            capture_output=True, text=True, cwd="/home/jerry/dev/scratch/karpathy/symph-test-10",
        )
        assert result.returncode == 1
        assert "not found" in result.stderr.lower()
