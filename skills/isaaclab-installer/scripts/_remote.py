"""SSH/SFTP wrapper for the isaaclab-installer skill.

Single-host remote install support. Password-based auth only (key auth and
fleet orchestration are explicitly out of scope for v1 of remote mode).

Requires the ``paramiko`` package, which is NOT a runtime dependency of the
local install path. The remote helpers load paramiko lazily so local users
never have to install it.

    pip install --user paramiko    # or: sudo apt install python3-paramiko

The RemoteRunner exposes only what the rest of the skill needs:

    runner = RemoteRunner(host="10.0.0.5", user="bob", password=...)
    runner.connect()                  # one-time auth
    runner.put_directory(local, remote)   # SFTP the scripts up
    runner.run(cmd, cwd, env, log_fh, sudo_password=...)  # streamed exec
    runner.fetch_text(remote_path)    # read a file back
    runner.disconnect()

Streaming semantics match the local Popen path: stdout + stderr are merged
through a PTY, lines are flushed to the local terminal AND the log file in
real time, the exit code is returned at the end.
"""

from __future__ import annotations

import getpass
import os
import posixpath
import shlex
import sys
import time
from pathlib import Path


PARAMIKO_INSTALL_HINT = (
    "Remote install requires the `paramiko` package, which is not bundled "
    "with this skill so that local installs stay zero-dependency. Install it "
    "with one of:\n"
    "  pip install --user paramiko\n"
    "  sudo apt install python3-paramiko\n"
)


def _require_paramiko():
    try:
        import paramiko  # noqa: F401
        return paramiko
    except ImportError:
        sys.stderr.write("\n" + PARAMIKO_INSTALL_HINT + "\n")
        raise SystemExit(2)


def parse_target(spec):
    """Split 'user@host[:port]' into (user, host, port). Defaults: $USER, port 22."""
    if not spec:
        raise SystemExit("Remote target must be of the form user@host[:port]")
    if "@" in spec:
        user, rest = spec.split("@", 1)
    else:
        user = os.environ.get("USER") or os.environ.get("LOGNAME") or "root"
        rest = spec
    if ":" in rest:
        host, port = rest.split(":", 1)
        try:
            port = int(port)
        except ValueError:
            raise SystemExit(f"Invalid port in remote target: {spec}")
    else:
        host = rest
        port = 22
    if not host:
        raise SystemExit(f"Empty host in remote target: {spec}")
    return user, host, port


def prompt_ssh_password(user, host):
    return getpass.getpass(f"SSH password for {user}@{host}: ")


def prompt_sudo_password(user, host):
    return getpass.getpass(f"sudo password for {user}@{host} (press Enter if passwordless): ")


class RemoteRunner:
    """Thin paramiko wrapper that runs commands as if they were local Popen."""

    def __init__(self, host, user, password, port=22, timeout=30, keepalive=30):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.timeout = timeout
        self.keepalive = keepalive
        self._paramiko = None
        self._client = None
        self._sftp = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self):
        paramiko = _require_paramiko()
        self._paramiko = paramiko
        client = paramiko.SSHClient()
        # AutoAddPolicy is acceptable for one-off setup. We warn the user.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                timeout=self.timeout,
                banner_timeout=self.timeout,
                auth_timeout=self.timeout,
                allow_agent=False,
                look_for_keys=False,
            )
        except paramiko.AuthenticationException:
            raise SystemExit(f"Authentication failed for {self.user}@{self.host}.")
        except paramiko.SSHException as e:
            raise SystemExit(f"SSH error connecting to {self.user}@{self.host}: {e}")
        except OSError as e:
            raise SystemExit(f"Could not reach {self.host}:{self.port}: {e}")
        transport = client.get_transport()
        if transport is not None:
            transport.set_keepalive(self.keepalive)
        self._client = client

    def disconnect(self):
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # Context manager sugar
    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

    # ------------------------------------------------------------------
    # SFTP
    # ------------------------------------------------------------------

    def _ensure_sftp(self):
        if self._sftp is None:
            if self._client is None:
                raise RuntimeError("connect() must be called before SFTP operations.")
            self._sftp = self._client.open_sftp()

    def remote_home(self):
        """Resolve the remote user's $HOME via an actual exec."""
        rc, out, _ = self.capture("echo $HOME")
        if rc != 0 or not out.strip():
            raise SystemExit(f"Could not resolve remote $HOME: rc={rc}")
        return out.strip().splitlines()[-1].strip()

    def remote_mkdirs(self, path):
        """Recursive mkdir over SFTP (paramiko has no mkdir -p)."""
        self._ensure_sftp()
        assert self._sftp is not None
        parts = [p for p in path.split("/") if p]
        cur = "" if not path.startswith("/") else ""
        for p in parts:
            cur = cur + "/" + p if path.startswith("/") else (cur + "/" + p if cur else p)
            try:
                self._sftp.stat(cur)
            except IOError:
                self._sftp.mkdir(cur)

    def put_file(self, local_path, remote_path):
        self._ensure_sftp()
        assert self._sftp is not None
        parent = posixpath.dirname(remote_path)
        if parent:
            self.remote_mkdirs(parent)
        self._sftp.put(str(local_path), remote_path)
        # Preserve executable bit for .py / .sh
        try:
            local_mode = Path(local_path).stat().st_mode & 0o777
            self._sftp.chmod(remote_path, local_mode)
        except Exception:
            pass

    def put_directory(self, local_dir, remote_dir):
        """Recursively upload local_dir into remote_dir. Skips __pycache__."""
        local_dir = Path(local_dir)
        for root, dirs, files in os.walk(local_dir):
            # Don't upload caches
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            rel = os.path.relpath(root, local_dir)
            if rel == ".":
                rdir = remote_dir
            else:
                rdir = posixpath.join(remote_dir, rel.replace(os.sep, "/"))
            self.remote_mkdirs(rdir)
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                lpath = os.path.join(root, f)
                rpath = posixpath.join(rdir, f)
                self.put_file(lpath, rpath)

    def fetch_text(self, remote_path):
        self._ensure_sftp()
        assert self._sftp is not None
        with self._sftp.open(remote_path, "r") as f:
            return f.read().decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Exec
    # ------------------------------------------------------------------

    def _build_command(self, cmd, cwd=None, env=None):
        prefix = ""
        if env:
            prefix = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + " "
        body = cmd
        if cwd:
            body = f"cd {shlex.quote(cwd)} && {body}"
        # Force a login-ish shell so PATH includes ~/.local/bin etc.
        return f"bash -lc {shlex.quote(prefix + body)}"

    def _stream_channel(self, channel, log_fh):
        """Drain a paramiko channel: write to stdout + log until exit."""
        # Combine merged via get_pty=True; we still poll both streams defensively.
        buf_remaining = True
        while buf_remaining:
            ready_stdout = channel.recv_ready()
            ready_stderr = channel.recv_stderr_ready()
            if ready_stdout:
                try:
                    chunk = channel.recv(4096).decode("utf-8", errors="replace")
                except Exception:
                    chunk = ""
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    if log_fh:
                        log_fh.write(chunk)
                        log_fh.flush()
            if ready_stderr:
                try:
                    chunk = channel.recv_stderr(4096).decode("utf-8", errors="replace")
                except Exception:
                    chunk = ""
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                    if log_fh:
                        log_fh.write(chunk)
                        log_fh.flush()
            if channel.exit_status_ready() and not ready_stdout and not ready_stderr:
                buf_remaining = False
            elif not ready_stdout and not ready_stderr:
                time.sleep(0.05)
        return channel.recv_exit_status()

    def capture(self, cmd, cwd=None, env=None, timeout=120):
        """Run cmd, capture stdout/stderr without streaming. Returns (rc, out, err)."""
        if self._client is None:
            raise RuntimeError("connect() must be called first.")
        full = self._build_command(cmd, cwd=cwd, env=env)
        stdin, stdout, stderr = self._client.exec_command(full, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        rc = stdout.channel.recv_exit_status()
        return rc, out, err

    def run(self, cmd, cwd=None, env=None, log_fh=None, sudo_password=None):
        """Run a command, streaming output. Returns exit code.

        If the command begins with `sudo` and a sudo_password is supplied, we
        rewrite the command to `sudo -S -p '' ...` and feed the password via
        stdin. If the command begins with `sudo` and no password is supplied,
        we use `sudo -n` (non-interactive) — useful when the remote user has
        NOPASSWD sudo.
        """
        if self._client is None:
            raise RuntimeError("connect() must be called first.")
        stripped = cmd.lstrip()
        if stripped.startswith("sudo "):
            tail = stripped[len("sudo "):]
            if sudo_password:
                cmd = f"sudo -S -p '' {tail}"
            else:
                cmd = f"sudo -n {tail}"
        full = self._build_command(cmd, cwd=cwd, env=env)
        if log_fh:
            log_fh.write(f"\n=== remote {self.user}@{self.host} $ {full}\n")
            log_fh.flush()
        stdin, stdout, stderr = self._client.exec_command(full, get_pty=True, timeout=None)
        if stripped.startswith("sudo ") and sudo_password:
            try:
                stdin.write(sudo_password + "\n")
                stdin.flush()
            except OSError:
                pass
        channel = stdout.channel
        return self._stream_channel(channel, log_fh)


# ---------------------------------------------------------------------------
# Helper for the CLI scripts: prompt + connect.
# ---------------------------------------------------------------------------

def open_runner(target):
    """Given a `user@host[:port]` string, prompt for password and connect.

    Returns a connected RemoteRunner. Caller is responsible for disconnect().
    """
    user, host, port = parse_target(target)
    sys.stderr.write(f"\n[remote] Connecting to {user}@{host}:{port}\n")
    sys.stderr.write("[remote] (host key will be auto-added on first connect — verify the host fingerprint out-of-band)\n")
    password = prompt_ssh_password(user, host)
    runner = RemoteRunner(host=host, user=user, password=password, port=port)
    runner.connect()
    sys.stderr.write("[remote] Connected.\n")
    return runner
