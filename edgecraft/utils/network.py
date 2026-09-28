import subprocess
import socket
import re
from typing import Tuple, Optional


def validate_ssh_target(device_ip: str) -> str:
    """Validate the single ``[user@]host`` form accepted by edge commands."""
    target = str(device_ip or "")
    if (
        not target
        or target != target.strip()
        or len(target) > 320
        or target.startswith("-")
        or any(character.isspace() for character in target)
        or target.count("@") > 1
    ):
        raise ValueError("device target must use the [user@]host form")
    user, host = target.split("@", 1) if "@" in target else ("", target)
    if user and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,63}", user):
        raise ValueError("device target contains an invalid SSH user")
    hostname = bool(
        re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,252}", host)
        and host not in {".", ".."}
    )
    bracketed_ipv6 = bool(re.fullmatch(r"\[[0-9A-Fa-f:.]+\]", host))
    if not (hostname or bracketed_ipv6):
        raise ValueError("device target contains an invalid SSH host")
    return target


def _ssh_base_args(device_ip: str, ssh_key: Optional[str] = None) -> list:
    """Build base SSH command list for device_ip (user@host or host)."""
    if not ssh_key:
        raise ValueError("an explicit SSH private key is required for edge connections")
    target = validate_ssh_target(device_ip)
    ssh_cmd = ["ssh", "-i", ssh_key]
    ssh_cmd.extend([
        "-o", "IdentitiesOnly=yes",
        "-o", "ConnectTimeout=10",
        "-o", "BatchMode=yes",
        "-o", "LogLevel=ERROR",
    ])
    ssh_cmd.append(target)
    return ssh_cmd


def run_remote_command(device_ip: str, command: str, ssh_key: Optional[str] = None, timeout: int = 15) -> Tuple[int, str, str]:
    """Run a single command on the device via SSH.

    Args:
        device_ip: SSH target in format "user@host" or "host"
        command: Shell command to run remotely (will be passed to sh -c).
        ssh_key: Explicit path to the SSH private key.
        timeout: Command timeout in seconds.

    Returns:
        Tuple of (returncode, stdout, stderr).
    """
    cmd = _ssh_base_args(device_ip, ssh_key) + [command]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout or "", result.stderr or ""


def test_ssh_connection(device_ip: str, ssh_key: str = None, timeout: int = 5) -> Tuple[bool, str]:
    """Test SSH connection to device.

    Args:
        device_ip: SSH target in format "user@host" or "host"
        ssh_key: Explicit path to the SSH private key.
        timeout: Connection timeout in seconds

    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        target = validate_ssh_target(device_ip)
        # Parse device_ip format (user@host or just host)
        if "@" in target:
            user, host = target.split("@", 1)
        else:
            user = None
            host = target

        # First, test if host is reachable (ping or port check)
        try:
            # Try to resolve the hostname or bracketed IPv6 literal.
            resolution_host = host[1:-1] if host.startswith("[") else host
            socket.getaddrinfo(resolution_host, None)
        except socket.gaierror:
            return False, f"Cannot resolve hostname: {host}"

        # Build SSH command
        target = f"{user}@{host}" if user else host
        ssh_cmd = _ssh_base_args(target, ssh_key)
        ssh_cmd.append("echo 'SSH connection test successful'")

        # Run SSH command with timeout
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return True, "Connection successful"
        else:
            error_msg = result.stderr.strip() if result.stderr else "Connection failed"
            # Extract meaningful error message
            if "Permission denied" in error_msg:
                return False, "Permission denied (check SSH key or credentials)"
            elif "Connection refused" in error_msg:
                return False, "Connection refused (SSH service may not be running)"
            elif "Connection timed out" in error_msg or "timed out" in error_msg:
                return False, "Connection timed out (host may be unreachable)"
            elif "Could not resolve hostname" in error_msg:
                return False, f"Cannot resolve hostname: {host}"
            else:
                return False, f"Connection failed: {error_msg[:100]}"

    except subprocess.TimeoutExpired:
        return False, f"Connection timeout after {timeout}s"
    except Exception as e:
        return False, f"Error testing connection: {str(e)}"


def ensure_edge_runner_active(
    device_ip: str,
    ssh_key: Optional[str] = None,
    timeout: int = 10,
) -> Tuple[bool, str]:
    """Check device readiness for the bundled one-shot runner or a daemon."""
    check_cmd = (
        "if command -v bash >/dev/null 2>&1 && command -v tar >/dev/null 2>&1 "
        "&& command -v gzip >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1 "
        "&& command -v flock >/dev/null 2>&1 && test -w \"$HOME\"; then "
        "echo ready:inline; exit 0; fi; "
        "if command -v systemctl >/dev/null 2>&1; then "
        "state_main=$(systemctl is-active edgecraft-edge-runner.service 2>/dev/null || true); "
        "if [ \"$state_main\" = \"active\" ]; then echo \"ready:main_service\"; exit 0; fi; "
        "user_name=$(id -un 2>/dev/null || echo unknown); "
        "state_tpl_svc=$(systemctl is-active \"edgecraft-edge-runner@${user_name}.service\" 2>/dev/null || true); "
        "state_tpl_path=$(systemctl is-active \"edgecraft-edge-runner@${user_name}.path\" 2>/dev/null || true); "
        "if [ \"$state_tpl_path\" = \"active\" ]; then echo \"ready:template_path:${user_name}:${state_tpl_svc}\"; exit 0; fi; "
        "if [ \"$state_tpl_svc\" = \"active\" ]; then echo \"ready:template_service:${user_name}\"; exit 0; fi; "
        "active_any_path=$(systemctl list-units --type=path --all \"edgecraft-edge-runner@*.path\" --no-legend 2>/dev/null | awk '$3==\"active\"{print $1; exit}'); "
        "if [ -n \"$active_any_path\" ]; then echo \"ready:any_template_path:${active_any_path}\"; exit 0; fi; "
        "echo \"state:main=${state_main:-unknown};tpl_path=${state_tpl_path:-unknown};tpl_svc=${state_tpl_svc:-unknown}\"; "
        "fi; "
        "if pgrep -f \"edgecraft-edge-runner\" >/dev/null 2>&1; then echo \"ready:process\"; exit 0; fi; "
        "exit 1"
    )

    try:
        code, stdout, stderr = run_remote_command(
            device_ip=device_ip,
            command=check_cmd,
            ssh_key=ssh_key,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"Runner status check timeout after {timeout}s"
    except Exception as exc:
        return False, f"Runner status check failed: {exc}"

    out = (stdout or "").strip()
    if code == 0:
        if out == "ready:inline":
            return True, "Bundled one-shot runner prerequisites are available"
        if out.startswith("ready:template_path:"):
            parts = out.split(":")
            user_name = parts[2] if len(parts) > 2 else "unknown"
            return True, f"edgecraft-edge-runner@{user_name}.path is active (watch mode)"
        if out.startswith("ready:template_service:"):
            parts = out.split(":")
            user_name = parts[2] if len(parts) > 2 else "unknown"
            return True, f"edgecraft-edge-runner@{user_name}.service is active"
        if out.startswith("ready:any_template_path:"):
            unit = out.split(":", 2)[2] if ":" in out else "edgecraft-edge-runner@*.path"
            return True, f"{unit} is active (watch mode)"
        if out == "ready:main_service":
            return True, "edgecraft-edge-runner.service is active"
        return True, "edgecraft-edge-runner process is active"

    if out.startswith("state:"):
        return (
            False,
            "Edge runner is not active "
            f"({out}). Run: sudo systemctl enable --now edgecraft-edge-runner@$(id -un).path",
        )

    err = (stderr or "").strip()
    return False, (
        "Edge runner is not active (no service/process detected). "
        "Run: sudo systemctl enable --now edgecraft-edge-runner@$(id -un).path"
        + (f" | details: {err[:120]}" if err else "")
    )
