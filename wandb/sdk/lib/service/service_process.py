"""Reliably launch and connect to backend server process (wandb service).

Backend server process can be connected to using tcp sockets transport.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING

from wandb import _sentry
from wandb.env import (
    core_debug,
    dcgm_profiling_enabled,
    error_reporting_enabled,
    is_require_legacy_service,
)
from wandb.errors import Error, WandbCoreNotAvailableError
from wandb.errors.term import termwarn
from wandb.sdk.service import port_file
from wandb.util import get_core_path

from . import service_token

if TYPE_CHECKING:
    from wandb.sdk.wandb_settings import Settings


class ServiceStartProcessError(Error):
    """Raised when a known error occurs when launching wandb service."""


class ServiceStartTimeoutError(Error):
    """Raised when service start times out."""


class ServiceStartPortError(Error):
    """Raised when service start fails to find a port."""


def start(settings: Settings) -> ServiceProcess:
    """Start the internal service process.

    Returns:
        A handle to the process.

    Raises:
        ServiceStartProcessError: if the process dies on startup.
        ServiceStartTimeoutError: if the process fails to become healthy.
        ServiceStartPortError: if we cannot connect to the process.
    """
    _sentry.configure_scope(tags=dict(settings), process_context="service")

    try:
        return _launch_server(settings)
    except Exception as e:
        _sentry.reraise(e)


class ServiceProcess:
    """A handle to a process running the internal service."""

    def __init__(
        self,
        *,
        connection_token: service_token.ServiceToken,
        process: subprocess.Popen,
    ) -> None:
        self._token = connection_token
        self._process = process

    @property
    def token(self) -> service_token.ServiceToken:
        """A token for connecting to the process."""
        return self._token

    def join(self) -> int:
        """Wait for the process to end and return its exit code."""
        return self._process.wait()


def _wait_for_ports(
    fname: str,
    proc: subprocess.Popen,
    settings: Settings,
) -> int:
    """Wait for the service to write the port file and then read it.

    Args:
        fname: The path to the port file.
        proc: The process to wait for.
        settings: W&B settings.

    Returns:
        The port number for connecting to the service process.

    Raises:
        ServiceStartTimeoutError: If the service takes too long to start.
        ServiceStartPortError: If the service writes an invalid port file or unable to read it.
        ServiceStartProcessError: If the service process exits unexpectedly.
    """
    time_max = time.monotonic() + settings.x_service_wait
    while time.monotonic() < time_max:
        if proc.poll():
            # process finished
            # define these variables for sentry context grab:
            # command = proc.args
            # sys_executable = sys.executable
            # which_python = shutil.which("python3")
            # proc_out = proc.stdout.read()
            # proc_err = proc.stderr.read()
            context = dict(
                command=proc.args,
                sys_executable=sys.executable,
                which_python=shutil.which("python3"),
                proc_out=proc.stdout.read() if proc.stdout else "",
                proc_err=proc.stderr.read() if proc.stderr else "",
            )
            raise ServiceStartProcessError(
                f"The wandb service process exited with {proc.returncode}. "
                "Ensure that `sys.executable` is a valid python interpreter. "
                "You can override it with the `_executable` setting "
                "or with the `WANDB_X_EXECUTABLE` environment variable."
                f"\n{context}",
                context=context,
            )

        if not os.path.isfile(fname):
            time.sleep(0.2)
            continue

        try:
            pf = port_file.PortFile()
            pf.read(fname)
        except Exception as e:
            # todo: point at the docs. this could be due to a number of reasons,
            #  for example, being unable to write to the port file etc.
            raise ServiceStartPortError(
                f"Failed to allocate port for wandb service: {e}."
            )

        if not pf.is_valid:
            time.sleep(0.2)
            continue

        assert pf.sock_port
        return pf.sock_port

    raise ServiceStartTimeoutError(
        "Timed out waiting for wandb service to start after"
        f" {settings.x_service_wait} seconds."
        " Try increasing the timeout with the `_service_wait` setting."
    )


def _launch_server(settings: Settings) -> ServiceProcess:
    """Launch server and set ports."""
    if platform.system() == "Windows":
        creationflags: int = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        start_new_session = False
    else:
        creationflags = 0
        start_new_session = True

    pid = str(os.getpid())

    with tempfile.TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, f"port-{pid}.txt")

        executable = settings.x_executable
        assert executable

        exec_cmd_list: list[str] = [executable, "-m"]
        service_args: list[str] = []

        if not is_require_legacy_service():
            try:
                core_path = get_core_path()
            except WandbCoreNotAvailableError as e:
                _sentry.reraise(e)

            service_args.extend([core_path])

            if not error_reporting_enabled():
                service_args.append("--no-observability")

            if core_debug(default="False"):
                service_args.extend(["--log-level", "-4"])

            if dcgm_profiling_enabled():
                service_args.append("--enable-dcgm-profiling")

            exec_cmd_list = []
        else:
            service_args.extend(["wandb", "service", "--debug"])
            termwarn(
                "Using legacy-service, which is deprecated. If this is"
                " unintentional, you can fix it by ensuring you do not call"
                " `wandb.require('legacy-service')` and do not set the"
                " WANDB_X_REQUIRE_LEGACY_SERVICE environment"
                " variable."
            )

        service_args += [
            "--port-filename",
            fname,
            "--pid",
            pid,
        ]

        proc = subprocess.Popen(
            exec_cmd_list + service_args,
            env=os.environ,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
        port = _wait_for_ports(fname, proc, settings)

        token = service_token.TCPServiceToken(
            parent_pid=os.getpid(),
            port=port,
        )

        return ServiceProcess(connection_token=token, process=proc)
