"""A managed, loopback-only Rerun web asset server."""

import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class Viewer:
    def __init__(self, cache: Path):
        self.cache = cache
        self.port = None
        self.process = None
        self.lock = threading.Lock()

    def start(self) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return
            binary = Path(sys.executable).with_name("rerun")
            if not binary.is_file():
                raise RuntimeError("Rerun is missing. Run `pixi install -e gui`.")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                self.port = sock.getsockname()[1]
            log_path = self.cache / "viewer.log"
            with log_path.open("wb") as log:
                self.process = subprocess.Popen(
                    [
                        str(binary),
                        "--bind",
                        "127.0.0.1",
                        "--serve-web",
                        "--web-viewer-port",
                        str(self.port),
                        # A directly viewable URI prevents starting an extra gRPC
                        # server. Each iframe supplies its actual recording URL.
                        "http://127.0.0.1/kiwi-no-selection.rrd",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            for _ in range(300):
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"Rerun web server stopped: {log_path.read_text()[-1500:]}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.port}/", timeout=1
                    ):
                        return
                except (OSError, urllib.error.URLError):
                    time.sleep(0.1)
            self.close()
            raise RuntimeError(f"Rerun web viewer did not start; inspect {log_path}.")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
