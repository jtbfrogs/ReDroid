"""
utils/diag_manager.py
─────────────────────
Pre-flight diagnostic system. Performs hardware and service reachability
checks before the main firmware loop starts, then writes a timestamped
.diag report to the /diagnostics folder.

Pre-flight checks:
  1. Serial port accessibility (Roomba UART)
  2. USB/CSI Camera accessibility (OpenCV)
  3. Ollama API reachability (HTTP)

The DiagManager.run_preflight() method returns a DiagReport dataclass
so main.py can make informed decisions about which subsystems to enable.

Usage:
    from utils.diag_manager import DiagManager
    from utils.logger import get_logger
    report = DiagManager(config, get_logger("diag")).run_preflight()
    if not report.roomba_ok:
        # disable drive features gracefully
"""

import os
import json
import socket
import platform
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import serial          # pyserial
import cv2
import requests

from utils.logger import get_logger

log = get_logger("diag")

# ── Diagnostics output directory ─────────────────────────────────────────────
_DIAG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diagnostics")


# ── Result Data Classes ───────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    latency_ms: Optional[float] = None


@dataclass
class DiagReport:
    timestamp: str
    hostname: str
    platform: str
    python_version: str

    roomba_ok: bool = False
    camera_ok: bool = False
    ollama_ok: bool = False
    all_pass: bool = False

    checks: list[CheckResult] = field(default_factory=list)
    diag_file: str = ""

    def summary(self) -> str:
        icons = {True: "✔", False: "✘"}
        lines = [
            f"  Roomba Serial  : {icons[self.roomba_ok]}",
            f"  Camera (CV2)   : {icons[self.camera_ok]}",
            f"  Ollama API     : {icons[self.ollama_ok]}",
            f"  Overall        : {'PASS ✔' if self.all_pass else 'FAIL ✘'}",
        ]
        return "\n".join(lines)


# ── DiagManager ───────────────────────────────────────────────────────────────

class DiagManager:
    """
    Runs pre-flight checks and writes .diag files to /diagnostics.

    Args:
        config: Parsed YAML config dict (top-level keys: system, features, ollama).
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._diag_dir = _DIAG_DIR
        os.makedirs(self._diag_dir, exist_ok=True)

    # ── Public Entry Point ────────────────────────────────────────────────────

    def run_preflight(self) -> DiagReport:
        """
        Execute all pre-flight checks, log results, write the .diag file,
        and return a DiagReport.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = DiagReport(
            timestamp=datetime.now().isoformat(),
            hostname=socket.gethostname(),
            platform=platform.platform(),
            python_version=platform.python_version(),
        )

        log.info("━" * 52)
        log.info("  PRE-FLIGHT DIAGNOSTIC SEQUENCE INITIATED")
        log.info("━" * 52)

        checks: list[CheckResult] = []

        # Run each check
        checks.append(self._check_serial())
        checks.append(self._check_camera())
        checks.append(self._check_ollama())

        # Populate report
        report.checks = checks
        for c in checks:
            if c.name == "serial":
                report.roomba_ok = c.passed
            elif c.name == "camera":
                report.camera_ok = c.passed
            elif c.name == "ollama":
                report.ollama_ok = c.passed

        report.all_pass = all(c.passed for c in checks)

        # Log summary
        log.info("━" * 52)
        for c in checks:
            icon = "✔" if c.passed else "✘"
            lat = f" [{c.latency_ms:.0f}ms]" if c.latency_ms is not None else ""
            level = log.info if c.passed else log.warning
            level(f"  {icon}  {c.name.upper():<16} {c.detail}{lat}")
        log.info("━" * 52)

        if report.all_pass:
            log.info("  ALL SYSTEMS NOMINAL — CLEARED FOR LAUNCH ✔")
        else:
            log.warning("  ONE OR MORE CHECKS FAILED — REVIEW ABOVE ✘")
        log.info("━" * 52)

        # Write .diag file
        report.diag_file = self._write_diag_file(ts, report)
        log.info(f"  Diagnostics saved → {report.diag_file}")

        return report

    # ── Individual Checks ─────────────────────────────────────────────────────

    def _check_serial(self) -> CheckResult:
        """Try to open the configured serial port and immediately close it."""
        port = self._cfg["system"]["port"]
        baud = self._cfg["system"]["baud"]
        timeout = self._cfg["system"].get("roomba_timeout", 2.0)

        log.debug(f"Checking serial port: {port} @ {baud} baud …")
        t0 = datetime.now()
        try:
            ser = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=timeout,
                rtscts=False,
                dsrdtr=False,
            )
            if ser.isOpen():
                ser.close()
                latency = (datetime.now() - t0).total_seconds() * 1000
                return CheckResult(
                    name="serial",
                    passed=True,
                    detail=f"Port {port} opened successfully.",
                    latency_ms=latency,
                )
            else:
                return CheckResult(
                    name="serial",
                    passed=False,
                    detail=f"Port {port} did not open.",
                )
        except serial.SerialException as e:
            return CheckResult(
                name="serial",
                passed=False,
                detail=f"SerialException: {e}",
            )
        except Exception as e:
            return CheckResult(
                name="serial",
                passed=False,
                detail=f"Unexpected error: {type(e).__name__}: {e}",
            )

    def _check_camera(self) -> CheckResult:
        """Try to open the configured camera index and grab one frame."""
        if not self._cfg.get("features", {}).get("enable_vision", True):
            return CheckResult(
                name="camera",
                passed=True,
                detail="Vision disabled in config — check skipped.",
            )

        cam_index = self._cfg["features"].get("camera_index", 0)
        log.debug(f"Checking camera index: {cam_index} …")
        t0 = datetime.now()
        cap = None
        try:
            cap = cv2.VideoCapture(cam_index)
            if not cap.isOpened():
                return CheckResult(
                    name="camera",
                    passed=False,
                    detail=f"cv2.VideoCapture({cam_index}) failed to open.",
                )
            ret, frame = cap.read()
            latency = (datetime.now() - t0).total_seconds() * 1000
            if ret and frame is not None:
                h, w = frame.shape[:2]
                return CheckResult(
                    name="camera",
                    passed=True,
                    detail=f"Camera {cam_index} open — frame size {w}×{h}.",
                    latency_ms=latency,
                )
            else:
                return CheckResult(
                    name="camera",
                    passed=False,
                    detail=f"Camera {cam_index} opened but returned no frame.",
                    latency_ms=latency,
                )
        except Exception as e:
            return CheckResult(
                name="camera",
                passed=False,
                detail=f"Exception: {type(e).__name__}: {e}",
            )
        finally:
            if cap is not None:
                cap.release()

    def _check_ollama(self) -> CheckResult:
        """Send a lightweight HTTP GET to the Ollama /api/tags endpoint."""
        if not self._cfg.get("features", {}).get("enable_chat", True):
            return CheckResult(
                name="ollama",
                passed=True,
                detail="Chat disabled in config — check skipped.",
            )

        host = self._cfg.get("ollama", {}).get("host", "http://localhost:11434")
        model = self._cfg.get("ollama", {}).get("model", "llama3.2")
        url = f"{host}/api/tags"
        log.debug(f"Checking Ollama at {url} …")
        t0 = datetime.now()
        try:
            resp = requests.get(url, timeout=5)
            latency = (datetime.now() - t0).total_seconds() * 1000
            if resp.status_code == 200:
                data = resp.json()
                available = [m.get("name", "") for m in data.get("models", [])]
                model_found = any(model in m for m in available)
                detail = (
                    f"Ollama reachable. Model '{model}' {'✔ found' if model_found else '✘ NOT found — run: ollama pull ' + model}."
                )
                return CheckResult(
                    name="ollama",
                    passed=model_found,
                    detail=detail,
                    latency_ms=latency,
                )
            else:
                return CheckResult(
                    name="ollama",
                    passed=False,
                    detail=f"HTTP {resp.status_code} from {url}.",
                    latency_ms=latency,
                )
        except requests.ConnectionError:
            return CheckResult(
                name="ollama",
                passed=False,
                detail=f"Connection refused at {host}. Is Ollama running?",
            )
        except requests.Timeout:
            return CheckResult(
                name="ollama",
                passed=False,
                detail=f"Request to {host} timed out after 5s.",
            )
        except Exception as e:
            return CheckResult(
                name="ollama",
                passed=False,
                detail=f"Exception: {type(e).__name__}: {e}",
            )

    # ── File Writer ───────────────────────────────────────────────────────────

    def _write_diag_file(self, timestamp: str, report: DiagReport) -> str:
        """Serialize the DiagReport to a human-readable .diag file."""
        filename = f"preflight_{timestamp}.diag"
        filepath = os.path.join(self._diag_dir, filename)

        # Convert checks list to serializable format
        report_dict = {
            "meta": {
                "timestamp": report.timestamp,
                "hostname": report.hostname,
                "platform": report.platform,
                "python_version": report.python_version,
            },
            "results": {
                "roomba_serial": report.roomba_ok,
                "camera": report.camera_ok,
                "ollama_api": report.ollama_ok,
                "all_systems_pass": report.all_pass,
            },
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "detail": c.detail,
                    "latency_ms": round(c.latency_ms, 2) if c.latency_ms else None,
                }
                for c in report.checks
            ],
        }

        # Write as formatted text with JSON payload
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  DROID FIRMWARE — PRE-FLIGHT DIAGNOSTIC REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"  Generated : {report.timestamp}\n")
            f.write(f"  Host      : {report.hostname}\n")
            f.write(f"  Platform  : {report.platform}\n")
            f.write(f"  Python    : {report.python_version}\n")
            f.write("=" * 60 + "\n\n")
            f.write("SUMMARY\n")
            f.write("─" * 40 + "\n")
            f.write(report.summary() + "\n\n")
            f.write("DETAILED RESULTS (JSON)\n")
            f.write("─" * 40 + "\n")
            f.write(json.dumps(report_dict, indent=2))
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("  END OF REPORT\n")
            f.write("=" * 60 + "\n")

        return filepath

    # ── Utility: Run a named check in isolation ───────────────────────────────

    def run_single_check(self, check_name: str) -> CheckResult:
        """
        Run a single named check. Useful for runtime health monitoring.

        Args:
            check_name: One of "serial", "camera", "ollama".
        """
        dispatch = {
            "serial": self._check_serial,
            "camera": self._check_camera,
            "ollama": self._check_ollama,
        }
        fn = dispatch.get(check_name.lower())
        if fn is None:
            return CheckResult(
                name=check_name,
                passed=False,
                detail=f"Unknown check name '{check_name}'.",
            )
        return fn()
