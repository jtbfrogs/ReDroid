"""
utils/logger.py
───────────────
Custom ANSI-colored terminal logger for the Droid firmware.
Provides a factory function get_logger() that returns a named Logger instance
pre-configured with both a colored StreamHandler and an optional FileHandler.

Usage:
    from utils.logger import get_logger
    log = get_logger("roomba_io")
    log.info("Motors initialized.")
    log.warning("Battery below 20%%.")
    log.error("Serial port unavailable.")
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


# ── ANSI Color Palette ────────────────────────────────────────────────────────

class _AnsiColor:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"

    # Foreground colors
    RED     = "\033[91m"
    YELLOW  = "\033[93m"
    GREEN   = "\033[92m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"

    # Background accents for CRITICAL
    BG_RED  = "\033[41m"


# ── Level → Color Mapping ─────────────────────────────────────────────────────

_LEVEL_STYLES: dict[int, str] = {
    logging.DEBUG:    _AnsiColor.GREY,
    logging.INFO:     _AnsiColor.CYAN,
    logging.WARNING:  _AnsiColor.YELLOW,
    logging.ERROR:    _AnsiColor.RED,
    logging.CRITICAL: _AnsiColor.BG_RED + _AnsiColor.WHITE + _AnsiColor.BOLD,
}

# Module name → accent color (cycles through palette for readability)
_MODULE_COLORS = [
    _AnsiColor.GREEN,
    _AnsiColor.MAGENTA,
    _AnsiColor.BLUE,
    _AnsiColor.YELLOW,
    _AnsiColor.CYAN,
    _AnsiColor.WHITE,
]
_module_color_cache: dict[str, str] = {}
_module_color_index: int = 0


def _get_module_color(name: str) -> str:
    global _module_color_index
    if name not in _module_color_cache:
        _module_color_cache[name] = _MODULE_COLORS[
            _module_color_index % len(_MODULE_COLORS)
        ]
        _module_color_index += 1
    return _module_color_cache[name]


# ── Custom Formatter ──────────────────────────────────────────────────────────

class _DroidFormatter(logging.Formatter):
    """
    Produces colored terminal output in the format:

      HH:MM:SS.mmm  [MODULE_NAME ]  LEVEL   message text
    """

    _WIDTH_MODULE = 14   # Fixed-width module name column
    _WIDTH_LEVEL  = 8    # Fixed-width level name column

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.") + \
             f"{int(record.msecs):03d}"
        ts_str = f"{_AnsiColor.DIM}{ts}{_AnsiColor.RESET}"

        # Module name (colored, fixed width)
        mod_color = _get_module_color(record.name)
        mod_str = f"{mod_color}{record.name:<{self._WIDTH_MODULE}}{_AnsiColor.RESET}"

        # Level name (colored, fixed width)
        level_color = _LEVEL_STYLES.get(record.levelno, "")
        level_name = record.levelname
        level_str = f"{level_color}{level_name:<{self._WIDTH_LEVEL}}{_AnsiColor.RESET}"

        # Message
        msg = record.getMessage()

        # Exception info (if any)
        exc_text = ""
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            exc_text = f"\n{_AnsiColor.RED}{record.exc_text}{_AnsiColor.RESET}"

        return f"{ts_str}  [{mod_str}]  {level_str}  {msg}{exc_text}"


class _PlainFormatter(logging.Formatter):
    """Plain formatter for file output (no ANSI escape codes)."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.") + \
             f"{int(record.msecs):03d}"
        msg = record.getMessage()
        exc_text = ""
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            exc_text = f"\n{record.exc_text}"
        return f"{ts}  [{record.name:<14}]  {record.levelname:<8}  {msg}{exc_text}"


# ── Logger Registry ───────────────────────────────────────────────────────────

_loggers: dict[str, logging.Logger] = {}
_root_configured: bool = False
_log_file_handler: Optional[logging.FileHandler] = None


def configure_root(
    level: int = logging.DEBUG,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure the root 'droid' logger once.
    Call this from main.py before any get_logger() calls.

    Args:
        level:    Minimum log level (e.g., logging.DEBUG).
        log_file: Optional path to write plain-text log file.
    """
    global _root_configured, _log_file_handler

    root = logging.getLogger("droid")
    root.setLevel(level)
    root.propagate = False

    if not root.handlers:
        # Console handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(_DroidFormatter())
        sh.setLevel(level)
        root.addHandler(sh)

    # Optional file handler
    if log_file and not _log_file_handler:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(_PlainFormatter())
        fh.setLevel(logging.DEBUG)
        root.addHandler(fh)
        _log_file_handler = fh

    _root_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'droid' namespace.

    Args:
        name: Short module identifier, e.g. "roomba_io", "vision", "brain".

    Returns:
        A Logger instance named 'droid.<name>'.
    """
    full_name = f"droid.{name}"
    if full_name not in _loggers:
        if not _root_configured:
            # Auto-configure with defaults if main.py forgot to call configure_root()
            configure_root()
        logger = logging.getLogger(full_name)
        # Child loggers inherit root level; no extra handlers needed
        _loggers[full_name] = logger
    return _loggers[full_name]


# ── Convenience Banner ────────────────────────────────────────────────────────

def print_banner(name: str, version: str = "1.0.0") -> None:
    """Print a startup ASCII banner to stdout."""
    c = _AnsiColor
    width = 60
    line = "─" * width
    print(f"\n{c.CYAN}{c.BOLD}{'':>2}╔{line}╗")
    print(f"{'':>2}║{'  ⬡  DROID FIRMWARE  ⬡':^{width}}║")
    print(f"{'':>2}║{f'  {name}  |  v{version}':^{width}}║")
    print(f"{'':>2}╚{line}╝{c.RESET}\n")
