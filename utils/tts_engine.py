"""
utils/tts_engine.py
───────────────────
Offline text-to-speech engine for the Droid firmware.

Uses pyttsx3 (which wraps espeak-ng on Linux) to speak reaction text
through the system audio output — separate from the Roomba's hardware speaker.

Architecture:
  TTSEngine runs pyttsx3 entirely inside a single dedicated daemon thread.
  pyttsx3 is NOT thread-safe and must be initialized and used in the same
  thread, so a Queue is used to pass text from the main loop to the worker.

  speak(text) → puts cleaned text on the queue (non-blocking, returns instantly)
  Worker thread → dequeues and calls engine.say() + engine.runAndWait()

Text cleaning:
  B2EMO reaction text contains bracketed sound cues like [alarmed beep-sequence]
  and [nervous gear-grind]. These are stripped before synthesis so only the
  spoken dialogue is passed to the TTS engine.

Linux setup (Pop!_OS / Ubuntu / Debian):
  sudo apt install espeak-ng
  pip install pyttsx3

Config (config.yaml):
  tts:
    enabled:     true
    rate:        145      # words per minute (lower = clearer for droid voice)
    volume:      1.0      # 0.0 – 1.0
    voice_index: 0        # 0 = first espeak voice; try 1+ for alternatives
"""

import queue
import re
import threading
from typing import Optional

from utils.logger import get_logger

log = get_logger("tts")

# Matches anything inside square brackets including the brackets: [like this]
_BRACKET_RE = re.compile(r"\[.*?\]")

# Collapse runs of whitespace left after bracket removal
_WHITESPACE_RE = re.compile(r"\s{2,}")


def clean_for_speech(text: str) -> str:
    """
    Strip bracketed stage-direction cues and tidy up whitespace.

    Examples:
        "[alarmed beep] Oh! S-something's there!"
        → "Oh! S-something's there!"

        "F-front bumper contact! [rattled chassis shudder] My apologies."
        → "F-front bumper contact! My apologies."
    """
    cleaned = _BRACKET_RE.sub("", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


class TTSEngine:
    """
    Non-blocking text-to-speech engine.

    All pyttsx3 calls happen inside a single dedicated daemon thread so
    the drive loop is never blocked by speech synthesis.

    Args:
        config: Full parsed config dict. Reads the 'tts' sub-dict.
    """

    # Sentinel value to signal the worker thread to exit cleanly
    _STOP_SENTINEL = None

    def __init__(self, config: dict):
        tts_cfg = config.get("tts", {})
        self._enabled     = tts_cfg.get("enabled",     True)
        self._rate        = tts_cfg.get("rate",        145)
        self._volume      = tts_cfg.get("volume",      1.0)
        self._voice_index = tts_cfg.get("voice_index", 0)

        self._queue: queue.Queue = queue.Queue()
        self._ready_event = threading.Event()   # set when engine is initialized
        self._init_ok: bool = False

        if self._enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="TTSWorker",
                daemon=True,
            )
            self._thread.start()
            # Wait up to 5 s for pyttsx3 to initialize before returning
            if not self._ready_event.wait(timeout=5.0):
                log.warning("TTS engine took too long to initialize — speech may be delayed.")
        else:
            log.info("TTS disabled in config.")

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Queue text for speech synthesis (non-blocking).
        Strips bracketed sound cues before queuing.
        Does nothing if TTS is disabled or the engine failed to initialize.

        Args:
            text: Raw reaction/response text, may contain [bracket cues].
        """
        if not self._enabled or not self._init_ok:
            return

        cleaned = clean_for_speech(text)
        if not cleaned:
            return

        log.debug(f"TTS queued: {cleaned[:60]}{'…' if len(cleaned) > 60 else ''}")
        self._queue.put(cleaned)

    def stop(self) -> None:
        """Signal the worker thread to exit and wait for it to finish."""
        if self._enabled:
            self._queue.put(self._STOP_SENTINEL)

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._init_ok

    # ── Worker Thread ─────────────────────────────────────────────────────────

    def _worker(self) -> None:
        """
        Dedicated TTS thread. Initializes pyttsx3, then processes the queue.
        pyttsx3 MUST be initialized and used from the same thread.
        """
        try:
            import pyttsx3
        except ImportError:
            log.error(
                "pyttsx3 not installed. Run: pip install pyttsx3  "
                "and: sudo apt install espeak-ng"
            )
            self._ready_event.set()   # unblock __init__ even on failure
            return

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)

            voices = engine.getProperty("voices")
            if voices:
                idx = min(self._voice_index, len(voices) - 1)
                engine.setProperty("voice", voices[idx].id)
                log.debug(f"TTS voice: {voices[idx].name} (index {idx})")
            else:
                log.warning("No TTS voices found — espeak-ng may not be installed.")

            log.info(
                f"TTS engine ready — rate={self._rate} wpm, "
                f"volume={self._volume:.0%}, voice_index={self._voice_index}"
            )
            self._init_ok = True

        except Exception as e:
            log.error(f"TTS engine init failed: {type(e).__name__}: {e}")
            self._ready_event.set()
            return

        self._ready_event.set()   # signal __init__ that engine is ready

        # ── Speech loop ───────────────────────────────────────────────────────
        while True:
            try:
                text = self._queue.get()

                if text is self._STOP_SENTINEL:
                    log.debug("TTS worker received stop signal.")
                    break

                engine.say(text)
                engine.runAndWait()

            except Exception as e:
                log.error(f"TTS speak error: {type(e).__name__}: {e}")
                # Don't exit the loop on a single utterance failure

        log.debug("TTS worker thread exited.")
