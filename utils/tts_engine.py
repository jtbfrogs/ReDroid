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
  B2EMO reaction text contains bracketed sound cues like [alarmed beep-sequence].
  These are stripped before synthesis so only the dialogue is spoken.

Voice selection (espeak-ng on Linux):
  Set voice_id in config.yaml to any espeak voice name.
  Variants are appended with +:
    "english"         standard British English
    "en-us"           American English
    "english+m1"–m7   male voice variants (try m3 for robotic)
    "english+f1"–f4   female voice variants
  At startup the engine logs all available voices so you can browse options.
  Run in terminal:  espeak-ng --voices | grep en

Linux setup (Pop!_OS / Ubuntu / Debian):
  sudo apt install espeak-ng
  pip install pyttsx3
"""

import queue
import re
import threading
from typing import Optional

from utils.logger import get_logger

log = get_logger("tts")

# Matches anything inside square brackets: [like this]
_BRACKET_RE   = re.compile(r"\[.*?\]")
_WHITESPACE_RE = re.compile(r"\s{2,}")


def clean_for_speech(text: str) -> str:
    """
    Strip bracketed stage-direction cues and tidy up whitespace.

    Examples:
        "[alarmed beep] Oh! S-something's there!"  → "Oh! S-something's there!"
        "Front contact! [rattled shudder] My apologies." → "Front contact! My apologies."
    """
    cleaned = _BRACKET_RE.sub("", text)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


class TTSEngine:
    """
    Non-blocking text-to-speech engine.

    All pyttsx3 calls happen inside a single dedicated daemon thread so
    the drive loop is never blocked by speech synthesis.

    Config keys read from config.yaml 'tts' block:
        enabled     (bool)   — master on/off switch
        rate        (int)    — words per minute (default 145; lower = clearer)
        volume      (float)  — 0.0 – 1.0
        pitch       (int)    — 0–99; only works on some espeak builds (default 50)
        voice_id    (str)    — espeak voice name, e.g. "english+m3" or "en-us"
                               Leave blank to use the first voice in the list.
    """

    _STOP_SENTINEL = None

    def __init__(self, config: dict):
        tts_cfg = config.get("tts", {})
        self._enabled  = tts_cfg.get("enabled",  True)
        self._rate     = int(tts_cfg.get("rate",   145))
        self._volume   = float(tts_cfg.get("volume", 1.0))
        self._pitch    = int(tts_cfg.get("pitch",  50))
        self._voice_id = tts_cfg.get("voice_id", "english+m3").strip()

        self._queue: queue.Queue        = queue.Queue()
        self._ready_event               = threading.Event()
        self._init_ok: bool             = False

        if self._enabled:
            self._thread = threading.Thread(
                target=self._worker,
                name="TTSWorker",
                daemon=True,
            )
            self._thread.start()
            # Wait up to 5 s for pyttsx3 to initialize
            if not self._ready_event.wait(timeout=5.0):
                log.warning("TTS engine took too long to initialize — speech may be delayed.")
        else:
            log.info("TTS disabled in config.")

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Queue text for speech synthesis (non-blocking).
        Strips bracketed sound cues before queuing.
        """
        if not self._enabled or not self._init_ok:
            return
        cleaned = clean_for_speech(text)
        if not cleaned:
            return
        log.debug(f"TTS queued: {cleaned[:70]}{'…' if len(cleaned) > 70 else ''}")
        self._queue.put(cleaned)

    def stop(self) -> None:
        """Signal the worker thread to exit cleanly."""
        if self._enabled:
            self._queue.put(self._STOP_SENTINEL)

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._init_ok

    # ── Worker Thread ─────────────────────────────────────────────────────────

    def _worker(self) -> None:
        """
        Dedicated TTS thread.
        Initializes pyttsx3, logs available voices, applies config, then
        processes the speech queue until a stop sentinel is received.
        """
        try:
            import pyttsx3
        except ImportError:
            log.error(
                "pyttsx3 not installed. Run:  pip install pyttsx3  "
                "and:  sudo apt install espeak-ng"
            )
            self._ready_event.set()
            return

        try:
            engine = pyttsx3.init()
        except Exception as e:
            log.error(f"pyttsx3 init failed: {type(e).__name__}: {e}")
            self._ready_event.set()
            return

        # ── Log available voices ──────────────────────────────────────────────
        voices = engine.getProperty("voices") or []
        log.info(f"TTS — {len(voices)} voice(s) available:")
        for i, v in enumerate(voices):
            log.info(f"    [{i:>2}]  {v.name:<30}  id: {v.id}")
        if not voices:
            log.warning(
                "No TTS voices found. Install espeak-ng:  sudo apt install espeak-ng"
            )

        # ── Apply rate and volume ─────────────────────────────────────────────
        engine.setProperty("rate",   self._rate)
        engine.setProperty("volume", self._volume)

        # ── Apply pitch (supported on some espeak builds) ─────────────────────
        try:
            engine.setProperty("pitch", self._pitch)
        except Exception:
            pass   # Not all backends support pitch — silently skip

        # ── Select voice by voice_id string ───────────────────────────────────
        selected_voice = None
        if self._voice_id and voices:
            # Try exact match on id first, then partial match on name or id
            vid_lower = self._voice_id.lower()
            selected_voice = (
                next((v for v in voices if v.id.lower() == vid_lower), None) or
                next((v for v in voices if vid_lower in v.id.lower()), None) or
                next((v for v in voices if vid_lower in v.name.lower()), None)
            )

        if selected_voice:
            engine.setProperty("voice", selected_voice.id)
            log.info(
                f"TTS voice set → '{selected_voice.name}'  (id: {selected_voice.id})"
            )
        elif self._voice_id:
            # voice_id not found in the pyttsx3 list — pass it directly to
            # espeak (works for variants like "english+m3" that aren't enumerated)
            try:
                engine.setProperty("voice", self._voice_id)
                log.info(f"TTS voice set directly → '{self._voice_id}'")
            except Exception as e:
                log.warning(
                    f"Could not set voice '{self._voice_id}': {e}  "
                    f"Using default. Check config.yaml tts.voice_id."
                )
        elif voices:
            engine.setProperty("voice", voices[0].id)
            log.info(f"TTS using default voice: '{voices[0].name}'")

        log.info(
            f"TTS ready — rate={self._rate} wpm | "
            f"volume={self._volume:.0%} | "
            f"pitch={self._pitch} | "
            f"voice='{self._voice_id}'"
        )
        self._init_ok = True
        self._ready_event.set()

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

        log.debug("TTS worker thread exited.")
