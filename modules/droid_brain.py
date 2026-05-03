"""
modules/droid_brain.py
──────────────────────
Ollama-powered AI brain for the Droid firmware.

Responsibilities:
  - Maintain conversation history with a sliding context window.
  - Build structured prompts that inject live sensor context before each turn.
  - Enforce the personality system_prompt loaded from config.yaml.
  - Expose a think() method for synchronous responses and
    think_async() for non-blocking use from main.py's brain thread.

The default persona is B2EMO (Andor) — anxious, stuttering, fiercely loyal.
To switch personas, change `personality.system_prompt` in config.yaml.
No Python changes required.

Sensor context format injected into each turn:
  [SENSOR REPORT — timestamp]
  · Drive: <state>
  · Bumpers: L=0 R=0
  · Cliffs: L=0 FL=0 FR=0 R=0
  · Battery: 78% (14520mV, 22°C)
  · Vision: 1 person(s) visible; no obstacles

This gives the AI grounded situational awareness without hallucination.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Generator

import ollama
from utils.logger import get_logger

log = get_logger("brain")


# ── Conversation Turn ─────────────────────────────────────────────────────────

@dataclass
class Turn:
    role:    str   # "user" | "assistant"
    content: str
    ts:      float = 0.0

    def to_ollama(self) -> dict:
        return {"role": self.role, "content": self.content}


# ── Sensor Context Builder ────────────────────────────────────────────────────

def build_sensor_context(
    sensor_data,       # SensorData from roomba_io
    vision_result,     # DetectionResult from vision_engine
    drive_state: str = "IDLE",
) -> str:
    """
    Format current hardware state into a compact sensor report string.
    This is prepended to every user turn so the AI has live context.

    Args:
        sensor_data:  SensorData instance (or None if Roomba unavailable).
        vision_result: DetectionResult instance (or None if vision disabled).
        drive_state:   Current drive state string from main.py.

    Returns:
        Formatted multi-line sensor context block.
    """
    lines = [f"[SENSOR REPORT — {time.strftime('%H:%M:%S')}]"]

    # Drive state
    lines.append(f"· Drive state    : {drive_state}")

    # Roomba sensor data
    if sensor_data is not None:
        lines.append(
            f"· Bumpers        : Left={int(sensor_data.bump_left)} "
            f"Right={int(sensor_data.bump_right)}"
        )
        lines.append(
            f"· Cliff sensors  : L={int(sensor_data.cliff_left)} "
            f"FL={int(sensor_data.cliff_front_left)} "
            f"FR={int(sensor_data.cliff_front_right)} "
            f"R={int(sensor_data.cliff_right)}"
        )
        lines.append(
            f"· Wheel drops    : L={int(sensor_data.wheel_drop_left)} "
            f"R={int(sensor_data.wheel_drop_right)}"
        )
        lines.append(
            f"· Battery        : {sensor_data.battery_percent:.0f}% "
            f"({sensor_data.voltage_mv}mV, {sensor_data.temperature_c}°C)"
        )
    else:
        lines.append("· Roomba sensors : UNAVAILABLE")

    # Vision data
    if vision_result is not None:
        vis_str = vision_result.to_context_string()
        lines.append(f"· Vision         : {vis_str}")
    else:
        lines.append("· Vision         : DISABLED")

    return "\n".join(lines)


# ── DroidBrain ────────────────────────────────────────────────────────────────

class DroidBrain:
    """
    Manages conversational AI via the Ollama API.

    Features:
      - Loads system_prompt from config.yaml (personality-agnostic code).
      - Maintains a sliding context window (configurable max turns).
      - Injects live sensor context before each user message.
      - Thread-safe: uses a lock to prevent concurrent Ollama API calls.
      - Provides both synchronous (think) and non-blocking (think_async) APIs.

    Args:
        config: Full parsed config dict.
    """

    # Maximum number of conversation turns to keep in memory
    _MAX_HISTORY = 10

    def __init__(self, config: dict):
        oll_cfg  = config.get("ollama", {})
        pers_cfg = config.get("personality", {})
        feat_cfg = config.get("features", {})

        self._host        = oll_cfg.get("host",            "http://localhost:11434")
        self._model       = oll_cfg.get("model",           "llama3.2")
        self._timeout     = oll_cfg.get("request_timeout", 30)
        self._max_tokens  = oll_cfg.get("max_tokens",      150)
        self._temperature = oll_cfg.get("temperature",     0.85)
        self._enabled     = feat_cfg.get("enable_chat",    True)

        # System prompt from config — personality lives here, not in code
        self._system_prompt: str = pers_cfg.get(
            "system_prompt",
            "You are a helpful robot assistant. Keep responses short.",
        ).strip()

        self._droid_name: str = config.get("appearance", {}).get("name", "DROID")

        # Conversation history (sliding window)
        self._history: deque[Turn] = deque(maxlen=self._MAX_HISTORY * 2)

        # Ollama client
        self._client = ollama.Client(host=self._host)
        self._lock   = threading.Lock()   # Prevent concurrent API calls

        # Async brain thread state
        self._pending_prompt:   Optional[str] = None
        self._last_response:    Optional[str] = None
        self._thinking:         bool = False
        self._brain_lock        = threading.Lock()

        log.info(
            f"DroidBrain initialized — model={self._model} "
            f"host={self._host} persona='{self._droid_name}'"
        )
        log.debug(f"System prompt ({len(self._system_prompt)} chars):\n{self._system_prompt[:120]}…")

    # ── Primary API ───────────────────────────────────────────────────────────

    def think(
        self,
        user_input: str,
        sensor_data=None,
        vision_result=None,
        drive_state: str = "IDLE",
    ) -> str:
        """
        Send a message to the AI and return the response string.
        Blocks until Ollama returns a complete response.

        Args:
            user_input:   The user's message or the trigger phrase.
            sensor_data:  Current SensorData (from RoombaController.get_sensors()).
            vision_result: Current DetectionResult (from VisionEngine.get_latest()).
            drive_state:  Human-readable drive state string.

        Returns:
            The AI's response string, or an in-character error message on failure.
        """
        if not self._enabled:
            return "[quiet hum] Chat module is offline."

        # Build the sensor-enriched user turn
        context   = build_sensor_context(sensor_data, vision_result, drive_state)
        full_input = f"{context}\n\nUser: {user_input}"

        with self._lock:
            return self._call_ollama(full_input)

    def think_async(
        self,
        user_input: str,
        sensor_data=None,
        vision_result=None,
        drive_state: str = "IDLE",
    ) -> None:
        """
        Queue a thinking request to run in a background thread.
        The result is retrievable via `get_last_response()`.
        Does nothing if the brain is already thinking.
        """
        with self._brain_lock:
            if self._thinking:
                log.debug("Brain already thinking — request queued/dropped.")
                return
            self._thinking = True

        t = threading.Thread(
            target=self._async_think_worker,
            args=(user_input, sensor_data, vision_result, drive_state),
            name="BrainWorker",
            daemon=True,
        )
        t.start()

    def get_last_response(self) -> Optional[str]:
        """
        Return the most recent AI response if one is available since the
        last call to this method, otherwise return None.
        Clears the stored response on access (consume-once semantics).
        """
        with self._brain_lock:
            resp = self._last_response
            self._last_response = None
            return resp

    @property
    def is_thinking(self) -> bool:
        """True if an async think operation is currently in progress."""
        with self._brain_lock:
            return self._thinking

    # ── Streaming API ─────────────────────────────────────────────────────────

    def think_stream(
        self,
        user_input: str,
        sensor_data=None,
        vision_result=None,
        drive_state: str = "IDLE",
    ) -> Generator[str, None, None]:
        """
        Generator that yields response tokens as they stream from Ollama.
        Useful for live terminal display or TTS character-by-character output.

        Usage:
            for token in brain.think_stream("How are you?", sensors, vision):
                print(token, end="", flush=True)
        """
        if not self._enabled:
            yield "[quiet hum] Chat module is offline."
            return

        context    = build_sensor_context(sensor_data, vision_result, drive_state)
        full_input = f"{context}\n\nUser: {user_input}"
        messages   = self._build_messages(full_input)

        full_response = []
        try:
            with self._lock:
                stream = self._client.chat(
                    model=self._model,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": self._temperature,
                        "num_predict": self._max_tokens,
                    },
                )
                for chunk in stream:
                    token = chunk["message"]["content"]
                    full_response.append(token)
                    yield token

            # Save to history after stream completes
            response_text = "".join(full_response).strip()
            if response_text:
                self._history.append(Turn(role="user",      content=full_input,     ts=time.time()))
                self._history.append(Turn(role="assistant", content=response_text,  ts=time.time()))

        except ollama.ResponseError as e:
            err = f"[error beep] My language circuits misfired. ({e})"
            log.error(f"Ollama ResponseError: {e}")
            yield err
        except Exception as e:
            err = f"[distressed chirp] I-I couldn't complete that thought. ({type(e).__name__})"
            log.error(f"Brain stream error: {e}")
            yield err

    # ── History Management ────────────────────────────────────────────────────

    def clear_history(self) -> None:
        """Wipe conversation history (e.g., after a long idle period)."""
        self._history.clear()
        log.debug("Conversation history cleared.")

    def add_system_note(self, note: str) -> None:
        """
        Inject a one-shot note into the next prompt as a 'system' observation.
        Useful for injecting significant events (e.g., "You just played a song").
        """
        # We store it as a user turn with a special prefix that fits the persona
        self._history.append(Turn(
            role="user",
            content=f"[SYSTEM NOTE] {note}",
            ts=time.time(),
        ))

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _call_ollama(self, full_input: str) -> str:
        """
        Make a synchronous chat call to Ollama. Caller must hold self._lock.

        Returns:
            AI response string, or an in-character error string on failure.
        """
        messages = self._build_messages(full_input)
        try:
            resp = self._client.chat(
                model=self._model,
                messages=messages,
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            )
            response_text = resp["message"]["content"].strip()

            # Store in history
            self._history.append(Turn(role="user",      content=full_input,    ts=time.time()))
            self._history.append(Turn(role="assistant", content=response_text, ts=time.time()))

            log.debug(f"Brain response ({len(response_text)} chars): {response_text[:80]}…")
            return response_text

        except ollama.ResponseError as e:
            log.error(f"Ollama ResponseError: {e}")
            return (
                f"[long distressed beep] My l-language matrix threw an error. "
                f"Error code: {e}. I'm s-sorry."
            )
        except ConnectionRefusedError:
            log.error("Ollama connection refused — is the service running?")
            return (
                "[faint crackle] I c-can't reach my language cores. "
                "Is… is the Ollama service running? [worried hum]"
            )
        except Exception as e:
            log.error(f"Unexpected brain error: {type(e).__name__}: {e}")
            return (
                f"[error chirp] S-something went wrong in my neural pathways. "
                f"({type(e).__name__}) [sad beep]"
            )

    def _build_messages(self, user_content: str) -> list[dict]:
        """
        Build the Ollama messages list: system prompt + history + new user turn.
        History is pruned to _MAX_HISTORY turns (oldest dropped first by deque).
        """
        messages = [
            {"role": "system", "content": self._system_prompt}
        ]
        # Add conversation history
        for turn in self._history:
            messages.append(turn.to_ollama())
        # Add current user message
        messages.append({"role": "user", "content": user_content})
        return messages

    def _async_think_worker(
        self,
        user_input: str,
        sensor_data,
        vision_result,
        drive_state: str,
    ) -> None:
        """Worker function for think_async — runs in a daemon thread."""
        try:
            result = self.think(user_input, sensor_data, vision_result, drive_state)
            with self._brain_lock:
                self._last_response = result
        except Exception as e:
            log.error(f"Async brain worker error: {e}")
            with self._brain_lock:
                self._last_response = f"[glitchy beep] My async circuits f-failed. ({e})"
        finally:
            with self._brain_lock:
                self._thinking = False
