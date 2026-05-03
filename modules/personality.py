"""
modules/personality.py
──────────────────────
Personality engine: maps hardware sensor events to in-character droid speech,
manages reaction cooldowns, and provides the "B2EMO Factor."

This module does NOT call Ollama. It handles instantaneous, rule-based reactions
to sensor changes — e.g., a bump triggers a fright phrase immediately, without
waiting for the AI to respond. This keeps the droid feeling alive even when the
brain is thinking or the LLM is slow.

Architecture:
  - PersonalityEngine.update(sensor_data, vision_result) → Optional[Reaction]
    Call this every loop tick. Returns a Reaction if a new event occurred,
    or None if nothing noteworthy changed.
  - PersonalityEngine.get_startup_line() → str
  - PersonalityEngine.get_shutdown_line() → str
  - Cooldown system: each event type has a minimum re-trigger interval to avoid
    flooding the speaker/terminal with the same phrase repeatedly.
"""

import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from utils.logger import get_logger

log = get_logger("personality")


# ── Event Types ───────────────────────────────────────────────────────────────

class EventType(Enum):
    BUMP_FRONT       = auto()
    BUMP_LEFT        = auto()
    BUMP_RIGHT       = auto()
    CLIFF_DETECTED   = auto()
    WHEEL_DROP       = auto()
    LOW_BATTERY      = auto()
    CRITICAL_BATTERY = auto()
    PERSON_DETECTED  = auto()
    PERSON_LOST      = auto()
    OBSTACLE_DETECTED= auto()
    OBSTACLE_CLEARED = auto()
    IDLE             = auto()
    STARTUP          = auto()
    SHUTDOWN         = auto()


# Cooldown table: minimum seconds between reactions for each event type.
# Frequent physical events (bumps, cliffs) have shorter cooldowns.
# Slow-changing states (battery, person) have longer ones.
_COOLDOWNS: dict[EventType, float] = {
    EventType.BUMP_FRONT:        2.0,
    EventType.BUMP_LEFT:         2.0,
    EventType.BUMP_RIGHT:        2.0,
    EventType.CLIFF_DETECTED:    3.0,
    EventType.WHEEL_DROP:        3.0,
    EventType.LOW_BATTERY:       60.0,
    EventType.CRITICAL_BATTERY:  30.0,
    EventType.PERSON_DETECTED:   10.0,
    EventType.PERSON_LOST:       15.0,
    EventType.OBSTACLE_DETECTED: 5.0,
    EventType.OBSTACLE_CLEARED:  8.0,
    EventType.IDLE:              30.0,
    EventType.STARTUP:           9999,
    EventType.SHUTDOWN:          9999,
}

# Battery thresholds (percent)
_BATTERY_LOW      = 20.0
_BATTERY_CRITICAL = 10.0


# ── Reaction Dataclass ────────────────────────────────────────────────────────

@dataclass
class Reaction:
    event:   EventType
    text:    str
    song:    Optional[str]   = None    # Optional song name to play (from SONGS dict)
    urgent:  bool            = False   # If True, interrupt current activity
    ts:      float           = 0.0

    def __str__(self) -> str:
        return f"[{self.event.name}] {self.text}"


# ── PersonalityEngine ─────────────────────────────────────────────────────────

class PersonalityEngine:
    """
    Stateful personality engine.

    Tracks previous sensor states, detects transitions (edge detection),
    applies cooldowns, and returns Reaction objects for the main loop to
    speak/display/act upon.

    Args:
        config: Full parsed config dict. Reads personality.reactions block.
    """

    def __init__(self, config: dict):
        pers_cfg     = config.get("personality", {})
        self._reactions: dict[str, list[str]] = pers_cfg.get("reactions", {})
        self._name:   str = config.get("appearance", {}).get("name", "DROID")
        self._p_type: str = config.get("appearance", {}).get("personality_type", "generic")

        # Cooldown tracker: event → last trigger time
        self._last_trigger: dict[EventType, float] = {}

        # Previous state (for edge detection)
        self._prev_bump_front:    bool  = False
        self._prev_bump_left:     bool  = False
        self._prev_bump_right:    bool  = False
        self._prev_cliff:         bool  = False
        self._prev_wheel_drop:    bool  = False
        self._prev_person:        bool  = False
        self._prev_obstacle:      bool  = False
        self._prev_batt_low:      bool  = False
        self._prev_batt_critical: bool  = False

        # Idle ticker
        self._last_activity_ts: float = time.monotonic()
        self._idle_cooldown:    float = _COOLDOWNS[EventType.IDLE]

        log.info(f"PersonalityEngine initialized — persona='{self._p_type}' name='{self._name}'")

    # ── Main Update ───────────────────────────────────────────────────────────

    def update(self, sensor_data, vision_result) -> Optional[Reaction]:
        """
        Evaluate current sensor state against previous state.
        Returns the highest-priority Reaction if an event fired, else None.

        Priority order (high→low):
          CLIFF > WHEEL_DROP > BUMP > CRITICAL_BATTERY > LOW_BATTERY >
          PERSON_DETECTED > OBSTACLE_DETECTED > IDLE

        Args:
            sensor_data:   SensorData from RoombaController (may be None).
            vision_result: DetectionResult from VisionEngine (may be None).

        Returns:
            A Reaction dataclass, or None.
        """
        now = time.monotonic()
        reaction: Optional[Reaction] = None

        # ── Roomba sensor events ──────────────────────────────────────────────
        if sensor_data is not None:

            # Cliff (highest priority physical event)
            cliff_now = sensor_data.any_cliff
            if cliff_now and not self._prev_cliff:
                reaction = self._make_reaction(EventType.CLIFF_DETECTED, song="alert", urgent=True)
            self._prev_cliff = cliff_now

            # Wheel drop
            if reaction is None:
                wd_now = sensor_data.any_wheel_drop
                if wd_now and not self._prev_wheel_drop:
                    reaction = self._make_reaction(EventType.WHEEL_DROP, song="uh_oh", urgent=True)
                self._prev_wheel_drop = wd_now

            # Bumps (split by side)
            if reaction is None:
                bump_front = sensor_data.bump_left and sensor_data.bump_right
                bump_left  = sensor_data.bump_left  and not sensor_data.bump_right
                bump_right = sensor_data.bump_right and not sensor_data.bump_left

                if bump_front and not self._prev_bump_front:
                    reaction = self._make_reaction(EventType.BUMP_FRONT, song="uh_oh")
                    self._prev_bump_left  = False
                    self._prev_bump_right = False
                elif bump_left and not self._prev_bump_left:
                    reaction = self._make_reaction(EventType.BUMP_LEFT)
                elif bump_right and not self._prev_bump_right:
                    reaction = self._make_reaction(EventType.BUMP_RIGHT)

                self._prev_bump_front = bump_front
                self._prev_bump_left  = sensor_data.bump_left
                self._prev_bump_right = sensor_data.bump_right

            # Battery: critical threshold
            if reaction is None:
                batt_crit = sensor_data.battery_percent <= _BATTERY_CRITICAL
                if batt_crit and not self._prev_batt_critical:
                    reaction = self._make_reaction(EventType.CRITICAL_BATTERY, song="sad", urgent=True)
                self._prev_batt_critical = batt_crit

            # Battery: low threshold
            if reaction is None and not self._prev_batt_critical:
                batt_low = sensor_data.battery_percent <= _BATTERY_LOW
                if batt_low and not self._prev_batt_low:
                    reaction = self._make_reaction(EventType.LOW_BATTERY, song="sad")
                self._prev_batt_low = batt_low

        # ── Vision events ─────────────────────────────────────────────────────
        if vision_result is not None and reaction is None:

            # Person detected (rising edge)
            person_now = vision_result.person_detected
            if person_now and not self._prev_person:
                reaction = self._make_reaction(EventType.PERSON_DETECTED, song="happy")
            elif not person_now and self._prev_person:
                reaction = self._make_reaction(EventType.PERSON_LOST)
            self._prev_person = person_now

            # Obstacle detected (rising edge — only when no person)
            if reaction is None:
                obs_now = vision_result.obstacle_detected and not person_now
                if obs_now and not self._prev_obstacle:
                    reaction = self._make_reaction(EventType.OBSTACLE_DETECTED)
                self._prev_obstacle = obs_now

        # ── Idle chatter ──────────────────────────────────────────────────────
        if reaction is None:
            if now - self._last_activity_ts >= self._idle_cooldown:
                reaction = self._make_reaction(EventType.IDLE)

        # Update activity timer if something happened
        if reaction is not None:
            self._last_activity_ts = now

        return reaction

    # ── Lifecycle Phrases ─────────────────────────────────────────────────────

    def get_startup_line(self) -> Reaction:
        """Return a startup reaction (call once at boot)."""
        return self._make_reaction(EventType.STARTUP, song="startup", urgent=True)

    def get_shutdown_line(self) -> Reaction:
        """Return a shutdown reaction (call before exiting)."""
        return self._make_reaction(EventType.SHUTDOWN, song="sad", urgent=True)

    # ── Reaction Factory ──────────────────────────────────────────────────────

    def _make_reaction(
        self,
        event: EventType,
        song:   Optional[str] = None,
        urgent: bool          = False,
    ) -> Optional[Reaction]:
        """
        Create a Reaction if the cooldown for this event has expired.

        Returns None if the event is still in cooldown.
        """
        now = time.monotonic()
        cooldown = _COOLDOWNS.get(event, 5.0)
        last_ts  = self._last_trigger.get(event, 0.0)

        if now - last_ts < cooldown:
            log.debug(f"Event {event.name} suppressed (cooldown {cooldown - (now - last_ts):.1f}s remaining).")
            return None

        # Pick phrase from config reactions
        text = self._pick_phrase(event)
        if text is None:
            log.debug(f"No phrase configured for event {event.name}.")
            return None

        self._last_trigger[event] = now
        r = Reaction(event=event, text=text, song=song, urgent=urgent, ts=now)
        log.info(f"Reaction → {r}")
        return r

    def _pick_phrase(self, event: EventType) -> Optional[str]:
        """
        Randomly select a phrase from config.yaml personality.reactions
        for the given EventType. Falls back to a hardcoded default if the
        config entry is missing.
        """
        key = event.name.lower()
        phrases = self._reactions.get(key)

        if phrases and isinstance(phrases, list) and len(phrases) > 0:
            return random.choice(phrases)

        # Hardcoded fallback phrases (always in-character, never naked error strings)
        fallbacks: dict[str, str] = {
            "bump_front":        "[startled clank] Oh! S-something's there!",
            "bump_left":         "[worried chirp] L-left side impact.",
            "bump_right":        "[sharp clank] R-right side impact.",
            "cliff_detected":    "[alarm beeping] CLIFF! I-I am NOT moving!",
            "wheel_drop":        "[deep worried hum] A-a wheel is in the air!",
            "low_battery":       "[mournful tone] P-power is low. I need to charge.",
            "critical_battery":  "[faint beep] P-power c-critical. Going dark soon.",
            "person_detected":   "[hopeful trill] I s-see a person!",
            "person_lost":       "[sad hum] They're… g-gone from visual range.",
            "obstacle_detected": "[analytical hum] Obstacle d-detected. Re-routing.",
            "obstacle_cleared":  "[relieved chirp] Path is c-clear again.",
            "idle":              "[quiet hum] S-standing by… mostly nominal.",
            "startup":           "[rising trill] B2EMO r-reporting for duty.",
            "shutdown":          "[descending tone] P-powering down. Goodbye.",
        }
        return fallbacks.get(key, f"[beep] {event.name}")

    # ── Debug / Status ────────────────────────────────────────────────────────

    def reset_cooldowns(self) -> None:
        """Force all cooldowns to expire (useful for testing/demo mode)."""
        self._last_trigger.clear()
        log.debug("All personality cooldowns reset.")

    @property
    def personality_type(self) -> str:
        return self._p_type

    @property
    def droid_name(self) -> str:
        return self._name
