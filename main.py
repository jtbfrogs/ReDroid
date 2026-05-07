"""
main.py — Droid Firmware Orchestrator
──────────────────────────────────────
Initializes all subsystems, runs the pre-flight diagnostic sequence,
then enters the main control loop with three concurrent concerns:

  1. DRIVE LOOP  (20 Hz, main thread) ─── highest priority
     Reads Roomba sensors, runs reactive navigation state machine,
     sends drive commands. Never blocked by AI or vision.

  2. VISION ENGINE (15 Hz, daemon thread) ─── started by VisionEngine
     Runs HOG person detection + obstacle detection continuously.
     Results available via VisionEngine.get_latest().

  3. BRAIN WORKER (on-demand, daemon thread) ─── DroidBrain.think_async()
     Fires off an Ollama API call when a personality reaction occurs
     or the user types a message. Returns result via get_last_response().

Threading model:
  ┌─────────────────────────────────────────────────────────┐
  │  main thread  │  VisionCapture  │  RoombaSensor  │  BrainWorker  │
  │   drive loop  │  (daemon)       │  (daemon)      │  (on-demand)  │
  │   pers. logic │  CV2 + HOG      │  serial poll   │  Ollama API   │
  └─────────────────────────────────────────────────────────┘
  Shared data accessed through thread-safe getters (no raw attribute access
  across threads). Drive loop never awaits the brain.

Drive State Machine:
  IDLE ──→ FORWARD ──→ BACKING_UP ──→ TURNING ──→ FORWARD
                ↑____________cliff/bump____________↑
"""

import os
import sys
import time
import threading
import signal
import yaml
from enum import Enum, auto
from typing import Optional

# ── Ensure project root is in path ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from utils.logger  import configure_root, get_logger, print_banner
from utils.diag_manager import DiagManager
from utils.tts_engine import TTSEngine
from modules.roomba_io   import RoombaController, SensorData
from modules.vision_engine import VisionEngine, DetectionResult
from modules.droid_brain import DroidBrain
from modules.personality import PersonalityEngine, EventType, Reaction

import logging

# ── Version ───────────────────────────────────────────────────────────────────
FIRMWARE_VERSION = "1.0.0"


# ── Drive State Machine ───────────────────────────────────────────────────────

class DriveState(Enum):
    IDLE       = auto()
    FORWARD    = auto()
    BACKING_UP = auto()
    TURNING    = auto()
    STOPPED    = auto()   # Intentionally halted (person present, etc.)


# ── Drive Tuning Constants ────────────────────────────────────────────────────

DRIVE_SPEED_NORMAL  = 150   # mm/s — nominal cruise speed
DRIVE_SPEED_SLOW    = 80    # mm/s — cautious approach (obstacle in vision)
BACKUP_SPEED        = -120  # mm/s — reverse after bump/cliff
BACKUP_DURATION     = 0.6   # seconds to back up
TURN_SPEED          = 120   # mm/s tangential speed for spin
TURN_DURATION_MIN   = 0.4   # seconds minimum turn time
TURN_DURATION_MAX   = 1.2   # seconds maximum turn time (randomized)


# ── Global Shutdown Event ─────────────────────────────────────────────────────

_shutdown_event = threading.Event()


def _signal_handler(signum, frame):
    """Handle Ctrl+C or SIGTERM gracefully."""
    log = get_logger("main")
    log.warning(f"Signal {signum} received — initiating graceful shutdown …")
    _shutdown_event.set()


# ── Config Loader ─────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Droid Orchestrator ────────────────────────────────────────────────────────

class DroidOrchestrator:
    """
    Top-level firmware orchestrator.
    Owns all subsystem instances and the main drive loop.
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._log = get_logger("main")

        # Feature flags
        feat = config.get("features", {})
        self._enable_drive  = feat.get("enable_drive",  True)
        self._enable_vision = feat.get("enable_vision", True)
        self._enable_chat   = feat.get("enable_chat",   True)

        # Drive state machine
        self._drive_state:      DriveState  = DriveState.IDLE
        self._state_entry_time: float       = time.monotonic()
        self._turn_duration:    float       = 0.8
        self._turn_direction:   int         = 1     # 1 = right, -1 = left
        self._drive_loop_dt:    float       = config["system"].get("drive_loop_interval", 0.05)

        # Subsystems (initialized in run())
        self._roomba:      Optional[RoombaController] = None
        self._vision:      Optional[VisionEngine]     = None
        self._brain:       Optional[DroidBrain]       = None
        self._personality: Optional[PersonalityEngine]= None
        self._tts:         Optional[TTSEngine]        = None

        # Input thread for CLI chat
        self._user_input_queue: list[str] = []
        self._input_lock = threading.Lock()

    # ── Startup ───────────────────────────────────────────────────────────────

    def run(self) -> int:
        """
        Full firmware lifecycle:
          1. Pre-flight diagnostics
          2. Subsystem initialization
          3. Startup announcement
          4. Main drive loop (blocking)
          5. Graceful shutdown

        Returns:
            Exit code (0 = clean shutdown, 1 = fatal error).
        """
        self._log.info("Starting Droid Firmware …")
        droid_name = self._cfg.get("appearance", {}).get("name", "DROID")

        # ── 1. Pre-flight Diagnostics ─────────────────────────────────────────
        diag = DiagManager(self._cfg)
        report = diag.run_preflight()

        # Adjust feature flags based on hardware availability
        if not report.roomba_ok:
            self._log.warning("Roomba not available — drive features DISABLED.")
            self._enable_drive = False
        if not report.camera_ok:
            self._log.warning("Camera not available — vision DISABLED.")
            self._enable_vision = False
        if not report.ollama_ok:
            self._log.warning("Ollama not available — chat DISABLED.")
            self._enable_chat = False

        # ── 2. Initialize Subsystems ──────────────────────────────────────────
        self._tts = TTSEngine(self._cfg)
        self._personality = PersonalityEngine(self._cfg)

        # Roomba
        if self._enable_drive:
            self._roomba = RoombaController(self._cfg)
            if not self._roomba.connect():
                self._log.error("Roomba connection failed at startup — drive DISABLED.")
                self._enable_drive = False
                self._roomba = None

        # Vision Engine
        if self._enable_vision:
            self._vision = VisionEngine(self._cfg, show_preview=False)
            if not self._vision.start():
                self._log.warning("Vision engine failed to start — vision DISABLED.")
                self._enable_vision = False
                self._vision = None

        # AI Brain
        if self._enable_chat:
            self._brain = DroidBrain(self._cfg)

        # ── 3. Startup Announcement ───────────────────────────────────────────
        startup_reaction = self._personality.get_startup_line()
        if startup_reaction:
            self._speak_reaction(startup_reaction)
        else:
            self._log.warning("Startup reaction suppressed by cooldown — skipping announcement.")
        time.sleep(1.5)  # Let the startup song finish

        # ── 4. Start CLI input thread ─────────────────────────────────────────
        if self._enable_chat:
            self._start_input_thread()

        # ── 5. Main Drive Loop ────────────────────────────────────────────────
        self._log.info("Entering main control loop. Press Ctrl+C to exit.")
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self._log.info("KeyboardInterrupt received.")
        finally:
            self._shutdown()

        return 0

    # ── Main Loop ─────────────────────────────────────────────────────────────

    def _main_loop(self) -> None:
        """
        Core orchestration loop running at `drive_loop_interval` Hz.

        Tick order per iteration:
          1. Read latest sensor state (non-blocking)
          2. Read latest vision state (non-blocking)
          3. Run personality event detection → optional Reaction
          4. Act on Reaction (speak, play song, query brain)
          5. Check for pending brain response → print/log
          6. Check for user chat input → send to brain
          7. Drive state machine tick
        """
        last_log_ts = time.monotonic()
        loop_count  = 0

        while not _shutdown_event.is_set():
            tick_start = time.monotonic()

            # ── 1. Gather sensor state ────────────────────────────────────────
            sensors = self._roomba.get_sensors() if self._roomba else None
            vision  = self._vision.get_latest()  if self._vision else None

            # ── 2. Personality: detect events, get reaction ───────────────────
            reaction = self._personality.update(sensors, vision)
            if reaction is not None:
                self._speak_reaction(reaction)
                # If the event is significant, trigger an AI elaboration
                if self._enable_chat and self._brain and not self._brain.is_thinking:
                    trigger = self._reaction_to_prompt(reaction)
                    if trigger:
                        drive_state_str = self._drive_state.name
                        self._brain.think_async(trigger, sensors, vision, drive_state_str)

            # ── 3. Consume pending brain response ────────────────────────────
            if self._enable_chat and self._brain:
                ai_resp = self._brain.get_last_response()
                if ai_resp:
                    self._log.info(f"[{self._cfg['appearance']['name']}] {ai_resp}")
                    print(f"\n  🤖  {ai_resp}\n")
                    if self._tts:
                        self._tts.speak(ai_resp)

            # ── 4. User chat input ────────────────────────────────────────────
            with self._input_lock:
                if self._user_input_queue:
                    user_msg = self._user_input_queue.pop(0)
                    if self._brain and not self._brain.is_thinking:
                        drive_state_str = self._drive_state.name
                        self._brain.think_async(user_msg, sensors, vision, drive_state_str)
                    else:
                        self._log.debug("Brain busy — user message deferred.")

            # ── 5. Drive state machine ────────────────────────────────────────
            if self._enable_drive and self._roomba:
                self._tick_drive_state_machine(sensors, vision)

            # ── 6. Periodic status log ────────────────────────────────────────
            loop_count += 1
            now = time.monotonic()
            if now - last_log_ts >= 30.0:
                self._log_status(sensors, vision)
                last_log_ts = now

            # ── Maintain loop rate ────────────────────────────────────────────
            elapsed = time.monotonic() - tick_start
            sleep_t = self._drive_loop_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ── Drive State Machine ───────────────────────────────────────────────────

    def _tick_drive_state_machine(
        self,
        sensors: Optional[SensorData],
        vision:  Optional[DetectionResult],
    ) -> None:
        """
        Reactive navigation state machine.
        Priority (high→low): cliff > wheel_drop > bump > person > obstacle > free_roam

        States:
          IDLE       → do nothing (manual control / initial state)
          FORWARD    → drive straight, watch for obstacles
          BACKING_UP → reverse after collision/cliff for BACKUP_DURATION seconds
          TURNING    → spin in place for randomized TURN_DURATION seconds
          STOPPED    → halt (person detected nearby, or override)
        """
        now      = time.monotonic()
        in_state = now - self._state_entry_time  # time spent in current state

        s = sensors   # shorthand
        v = vision

        # ── Safety: always override on cliff / wheel drop ──────────────────
        if s and (s.any_cliff or s.any_wheel_drop):
            if self._drive_state not in (DriveState.BACKING_UP, DriveState.TURNING):
                self._log.warning(
                    f"Safety halt: cliff={s.any_cliff} wheel_drop={s.any_wheel_drop}"
                )
                self._transition(DriveState.BACKING_UP)
                self._roomba.drive_backward(DRIVE_SPEED_SLOW)
            return

        # ── State transitions ──────────────────────────────────────────────
        if self._drive_state == DriveState.IDLE:
            # Auto-start driving after a brief settle period
            if in_state > 2.0:
                self._transition(DriveState.FORWARD)
                self._roomba.drive_forward(DRIVE_SPEED_NORMAL)

        elif self._drive_state == DriveState.FORWARD:
            # Check for person → stop and greet
            if v and v.person_detected:
                self._log.info("Person detected — stopping to greet.")
                self._transition(DriveState.STOPPED)
                self._roomba.stop_drive()
                return

            # Check for bump → back up
            if s and s.any_bump:
                self._log.info(f"Bump detected (L={s.bump_left} R={s.bump_right}) — backing up.")
                self._transition(DriveState.BACKING_UP)
                self._roomba.drive_backward(abs(BACKUP_SPEED))
                return

            # Check for vision obstacle → slow down
            if v and v.obstacle_detected and not v.person_detected:
                self._roomba.drive_forward(DRIVE_SPEED_SLOW)
            else:
                self._roomba.drive_forward(DRIVE_SPEED_NORMAL)

        elif self._drive_state == DriveState.BACKING_UP:
            if in_state >= BACKUP_DURATION:
                # Randomize turn direction and duration for natural behavior
                import random
                self._turn_direction = random.choice([-1, 1])
                self._turn_duration  = random.uniform(TURN_DURATION_MIN, TURN_DURATION_MAX)
                self._transition(DriveState.TURNING)
                if self._turn_direction == 1:
                    self._roomba.spin_right(TURN_SPEED)
                else:
                    self._roomba.spin_left(TURN_SPEED)

        elif self._drive_state == DriveState.TURNING:
            if in_state >= self._turn_duration:
                self._transition(DriveState.FORWARD)
                self._roomba.drive_forward(DRIVE_SPEED_NORMAL)

        elif self._drive_state == DriveState.STOPPED:
            # Resume driving when person leaves the frame
            if v and not v.person_detected:
                self._log.info("Person gone — resuming navigation.")
                self._transition(DriveState.FORWARD)
                self._roomba.drive_forward(DRIVE_SPEED_NORMAL)

    def _transition(self, new_state: DriveState) -> None:
        """Transition to a new drive state and record entry time."""
        if new_state != self._drive_state:
            self._log.debug(f"Drive: {self._drive_state.name} → {new_state.name}")
            self._drive_state      = new_state
            self._state_entry_time = time.monotonic()

    # ── Speech / Reaction Handler ─────────────────────────────────────────────

    def _speak_reaction(self, reaction: Reaction) -> None:
        """
        Output a personality reaction to the terminal and optionally
        trigger a song on the Roomba speaker.
        """
        name = self._cfg.get("appearance", {}).get("name", "DROID")
        print(f"\n  ⬡  [{name}]  {reaction.text}\n")
        self._log.info(f"REACTION [{reaction.event.name}]: {reaction.text}")

        # Speak the reaction text through system audio (non-blocking)
        if self._tts:
            self._tts.speak(reaction.text)

        # Play associated song if Roomba is available and song is specified
        if reaction.song and self._roomba and self._enable_drive:
            self._roomba.play_song(reaction.song)

    def _reaction_to_prompt(self, reaction: Reaction) -> Optional[str]:
        """
        Convert a personality Reaction into a short AI prompt that will
        produce an elaborated in-character response.
        Returns None for events that don't warrant AI elaboration.
        """
        _NO_ELABORATE = {EventType.IDLE, EventType.OBSTACLE_CLEARED, EventType.PERSON_LOST}
        if reaction.event in _NO_ELABORATE:
            return None

        prompts = {
            EventType.BUMP_FRONT:        "You just bumped into something directly in front of you. Express your startled reaction and describe what happened.",
            EventType.BUMP_LEFT:         "Your left bumper just hit something. Describe the impact and your feelings about it.",
            EventType.BUMP_RIGHT:        "Your right bumper just hit something. Describe the impact.",
            EventType.CLIFF_DETECTED:    "Your cliff sensors just detected a dangerous drop! Express your alarm and explain why you stopped.",
            EventType.WHEEL_DROP:        "One of your wheels has lost contact with the ground! Express your distress.",
            EventType.LOW_BATTERY:       "Your battery is running low. Express your concern and desire to find a charging dock.",
            EventType.CRITICAL_BATTERY:  "Your battery is critically low. Express your urgency and fear of going dark.",
            EventType.PERSON_DETECTED:   "You just detected a person on your visual sensors. Greet them warmly in your anxious, loyal way.",
            EventType.OBSTACLE_DETECTED: "Your navigation sensors show an obstacle ahead. Explain that you are rerouting.",
            EventType.STARTUP:           "You have just completed your boot sequence. Introduce yourself and report your initial system status.",
            EventType.SHUTDOWN:          "You are about to power down. Say a brief, emotional goodbye.",
        }
        return prompts.get(reaction.event)

    # ── CLI Input Thread ──────────────────────────────────────────────────────

    def _start_input_thread(self) -> None:
        """Launch a daemon thread for reading user CLI input without blocking the loop."""
        t = threading.Thread(
            target=self._input_reader,
            name="CLIInput",
            daemon=True,
        )
        t.start()
        self._log.debug("CLI input thread started. Type messages and press Enter to chat.")
        print(
            f"\n  Type a message and press Enter to chat with "
            f"{self._cfg['appearance']['name']}.\n"
            f"  Press Ctrl+C to exit.\n"
        )

    def _input_reader(self) -> None:
        """Read stdin lines and push them to the input queue."""
        while not _shutdown_event.is_set():
            try:
                line = input("  You: ").strip()
                if line:
                    with self._input_lock:
                        self._user_input_queue.append(line)
            except EOFError:
                break
            except Exception:
                break

    # ── Status Logger ─────────────────────────────────────────────────────────

    def _log_status(
        self,
        sensors: Optional[SensorData],
        vision:  Optional[DetectionResult],
    ) -> None:
        """Print a periodic health summary to the terminal."""
        name = self._cfg["appearance"]["name"]
        self._log.info(f"── STATUS REPORT ──────────────────────────────")
        self._log.info(f"  Droid       : {name} | Drive: {self._drive_state.name}")
        if sensors:
            self._log.info(f"  Sensors     : {sensors}")
        else:
            self._log.info(f"  Sensors     : OFFLINE")
        if vision:
            self._log.info(f"  Vision      : {vision.to_context_string()}")
        else:
            self._log.info(f"  Vision      : OFFLINE")
        if self._brain:
            self._log.info(f"  Brain       : {'THINKING' if self._brain.is_thinking else 'IDLE'}")
        self._log.info(f"──────────────────────────────────────────────")

    # ── Graceful Shutdown ─────────────────────────────────────────────────────

    def _shutdown(self) -> None:
        """Shutdown all subsystems in reverse-init order."""
        self._log.info("Initiating graceful shutdown sequence …")

        # Shutdown announcement
        if self._personality:
            shutdown_reaction = self._personality.get_shutdown_line()
            self._speak_reaction(shutdown_reaction)
            time.sleep(2.0)   # Let the farewell song finish

        # Stop Roomba (stop drive before closing serial)
        if self._roomba:
            self._roomba.stop_drive()
            time.sleep(0.1)
            self._roomba.disconnect()

        # Stop vision engine
        if self._vision:
            self._vision.stop()

        if self._tts:
            self._tts.stop()

        self._log.info("All systems down. Goodbye.")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> int:
    # Register signal handlers
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Load configuration
    try:
        config = load_config("config.yaml")
    except FileNotFoundError as e:
        print(f"[FATAL] {e}")
        return 1
    except yaml.YAMLError as e:
        print(f"[FATAL] config.yaml parse error: {e}")
        return 1

    # Configure logging
    droid_name = config.get("appearance", {}).get("name", "DROID")
    log_file   = os.path.join(
        os.path.dirname(__file__),
        "diagnostics",
        f"session_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )
    configure_root(level=logging.DEBUG, log_file=log_file)
    print_banner(droid_name, FIRMWARE_VERSION)

    # Run orchestrator
    orchestrator = DroidOrchestrator(config)
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
