"""
modules/roomba_io.py
────────────────────
Low-level Roomba 650 Open Interface (OI) controller.

Implements the iRobot Roomba 600 Series Open Interface Specification v1.0.
Key responsibilities:
  - RTS/Device Detect wake-up pulse
  - OI mode management (Passive → Safe → Full)
  - Thread-safe drive commands
  - Background sensor polling (10 Hz default)
  - Song definition and playback
  - Graceful shutdown / Stop mode

Hardware notes:
  - Roomba 650 Mini-DIN connector: Pin 3 = TXD (Roomba RX), Pin 4 = RXD (Roomba TX),
    Pin 5 = Device Detect (DD), Pin 6/7 = GND, Pin 1/2 = unregulated battery voltage.
  - USB-UART adapter: connect TXD→Pin3, RXD→Pin4, RTS→Pin5, GND→Pin6.
  - ⚠ Use 3.3V logic adapter. 5V on Pin 3 can damage the Roomba's serial receiver.
  - Device Detect (DD) wake-up: assert LOW for ≥500ms. On most CH340/CP2102 TTL adapters,
    ser.rts = True drives the RTS line LOW at the physical pin. Verify with a multimeter.

OI Opcodes used (Roomba 600 series):
  128 START      133 POWER      137 DRIVE
  131 SAFE       138 MOTORS     140 SONG
  132 FULL        141 PLAY       149 QUERY_LIST
"""

import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import serial
from utils.logger import get_logger

log = get_logger("roomba_io")


# ── OI Opcodes ────────────────────────────────────────────────────────────────

class OI(IntEnum):
    START       = 128
    BAUD        = 129
    SAFE        = 131
    FULL        = 132
    POWER       = 133
    DRIVE       = 137
    DRIVE_DIRECT= 145
    MOTORS      = 138
    LEDS        = 139
    SONG        = 140
    PLAY        = 141
    SENSORS     = 142
    QUERY_LIST  = 149
    STREAM      = 148
    STREAM_PAUSE= 150


# ── Sensor Packet IDs ─────────────────────────────────────────────────────────

class Pkt(IntEnum):
    BUMPS_DROPS     = 7     # 1 byte  unsigned
    CLIFF_LEFT      = 9     # 1 byte  bool
    CLIFF_FRONT_L   = 10    # 1 byte  bool
    CLIFF_FRONT_R   = 11    # 1 byte  bool
    CLIFF_RIGHT     = 12    # 1 byte  bool
    VOLTAGE         = 22    # 2 bytes unsigned (mV)
    TEMPERATURE     = 24    # 1 byte  signed   (°C)
    BATTERY_CHARGE  = 25    # 2 bytes unsigned (mAh)
    BATTERY_CAPACITY= 26    # 2 bytes unsigned (mAh)

# Ordered packet list for QUERY_LIST command
_QUERY_PACKETS = [
    Pkt.BUMPS_DROPS,
    Pkt.CLIFF_LEFT,
    Pkt.CLIFF_FRONT_L,
    Pkt.CLIFF_FRONT_R,
    Pkt.CLIFF_RIGHT,
    Pkt.VOLTAGE,
    Pkt.TEMPERATURE,
    Pkt.BATTERY_CHARGE,
    Pkt.BATTERY_CAPACITY,
]
# Struct format: B B B B B H b H H  → 12 bytes total
_SENSOR_FORMAT = ">BBBBBHbHH"
_SENSOR_SIZE   = struct.calcsize(_SENSOR_FORMAT)  # 12


# ── Predefined Songs ──────────────────────────────────────────────────────────
# Format: list of (note_number, duration_64ths)
# Middle C = 60, duration in 1/64-second units (64 = 1.0s, 16 = 0.25s)

SONGS: dict[str, list[tuple[int, int]]] = {
    "startup": [
        (72, 12), (74, 12), (76, 20), (79, 28),   # C5 D5 E5 G5 ascending
    ],
    "happy": [
        (67, 8), (72, 8), (76, 8), (79, 16), (76, 8), (79, 24),
    ],
    "sad": [
        (72, 16), (71, 16), (69, 16), (67, 32),   # descending minor
    ],
    "alert": [
        (79, 8), (79, 8), (79, 8), (79, 24),       # repeated G5 beeps
    ],
    "uh_oh": [
        (72, 8), (65, 32),                          # C5 → F4 drop
    ],
    "victory": [
        (60, 8), (64, 8), (67, 8), (72, 8), (76, 8), (79, 24), (76, 8), (79, 32),
    ],
}

# Roomba 650 supports songs 0–4 (5 song slots)
SONG_SLOTS = {name: idx for idx, name in enumerate(list(SONGS.keys())[:5])}


# ── Sensor Data Dataclass ─────────────────────────────────────────────────────

@dataclass
class SensorData:
    bump_right:         bool  = False
    bump_left:          bool  = False
    wheel_drop_right:   bool  = False
    wheel_drop_left:    bool  = False
    cliff_left:         bool  = False
    cliff_front_left:   bool  = False
    cliff_front_right:  bool  = False
    cliff_right:        bool  = False
    voltage_mv:         int   = 0
    temperature_c:      int   = 0
    battery_charge_mah: int   = 0
    battery_capacity_mah: int = 0
    timestamp:          float = 0.0

    @property
    def any_bump(self) -> bool:
        return self.bump_right or self.bump_left

    @property
    def any_cliff(self) -> bool:
        return (self.cliff_left or self.cliff_front_left or
                self.cliff_front_right or self.cliff_right)

    @property
    def any_wheel_drop(self) -> bool:
        return self.wheel_drop_right or self.wheel_drop_left

    @property
    def battery_percent(self) -> float:
        if self.battery_capacity_mah <= 0:
            return 0.0
        return min(100.0, (self.battery_charge_mah / self.battery_capacity_mah) * 100.0)

    def __str__(self) -> str:
        return (
            f"Bump(L={int(self.bump_left)} R={int(self.bump_right)}) "
            f"Cliff(L={int(self.cliff_left)} FL={int(self.cliff_front_left)} "
            f"FR={int(self.cliff_front_right)} R={int(self.cliff_right)}) "
            f"Batt={self.battery_percent:.0f}% ({self.voltage_mv}mV) "
            f"Temp={self.temperature_c}°C"
        )


# ── Drive State Machine ───────────────────────────────────────────────────────

class DriveState(IntEnum):
    IDLE       = 0
    FORWARD    = 1
    TURNING    = 2
    BACKING_UP = 3
    STOPPED    = 4


# ── RoombaController ──────────────────────────────────────────────────────────

class RoombaController:
    """
    Thread-safe interface to the Roomba 650 via the iRobot Open Interface.

    The class owns:
      - A serial.Serial connection on the configured port.
      - A background daemon thread for sensor polling.

    Public API (all methods are thread-safe):
      connect()           → open serial, wake, start OI, upload songs
      disconnect()        → send STOP, close serial
      drive(vel, radius)  → set drive velocity + radius
      drive_direct(r, l)  → set individual wheel velocities
      stop_drive()        → zero velocity
      get_sensors()       → return latest SensorData snapshot
      play_song(name)     → play a named song from SONGS dict
      set_safe_mode()     → enter OI SAFE mode
      set_full_mode()     → enter OI FULL mode (disables cliff/bump safety)
    """

    def __init__(self, config: dict):
        sys_cfg = config["system"]
        self._port     = sys_cfg["port"]
        self._baud     = sys_cfg["baud"]
        self._timeout  = sys_cfg.get("roomba_timeout", 2.0)
        self._poll_interval = sys_cfg.get("sensor_poll_interval", 0.1)
        self._max_speed     = sys_cfg.get("max_drive_speed", 250)
        self._wake_pulse    = sys_cfg.get("wakeup_pulse_duration", 0.5)
        self._wake_settle   = sys_cfg.get("wakeup_settle_delay", 1.0)
        self._enable_songs  = config.get("features", {}).get("enable_songs", True)

        self._ser: Optional[serial.Serial] = None
        self._serial_lock = threading.RLock()    # guards all serial I/O
        self._sensor_lock = threading.Lock()     # guards _sensor_data reads/writes

        self._sensor_data: SensorData = SensorData()
        self._connected: bool = False
        self._oi_mode: str = "off"               # "off" | "passive" | "safe" | "full"

        self._sensor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Connection Lifecycle ──────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Open serial port, wake the Roomba via RTS pulse, initialize OI,
        and start the background sensor polling thread.

        Returns:
            True if connection succeeded, False otherwise.
        """
        log.info(f"Connecting to Roomba on {self._port} @ {self._baud} baud …")
        try:
            self._ser = serial.Serial(
                port=self._port,
                baudrate=self._baud,
                timeout=self._timeout,
                rtscts=False,
                dsrdtr=False,
            )
            self._ser.flushInput()
            self._ser.flushOutput()
        except serial.SerialException as e:
            log.error(f"Failed to open serial port {self._port}: {e}")
            return False

        # Wake via Device Detect (RTS) pulse
        self._wake_via_rts()

        # Initialize OI
        self._send_bytes([OI.START])
        time.sleep(0.1)
        self._send_bytes([OI.SAFE])
        time.sleep(0.1)
        self._oi_mode = "safe"
        log.info("OI initialized — SAFE mode active.")

        # Upload all songs
        if self._enable_songs:
            self._upload_all_songs()

        self._connected = True
        self._stop_event.clear()

        # Start sensor polling daemon thread
        self._sensor_thread = threading.Thread(
            target=self._sensor_loop,
            name="RoombaSensor",
            daemon=True,
        )
        self._sensor_thread.start()
        log.info("Sensor polling thread started.")
        return True

    def disconnect(self) -> None:
        """Send POWER opcode (sleep), stop the sensor thread, and close serial."""
        log.info("Disconnecting Roomba …")
        self._stop_event.set()

        if self._sensor_thread and self._sensor_thread.is_alive():
            self._sensor_thread.join(timeout=2.0)

        if self._ser and self._ser.isOpen():
            try:
                self.stop_drive()
                time.sleep(0.05)
                # Return to passive (safe power-down)
                with self._serial_lock:
                    self._send_bytes([OI.POWER])
            except Exception:
                pass
            finally:
                self._ser.close()

        self._connected = False
        self._oi_mode = "off"
        log.info("Roomba disconnected.")

    # ── Wake-Up Pulse ─────────────────────────────────────────────────────────

    def _wake_via_rts(self) -> None:
        """
        Assert Device Detect LOW via RTS for `_wake_pulse` seconds, then release.

        On most TTL USB-UART adapters (CH340, CP2102, FTDI in TTL mode):
          ser.rts = True  → Pin driven LOW  (asserts DD)
          ser.rts = False → Pin driven HIGH (releases DD)

        If your adapter has inverted RTS logic, swap True/False below.
        Verify with a multimeter: the RTS pin should read ~0V during the pulse.
        """
        log.debug(f"Sending Device Detect (RTS) wake pulse for {self._wake_pulse}s …")
        with self._serial_lock:
            self._ser.rts = True          # Assert LOW → Device Detect active
            time.sleep(self._wake_pulse)
            self._ser.rts = False         # Release
            time.sleep(self._wake_settle) # Allow Roomba firmware to boot
        log.debug("Wake pulse complete.")

    # ── OI Mode Management ────────────────────────────────────────────────────

    def set_safe_mode(self) -> None:
        """Enter SAFE mode: full OI control with cliff/wheel-drop safety halts."""
        with self._serial_lock:
            self._send_bytes([OI.SAFE])
        self._oi_mode = "safe"
        log.debug("OI mode → SAFE")

    def set_full_mode(self) -> None:
        """
        Enter FULL mode: complete OI control, safety checks disabled.
        ⚠ In FULL mode the Roomba WILL drive off cliffs. Use with caution.
        """
        with self._serial_lock:
            self._send_bytes([OI.FULL])
        self._oi_mode = "full"
        log.warning("OI mode → FULL (cliff/bump safety DISABLED)")

    # ── Drive Commands ────────────────────────────────────────────────────────

    def drive(self, velocity: int, radius: int) -> None:
        """
        Set drive velocity and turn radius (Opcode 137).

        Args:
            velocity: Forward speed in mm/s. Range: -500 to 500.
                      Positive = forward, negative = backward.
            radius:   Turn radius in mm. Range: -2000 to 2000.
                      Special values:
                        0x8000 (32768) = drive straight
                        1              = spin clockwise in place
                        -1             = spin counter-clockwise in place

        The velocity is clamped to ±`max_drive_speed` from config.yaml.
        """
        if not self._connected:
            return
        velocity = max(-self._max_speed, min(self._max_speed, int(velocity)))
        # Allow full radius range (not speed-limited)
        radius = max(-2000, min(2000, int(radius)))
        data = struct.pack(">Bhh", OI.DRIVE, velocity, radius)
        with self._serial_lock:
            self._ser.write(data)

    def drive_direct(self, right_mm_s: int, left_mm_s: int) -> None:
        """
        Drive each wheel at an independent velocity (Opcode 145).
        Positive = forward, negative = backward. Range: -500 to 500 mm/s.
        """
        if not self._connected:
            return
        r = max(-self._max_speed, min(self._max_speed, int(right_mm_s)))
        l = max(-self._max_speed, min(self._max_speed, int(left_mm_s)))
        data = struct.pack(">Bhh", OI.DRIVE_DIRECT, r, l)
        with self._serial_lock:
            self._ser.write(data)

    def stop_drive(self) -> None:
        """Immediately halt all wheel motion."""
        if not self._connected:
            return
        data = struct.pack(">Bhh", OI.DRIVE, 0, 0)
        with self._serial_lock:
            self._ser.write(data)

    def spin_left(self, speed: int = 150) -> None:
        """Spin counter-clockwise in place."""
        self.drive(speed, -1)

    def spin_right(self, speed: int = 150) -> None:
        """Spin clockwise in place."""
        self.drive(speed, 1)

    def drive_forward(self, speed: Optional[int] = None) -> None:
        """Drive straight forward."""
        speed = speed or self._max_speed
        self.drive(speed, 0x8000)

    def drive_backward(self, speed: Optional[int] = None) -> None:
        """Drive straight backward."""
        speed = speed or self._max_speed
        self.drive(-speed, 0x8000)

    # ── Song System ───────────────────────────────────────────────────────────

    def _upload_all_songs(self) -> None:
        """Upload all predefined songs to Roomba slots 0-4 at boot time."""
        for name, slot in SONG_SLOTS.items():
            notes = SONGS[name]
            self._define_song(slot, notes)
        log.debug(f"Uploaded {len(SONG_SLOTS)} songs to Roomba.")

    def _define_song(self, slot: int, notes: list[tuple[int, int]]) -> None:
        """
        Define a song in a Roomba song slot (Opcode 140).

        Args:
            slot:  Song slot index (0–4).
            notes: List of (note_number, duration_64ths) tuples.
                   Note numbers: 31–127 (MIDI). Duration: 1–255 (1/64 sec units).
        """
        notes = notes[:16]  # Roomba limit: 16 notes per song
        payload = [OI.SONG, slot, len(notes)]
        for note, dur in notes:
            payload.extend([note, dur])
        with self._serial_lock:
            self._ser.write(bytes(payload))
        time.sleep(0.02)

    def play_song(self, name: str) -> bool:
        """
        Play a named song from the SONGS dictionary.

        Args:
            name: Key in the SONGS dict (e.g., "startup", "happy", "alert").

        Returns:
            True if the song was found and play command sent, False otherwise.
        """
        if not self._connected or not self._enable_songs:
            return False
        if name not in SONG_SLOTS:
            log.warning(f"Song '{name}' not in SONG_SLOTS. Available: {list(SONG_SLOTS.keys())}")
            return False
        slot = SONG_SLOTS[name]
        with self._serial_lock:
            self._send_bytes([OI.PLAY, slot])
        log.debug(f"Playing song '{name}' from slot {slot}.")
        return True

    # ── Sensor Polling ────────────────────────────────────────────────────────

    def _sensor_loop(self) -> None:
        """
        Background daemon thread: poll Roomba sensors at `_poll_interval` Hz.
        On serial error, waits 1 second and retries. Exits on stop event.
        """
        log.debug("Sensor loop running …")
        while not self._stop_event.is_set():
            try:
                data = self._request_sensors()
                if data is not None:
                    with self._sensor_lock:
                        self._sensor_data = data
            except serial.SerialException as e:
                log.error(f"Sensor loop serial error: {e} — retrying in 1s")
                time.sleep(1.0)
            except Exception as e:
                log.error(f"Sensor loop unexpected error: {e}")
                time.sleep(1.0)
            time.sleep(self._poll_interval)
        log.debug("Sensor loop terminated.")

    def _request_sensors(self) -> Optional[SensorData]:
        """
        Send a QUERY_LIST request for the sensor packets in _QUERY_PACKETS
        and parse the binary response into a SensorData instance.

        Returns:
            Parsed SensorData, or None if the response was malformed.
        """
        cmd = [OI.QUERY_LIST, len(_QUERY_PACKETS)] + [int(p) for p in _QUERY_PACKETS]
        with self._serial_lock:
            self._ser.flushInput()
            self._ser.write(bytes(cmd))
            raw = self._ser.read(_SENSOR_SIZE)

        if len(raw) != _SENSOR_SIZE:
            log.debug(f"Sensor read short: expected {_SENSOR_SIZE}B, got {len(raw)}B")
            return None

        try:
            (bumps, clf_l, clf_fl, clf_fr, clf_r,
             voltage, temp, charge, capacity) = struct.unpack(_SENSOR_FORMAT, raw)
        except struct.error as e:
            log.warning(f"Sensor unpack error: {e}")
            return None

        sd = SensorData(
            bump_right          = bool(bumps & 0x01),
            bump_left           = bool(bumps & 0x02),
            wheel_drop_right    = bool(bumps & 0x04),
            wheel_drop_left     = bool(bumps & 0x08),
            cliff_left          = bool(clf_l),
            cliff_front_left    = bool(clf_fl),
            cliff_front_right   = bool(clf_fr),
            cliff_right         = bool(clf_r),
            voltage_mv          = voltage,
            temperature_c       = temp,
            battery_charge_mah  = charge,
            battery_capacity_mah= capacity,
            timestamp           = time.monotonic(),
        )
        return sd

    def get_sensors(self) -> SensorData:
        """
        Return the most recent SensorData snapshot (thread-safe).
        This never blocks — returns the last successfully polled data.
        """
        with self._sensor_lock:
            return self._sensor_data

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def oi_mode(self) -> str:
        return self._oi_mode

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _send_bytes(self, data: list[int]) -> None:
        """Write a list of integers as raw bytes. Caller must hold _serial_lock."""
        self._ser.write(bytes(data))

    def __repr__(self) -> str:
        return (
            f"<RoombaController port={self._port} "
            f"connected={self._connected} mode={self._oi_mode}>"
        )
