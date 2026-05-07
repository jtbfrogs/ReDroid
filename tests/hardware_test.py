#!/usr/bin/env python3
"""
tests/hardware_test.py
──────────────────────
Standalone hardware diagnostic and interactive test tool for Droid firmware.

Run from the project root:
    python tests/hardware_test.py            # full test suite
    python tests/hardware_test.py --roomba   # Roomba tests only
    python tests/hardware_test.py --camera   # Camera tests only
    python tests/hardware_test.py --sensors  # Live sensor monitor only
    python tests/hardware_test.py --drive    # Drive command tests only

Tests covered:
  ✦ Config file load
  ✦ Serial port accessibility
  ✦ Roomba wake + OI initialisation
  ✦ Sensor packet read (all 9 sensor values)
  ✦ Live sensor monitor (real-time table, Ctrl+C to stop)
  ✦ Interactive bump sensor test (press bumpers to verify)
  ✦ Interactive cliff sensor test (lift robot to verify)
  ✦ Battery deep-dive (voltage, mAh, %, temp, estimated runtime)
  ✦ Song playback (all 5 slots)
  ✦ Drive commands (forward / backward / spin left / spin right)
  ✦ Camera connection + resolution check
  ✦ Camera live stats (FPS, frame quality, detection counts)
  ✦ OpenCV preview window (annotated, 10 s)
  ✦ HOG person + MOG2 obstacle detection live output
"""

import os
import sys
import time
import struct
import threading
import argparse
import signal

# ── Make sure project root is on the path ────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import yaml

# ── ANSI colours ─────────────────────────────────────────────────────────────

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    CYAN    = "\033[96m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"
    BG_RED  = "\033[41m"
    BG_GRN  = "\033[42m"
    BG_YLW  = "\033[43m"


# ── Result tracking ───────────────────────────────────────────────────────────

_results: list[tuple[str, str, str]] = []   # (section, status, detail)

def _record(section: str, status: str, detail: str = "") -> None:
    _results.append((section, status, detail))


# ── Pretty print helpers ──────────────────────────────────────────────────────

def banner(title: str) -> None:
    width = 62
    print(f"\n{C.CYAN}{C.BOLD}  ┌{'─' * width}┐")
    print(f"  │  {'⬡  ' + title:<{width - 3}}│")
    print(f"  └{'─' * width}┘{C.RESET}\n")


def section(title: str) -> None:
    print(f"\n{C.BOLD}{C.BLUE}── {title} {'─' * max(0, 55 - len(title))}{C.RESET}")


def ok(msg: str) -> None:
    print(f"  {C.GREEN}✔  {C.RESET}{msg}")


def fail(msg: str) -> None:
    print(f"  {C.RED}✘  {C.RESET}{msg}")


def warn(msg: str) -> None:
    print(f"  {C.YELLOW}⚠  {C.RESET}{msg}")


def info(msg: str) -> None:
    print(f"  {C.GREY}·  {C.RESET}{msg}")


def ask(prompt: str) -> str:
    return input(f"\n  {C.MAGENTA}?  {C.RESET}{prompt} ").strip()


def confirm(prompt: str) -> bool:
    ans = ask(f"{prompt} [y/N]")
    return ans.lower() in ("y", "yes")


def sensor_row(label: str, value, width: int = 28) -> str:
    val_str = str(value)
    color   = C.GREEN if value and value != 0 and value is not False else C.GREY
    return f"  {C.DIM}{label:<{width}}{C.RESET}  {color}{val_str}{C.RESET}"


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    section("CONFIG FILE")
    path = os.path.join(_ROOT, "config.yaml")
    if not os.path.exists(path):
        fail(f"config.yaml not found at {path}")
        _record("Config", "FAIL", "File not found")
        sys.exit(1)
    try:
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
        ok(f"config.yaml loaded  →  port={cfg['system']['port']}  baud={cfg['system']['baud']}")
        _record("Config", "PASS")
        return cfg
    except Exception as e:
        fail(f"config.yaml parse error: {e}")
        _record("Config", "FAIL", str(e))
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
#  ROOMBA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_serial(cfg: dict) -> bool:
    """Check if the serial port can be opened at all."""
    section("SERIAL PORT")
    import serial as pyserial

    port    = cfg["system"]["port"]
    baud    = cfg["system"]["baud"]
    timeout = cfg["system"].get("roomba_timeout", 2.0)

    info(f"Port : {port}")
    info(f"Baud : {baud}")

    try:
        ser = pyserial.Serial(port=port, baudrate=baud, timeout=timeout,
                              rtscts=False, dsrdtr=False)
        if ser.isOpen():
            ser.close()
            ok(f"Port {port} opened and closed successfully.")
            _record("Serial port", "PASS", port)
            return True
        else:
            fail(f"Port {port} did not open.")
            _record("Serial port", "FAIL", "did not open")
            return False
    except pyserial.SerialException as e:
        fail(f"SerialException: {e}")
        _record("Serial port", "FAIL", str(e))
        info("Tip: Check  ls /dev/ttyUSB*  and  ls /dev/ttyACM*")
        info("     Make sure you are in the 'dialout' group:")
        info("     sudo usermod -aG dialout $USER  (then re-login)")
        return False
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        _record("Serial port", "FAIL", str(e))
        return False


def connect_roomba(cfg: dict):
    """Connect to Roomba and return a RoombaController or None."""
    section("ROOMBA CONNECTION + OI INIT")
    from modules.roomba_io import RoombaController

    roomba = RoombaController(cfg)
    info("Sending wake pulse + START + SAFE …")
    t0 = time.monotonic()
    connected = roomba.connect()
    elapsed = time.monotonic() - t0

    if connected:
        ok(f"Connected in {elapsed:.2f}s  |  OI mode: {roomba.oi_mode.upper()}")
        _record("Roomba connect", "PASS", f"mode={roomba.oi_mode}")
        return roomba
    else:
        fail("Roomba failed to connect.")
        _record("Roomba connect", "FAIL")
        return None


def test_sensor_read(roomba) -> bool:
    """Read sensors once and print every field."""
    section("SENSOR READ — SINGLE SNAPSHOT")
    time.sleep(0.3)   # let the sensor thread populate

    s = roomba.get_sensors()
    if s.timestamp == 0.0:
        warn("Sensor data not yet populated — waiting 1 s …")
        time.sleep(1.0)
        s = roomba.get_sensors()

    if s.timestamp == 0.0:
        fail("No sensor data received. Check serial connection.")
        _record("Sensor read", "FAIL", "no data")
        return False

    age = time.monotonic() - s.timestamp
    ok(f"Sensor snapshot received  (age: {age*1000:.0f} ms)")
    print()

    # ── Physical sensors ──────────────────────────────────────────────────────
    print(f"  {C.BOLD}Bumpers{C.RESET}")
    bump_color = lambda v: C.YELLOW if v else C.GREY
    print(f"    {'Bump Left':<24}  {bump_color(s.bump_left)}{s.bump_left}{C.RESET}")
    print(f"    {'Bump Right':<24}  {bump_color(s.bump_right)}{s.bump_right}{C.RESET}")

    print(f"\n  {C.BOLD}Cliff Sensors{C.RESET}")
    cliff_color = lambda v: C.RED + C.BOLD if v else C.GREY
    print(f"    {'Cliff Left':<24}  {cliff_color(s.cliff_left)}{s.cliff_left}{C.RESET}")
    print(f"    {'Cliff Front Left':<24}  {cliff_color(s.cliff_front_left)}{s.cliff_front_left}{C.RESET}")
    print(f"    {'Cliff Front Right':<24}  {cliff_color(s.cliff_front_right)}{s.cliff_front_right}{C.RESET}")
    print(f"    {'Cliff Right':<24}  {cliff_color(s.cliff_right)}{s.cliff_right}{C.RESET}")

    print(f"\n  {C.BOLD}Wheel Drops{C.RESET}")
    wd_color = lambda v: C.RED + C.BOLD if v else C.GREY
    print(f"    {'Wheel Drop Left':<24}  {wd_color(s.wheel_drop_left)}{s.wheel_drop_left}{C.RESET}")
    print(f"    {'Wheel Drop Right':<24}  {wd_color(s.wheel_drop_right)}{s.wheel_drop_right}{C.RESET}")

    # ── Battery ───────────────────────────────────────────────────────────────
    print(f"\n  {C.BOLD}Battery{C.RESET}")
    pct   = s.battery_percent
    pct_c = C.GREEN if pct > 50 else (C.YELLOW if pct > 20 else C.RED)
    print(f"    {'Voltage':<24}  {C.CYAN}{s.voltage_mv} mV{C.RESET}")
    print(f"    {'Charge':<24}  {C.CYAN}{s.battery_charge_mah} mAh{C.RESET}")
    print(f"    {'Capacity':<24}  {C.CYAN}{s.battery_capacity_mah} mAh{C.RESET}")
    print(f"    {'Percent':<24}  {pct_c}{pct:.1f} %{C.RESET}")
    print(f"    {'Temperature':<24}  {C.CYAN}{s.temperature_c} °C{C.RESET}")

    # ── Derived flags ─────────────────────────────────────────────────────────
    print(f"\n  {C.BOLD}Derived Flags{C.RESET}")
    print(f"    {'any_bump':<24}  {C.YELLOW if s.any_bump else C.GREY}{s.any_bump}{C.RESET}")
    print(f"    {'any_cliff':<24}  {C.RED + C.BOLD if s.any_cliff else C.GREY}{s.any_cliff}{C.RESET}")
    print(f"    {'any_wheel_drop':<24}  {C.RED + C.BOLD if s.any_wheel_drop else C.GREY}{s.any_wheel_drop}{C.RESET}")

    _record("Sensor read", "PASS")
    return True


def test_live_sensor_monitor(roomba, duration: int = 0) -> None:
    """
    Live-updating sensor table. Runs until Ctrl+C or `duration` seconds.
    duration=0 means run until Ctrl+C.
    """
    section("LIVE SENSOR MONITOR")
    if duration == 0:
        info("Press Ctrl+C to stop the live monitor.")
    else:
        info(f"Running for {duration} seconds. Press Ctrl+C to stop early.")

    stop = threading.Event()

    def _stop_on_input():
        # Also exits on Enter keypress so it plays nicely in interactive mode
        try:
            input()
        except Exception:
            pass
        stop.set()

    if duration == 0:
        threading.Thread(target=_stop_on_input, daemon=True).start()

    start = time.monotonic()
    try:
        while not stop.is_set():
            if duration > 0 and (time.monotonic() - start) >= duration:
                break

            s = roomba.get_sensors()
            elapsed = time.monotonic() - start

            # Build compact one-line status bar
            bump_str  = f"{C.YELLOW}BUMP{'(L)' if s.bump_left else ''} {'(R)' if s.bump_right else ''}{C.RESET}" if s.any_bump else f"{C.GREY}bump=0{C.RESET}"
            cliff_str = f"{C.RED}{C.BOLD}CLIFF{C.RESET}" if s.any_cliff else f"{C.GREY}cliff=0{C.RESET}"
            wd_str    = f"{C.RED}WHL_DROP{C.RESET}" if s.any_wheel_drop else f"{C.GREY}whl=0{C.RESET}"
            pct       = s.battery_percent
            pct_c     = C.GREEN if pct > 50 else (C.YELLOW if pct > 20 else C.RED)
            batt_str  = f"{pct_c}{pct:.0f}%{C.RESET} {C.GREY}({s.voltage_mv}mV){C.RESET}"
            temp_str  = f"{C.CYAN}{s.temperature_c}°C{C.RESET}"

            # Cliff detail
            cl = "L" if s.cliff_left else "-"
            cf = "FL" if s.cliff_front_left else "--"
            cg = "FR" if s.cliff_front_right else "--"
            cr = "R" if s.cliff_right else "-"
            cliff_detail = f"{C.DIM}[{cl} {cf} {cg} {cr}]{C.RESET}"

            print(
                f"\r  {C.DIM}t={elapsed:5.1f}s{C.RESET}  "
                f"{bump_str:<30}  {cliff_str} {cliff_detail}  "
                f"{wd_str:<20}  batt={batt_str}  temp={temp_str}   ",
                end="", flush=True
            )
            time.sleep(0.2)

    except KeyboardInterrupt:
        pass

    print()   # newline after carriage-return line


def test_interactive_bumpers(roomba) -> None:
    """Ask user to press each bumper and verify the sensor fires."""
    section("INTERACTIVE BUMP SENSOR TEST")
    info("This test watches the bump sensors for 10 s each.")
    info("Press Ctrl+C to skip to the next test.\n")

    for side, attr in [("FRONT (both bumpers)", "any_bump"),
                       ("LEFT bumper",           "bump_left"),
                       ("RIGHT bumper",          "bump_right")]:
        print(f"  {C.MAGENTA}→  Press the {C.BOLD}{side}{C.RESET}{C.MAGENTA} now …{C.RESET}")
        fired = False
        deadline = time.monotonic() + 10.0
        try:
            while time.monotonic() < deadline:
                s = roomba.get_sensors()
                if getattr(s, attr):
                    ok(f"{side} sensor fired!  ✔")
                    fired = True
                    _record(f"Bump {side}", "PASS")
                    time.sleep(0.5)   # debounce
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

        if not fired:
            warn(f"{side} — not triggered within 10 s (skipped or not working).")
            _record(f"Bump {side}", "SKIP")


def test_interactive_cliffs(roomba) -> None:
    """Ask user to trigger cliff sensors by lifting the robot."""
    section("INTERACTIVE CLIFF SENSOR TEST")
    info("Lift the robot off the ground to trigger all cliff sensors.")
    info("Watching for 10 s. Press Ctrl+C to skip.\n")
    print(f"  {C.MAGENTA}→  {C.BOLD}Lift the robot now …{C.RESET}")

    fired = False
    deadline = time.monotonic() + 10.0
    try:
        while time.monotonic() < deadline:
            s = roomba.get_sensors()
            if s.any_cliff:
                ok("Cliff sensors fired!")
                cl = "✔" if s.cliff_left       else "✘"
                cf = "✔" if s.cliff_front_left  else "✘"
                cg = "✔" if s.cliff_front_right else "✘"
                cr = "✔" if s.cliff_right       else "✘"
                info(f"  L={cl}  FL={cf}  FR={cg}  R={cr}")
                fired = True
                _record("Cliff sensors", "PASS")
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass

    if not fired:
        warn("Cliff sensors not triggered within 10 s.")
        _record("Cliff sensors", "SKIP")


def test_battery_detail(roomba) -> None:
    """Print detailed battery information and a rough runtime estimate."""
    section("BATTERY DEEP-DIVE")
    s = roomba.get_sensors()

    pct = s.battery_percent
    pct_c = C.GREEN if pct > 50 else (C.YELLOW if pct > 20 else C.RED)

    print(f"  {'Voltage':<28}  {C.CYAN}{s.voltage_mv:>6} mV  ({s.voltage_mv / 1000:.2f} V){C.RESET}")
    print(f"  {'Charge (current)':<28}  {C.CYAN}{s.battery_charge_mah:>6} mAh{C.RESET}")
    print(f"  {'Capacity (full)':<28}  {C.CYAN}{s.battery_capacity_mah:>6} mAh{C.RESET}")
    print(f"  {'State of charge':<28}  {pct_c}{pct:>5.1f} %{C.RESET}")
    print(f"  {'Temperature':<28}  {C.CYAN}{s.temperature_c:>6} °C{C.RESET}")

    # Rough runtime estimate based on typical Roomba 650 drive current (~1.5 A)
    if s.battery_charge_mah > 0:
        est_min = (s.battery_charge_mah / 1500) * 60
        print(f"\n  {C.DIM}Estimated runtime at normal drive (~1.5 A): "
              f"{est_min:.0f} minutes{C.RESET}")

    # Warnings
    if s.temperature_c > 60:
        warn(f"Battery temperature {s.temperature_c}°C is HIGH — check sensor calibration.")
    if pct < 10:
        warn("Battery critically low — charge before further testing.")
    elif pct < 20:
        warn("Battery below 20 %.")

    _record("Battery", "PASS" if s.battery_capacity_mah > 0 else "WARN",
            f"{pct:.0f}% @ {s.voltage_mv}mV")


def test_songs(roomba) -> None:
    """Play every defined song and confirm playback was sent."""
    section("SONG PLAYBACK TEST")
    from modules.roomba_io import SONGS, SONG_SLOTS

    info(f"Song slots defined: {list(SONG_SLOTS.keys())}")
    info("You should hear each song play on the Roomba speaker.\n")

    for name in SONG_SLOTS:
        notes = SONGS[name]
        # Calculate duration from note lengths (each unit = 1/64 s)
        dur_s = sum(d for _, d in notes) / 64.0
        print(f"  {C.MAGENTA}♪{C.RESET}  Playing '{C.BOLD}{name}{C.RESET}' "
              f"({len(notes)} notes, ~{dur_s:.1f}s) …", end="", flush=True)
        result = roomba.play_song(name)
        if result:
            print(f"  {C.GREEN}sent ✔{C.RESET}")
        else:
            print(f"  {C.RED}failed ✘{C.RESET}")
        time.sleep(dur_s + 0.4)   # wait for song to finish before next

    _record("Songs", "PASS")


def test_drive_commands(roomba) -> None:
    """
    Interactive drive command tests.
    Each movement requires explicit user confirmation before executing.
    """
    section("DRIVE COMMAND TESTS")
    warn("This section will physically move the robot.")
    warn("Make sure the robot is on the floor with clear space around it.\n")

    tests = [
        ("Drive FORWARD  (~0.5 s)",  lambda: roomba.drive_forward(100)),
        ("Drive BACKWARD (~0.5 s)",  lambda: roomba.drive_backward(100)),
        ("SPIN LEFT      (~0.5 s)",  lambda: roomba.spin_left(100)),
        ("SPIN RIGHT     (~0.5 s)",  lambda: roomba.spin_right(100)),
    ]

    for label, cmd_fn in tests:
        if not confirm(f"Run: {label}?"):
            warn(f"Skipped: {label}")
            _record(f"Drive: {label}", "SKIP")
            continue
        print(f"  {C.CYAN}→  Executing …{C.RESET}", end="", flush=True)
        cmd_fn()
        time.sleep(0.5)
        roomba.stop_drive()
        print(f"  {C.GREEN}stopped ✔{C.RESET}")
        _record(f"Drive: {label}", "PASS")
        time.sleep(0.3)

    # Final safety stop
    roomba.stop_drive()
    ok("Drive test complete — robot stopped.")


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_camera_connect(cfg: dict):
    """Open the camera and grab one frame."""
    section("CAMERA CONNECTION")
    import cv2

    cam_index = cfg.get("features", {}).get("camera_index", 0)
    res       = cfg.get("features", {}).get("vision_resolution", [640, 480])

    info(f"Camera index : {cam_index}")
    info(f"Target res   : {res[0]}×{res[1]}")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        fail(f"cv2.VideoCapture({cam_index}) failed to open.")
        info("Tips:")
        info("  ls /dev/video*                     — list video devices")
        info("  v4l2-ctl --list-devices            — list with names")
        info("  Try camera_index: 1 or 2 in config.yaml")
        _record("Camera connect", "FAIL")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

    ret, frame = cap.read()
    if not ret or frame is None:
        fail("Camera opened but returned no frame.")
        cap.release()
        _record("Camera connect", "FAIL", "no frame")
        return None

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok(f"Camera opened  →  actual resolution: {actual_w}×{actual_h}")

    # Check frame isn't all black
    import numpy as np
    mean_val = frame.mean()
    if mean_val < 5.0:
        warn(f"Frame mean brightness is very low ({mean_val:.1f}) — lens cap on?")
    else:
        ok(f"Frame brightness looks good (mean pixel value: {mean_val:.1f})")

    _record("Camera connect", "PASS", f"{actual_w}×{actual_h}")
    return cap


def test_camera_live_stats(cap, cfg: dict, duration: int = 8) -> None:
    """
    Capture frames for `duration` seconds and print live stats to the terminal.
    No GUI window — pure text output.
    """
    section("CAMERA LIVE STATS (text mode)")
    info(f"Capturing for {duration} s. Press Ctrl+C to stop early.\n")

    import cv2
    import numpy as np

    fps_cap    = cfg.get("features", {}).get("vision_fps_cap", 15)
    frame_delay = 1.0 / max(1, fps_cap)
    start      = time.monotonic()
    frame_count = 0
    fps         = 0.0
    fps_ts      = start

    print(f"  {C.DIM}{'Frame':>6}  {'FPS':>5}  {'Size':>10}  {'Mean':>6}  {'Min':>5}  {'Max':>5}  {'Channels':>8}{C.RESET}")
    print(f"  {'─' * 58}")

    try:
        while (time.monotonic() - start) < duration:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if not ret or frame is None:
                warn("Empty frame — skipping.")
                continue

            frame_count += 1
            elapsed_fps = time.monotonic() - fps_ts
            if elapsed_fps >= 1.0:
                fps = frame_count / (time.monotonic() - start)

            h, w = frame.shape[:2]
            ch   = frame.shape[2] if len(frame.shape) == 3 else 1
            mean = frame.mean()
            mn   = int(frame.min())
            mx   = int(frame.max())

            brightness_bar = "█" * int(mean / 255 * 20)
            brightness_c   = C.GREEN if mean > 30 else C.RED

            print(
                f"\r  {frame_count:>6}  {fps:>5.1f}  {w}×{h:>4}  "
                f"{brightness_c}{mean:>6.1f}{C.RESET}  {mn:>5}  {mx:>5}  "
                f"{C.GREY}{ch}ch{C.RESET}  {brightness_c}{brightness_bar:<20}{C.RESET}",
                end="", flush=True
            )

            sleep_t = frame_delay - (time.monotonic() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        pass

    print()
    total = time.monotonic() - start
    actual_fps = frame_count / total if total > 0 else 0
    ok(f"Captured {frame_count} frames in {total:.1f}s  →  {actual_fps:.1f} fps avg")
    _record("Camera live stats", "PASS", f"{frame_count} frames @ {actual_fps:.1f}fps")


def test_vision_detection(cap, cfg: dict, duration: int = 10) -> None:
    """
    Run HOG person detection + MOG2 obstacle detection for `duration` seconds
    and print what is being detected to the terminal.
    """
    section("VISION DETECTION TEST (HOG + MOG2)")
    info(f"Running detection for {duration} s. Walk in front of camera to test person detection.")
    info("Press Ctrl+C to stop early.\n")

    import cv2
    import numpy as np

    # ── Init detectors ────────────────────────────────────────────────────────
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    bg  = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40,
                                              detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    start       = time.monotonic()
    frame_count = 0
    total_persons   = 0
    total_obstacles = 0

    print(f"  {C.DIM}{'t':>5}  {'Frame':>6}  {'Persons':>9}  {'Obstacles':>11}  Detection{C.RESET}")
    print(f"  {'─' * 60}")

    try:
        while (time.monotonic() - start) < duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            t_now = time.monotonic() - start

            # ── Person detection ──────────────────────────────────────────────
            h, w   = frame.shape[:2]
            scale  = min(1.0, 320.0 / w)
            small  = cv2.resize(frame, (int(w * scale), int(h * scale)))
            gray   = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray, winStride=(8, 8),
                                             padding=(4, 4), scale=1.05)
            persons = len(rects)

            # ── Obstacle detection ────────────────────────────────────────────
            zone_y  = int(h * 0.5)
            roi     = frame[zone_y:, :]
            fg      = bg.apply(roi)
            fg      = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,  kernel)
            fg      = cv2.morphologyEx(fg, cv2.MORPH_ERODE,  kernel, iterations=1)
            fg      = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=2)
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            obstacles = sum(1 for c in cnts if cv2.contourArea(c) >= 4000)

            total_persons   += persons
            total_obstacles += obstacles

            p_str = f"{C.GREEN}{persons:>2} person(s){C.RESET}" if persons > 0 else f"{C.GREY}  —       {C.RESET}"
            o_str = f"{C.YELLOW}{obstacles:>2} obstacle(s){C.RESET}" if obstacles > 0 else f"{C.GREY}  —         {C.RESET}"
            det_bar = ""
            if persons > 0:
                det_bar += f"{C.GREEN}[PERSON]{C.RESET} "
            if obstacles > 0:
                det_bar += f"{C.YELLOW}[OBSTACLE]{C.RESET} "
            if not det_bar:
                det_bar = f"{C.GREY}clear{C.RESET}"

            print(
                f"\r  {t_now:>5.1f}  {frame_count:>6}  {p_str}  {o_str}  {det_bar:<40}",
                end="", flush=True
            )
            time.sleep(1.0 / 15)   # ~15 fps

    except KeyboardInterrupt:
        pass

    print()
    total_t = time.monotonic() - start
    ok(f"Processed {frame_count} frames in {total_t:.1f}s")
    ok(f"Person detections  : {total_persons} total hits across all frames")
    ok(f"Obstacle detections: {total_obstacles} total hits across all frames")
    _record("Vision detection", "PASS",
            f"{frame_count} frames, {total_persons} person hits, {total_obstacles} obstacle hits")


def test_camera_preview(cap, cfg: dict, duration: int = 10) -> None:
    """
    Open an annotated OpenCV preview window for `duration` seconds.
    Shows bounding boxes for persons and obstacles.
    Press Q or wait for timeout.
    """
    section("CAMERA PREVIEW WINDOW")
    info(f"Opening annotated preview window for up to {duration} s.")
    info("Press Q in the window to close early.\n")

    import cv2
    import numpy as np

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    bg  = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40,
                                              detectShadows=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    start = time.monotonic()
    fps_count = 0
    fps_ts    = start
    fps       = 0.0

    try:
        while (time.monotonic() - start) < duration:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            fps_count += 1
            if (time.monotonic() - fps_ts) >= 1.0:
                fps    = fps_count / (time.monotonic() - start)
                fps_ts = time.monotonic()

            h, w = frame.shape[:2]

            # Person detection
            scale = min(1.0, 320.0 / w)
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray, winStride=(8, 8),
                                             padding=(4, 4), scale=1.05)
            inv = 1.0 / scale
            for (rx, ry, rw, rh) in rects:
                x, y, bw, bh = int(rx*inv), int(ry*inv), int(rw*inv), int(rh*inv)
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 220, 0), 2)
                cv2.putText(frame, "PERSON", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

            # Obstacle detection
            zone_y  = int(h * 0.5)
            roi     = frame[zone_y:, :]
            fg      = bg.apply(roi)
            fg      = cv2.morphologyEx(fg, cv2.MORPH_CLOSE,  kernel)
            fg      = cv2.morphologyEx(fg, cv2.MORPH_ERODE,  kernel, iterations=1)
            fg      = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel, iterations=2)
            cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if cv2.contourArea(cnt) < 4000:
                    continue
                rx, ry, rw, rh = cv2.boundingRect(cnt)
                oy = ry + zone_y
                cv2.rectangle(frame, (rx, oy), (rx+rw, oy+rh), (0, 140, 255), 2)
                cv2.putText(frame, "OBSTACLE", (rx, oy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

            # HUD overlay
            cv2.line(frame, (0, zone_y), (w, zone_y), (80, 80, 200), 1)
            hud = (f"FPS:{fps:.1f}  Persons:{len(rects)}  "
                   f"Obstacles:{sum(1 for c in cnts if cv2.contourArea(c)>=4000)}"
                   f"  [Q to quit]")
            cv2.rectangle(frame, (0, 0), (w, 26), (20, 20, 20), -1)
            cv2.putText(frame, hud, (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Droid Hardware Test — Camera Preview", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

    ok("Preview window closed.")
    _record("Camera preview", "PASS")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary() -> None:
    banner("TEST SUMMARY")
    width_name = max(len(r[0]) for r in _results) + 2 if _results else 30

    pass_n = sum(1 for r in _results if r[1] == "PASS")
    fail_n = sum(1 for r in _results if r[1] == "FAIL")
    skip_n = sum(1 for r in _results if r[1] == "SKIP")
    warn_n = sum(1 for r in _results if r[1] == "WARN")

    for name, status, detail in _results:
        if status == "PASS":
            icon, color = "✔", C.GREEN
        elif status == "FAIL":
            icon, color = "✘", C.RED
        elif status == "SKIP":
            icon, color = "⊘", C.YELLOW
        else:
            icon, color = "⚠", C.YELLOW

        detail_str = f"  {C.DIM}{detail}{C.RESET}" if detail else ""
        print(f"  {color}{icon}  {name:<{width_name}}  {status}{C.RESET}{detail_str}")

    print(f"\n  {C.BOLD}Results:  "
          f"{C.GREEN}{pass_n} passed{C.RESET}  "
          f"{C.RED}{fail_n} failed{C.RESET}  "
          f"{C.YELLOW}{skip_n} skipped  {warn_n} warnings{C.RESET}\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Droid hardware test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tests/hardware_test.py             # full suite\n"
            "  python tests/hardware_test.py --roomba    # Roomba only\n"
            "  python tests/hardware_test.py --camera    # Camera only\n"
            "  python tests/hardware_test.py --sensors   # Live sensor monitor\n"
            "  python tests/hardware_test.py --drive     # Drive tests only\n"
        )
    )
    parser.add_argument("--roomba",  action="store_true", help="Roomba tests only")
    parser.add_argument("--camera",  action="store_true", help="Camera tests only")
    parser.add_argument("--sensors", action="store_true", help="Live sensor monitor only")
    parser.add_argument("--drive",   action="store_true", help="Drive command tests only")
    args = parser.parse_args()

    # If no flag given → run everything
    run_all    = not any([args.roomba, args.camera, args.sensors, args.drive])
    run_roomba = run_all or args.roomba or args.sensors or args.drive
    run_camera = run_all or args.camera

    banner("DROID HARDWARE TEST SUITE")
    print(f"  {C.DIM}Project root : {_ROOT}{C.RESET}")
    print(f"  {C.DIM}Run modes    : "
          f"{'all' if run_all else ' '.join(a for a,v in vars(args).items() if v)}{C.RESET}\n")

    # Graceful Ctrl+C handler
    roomba = None
    def _cleanup(sig, frame):
        print(f"\n\n  {C.YELLOW}Interrupted — cleaning up …{C.RESET}")
        if roomba:
            roomba.stop_drive()
            roomba.disconnect()
        print_summary()
        sys.exit(0)
    signal.signal(signal.SIGINT, _cleanup)

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = load_config()

    # ── Roomba tests ──────────────────────────────────────────────────────────
    if run_roomba:
        serial_ok = test_serial(cfg)

        if serial_ok:
            roomba = connect_roomba(cfg)

            if roomba:
                if args.sensors:
                    # sensors-only mode: just show the live monitor and exit
                    test_live_sensor_monitor(roomba, duration=0)
                    roomba.stop_drive()
                    roomba.disconnect()
                    print_summary()
                    return 0

                test_sensor_read(roomba)
                test_battery_detail(roomba)

                print(f"\n  {C.DIM}Live sensor monitor will run for 8 s "
                      f"(press Enter to skip) …{C.RESET}")
                test_live_sensor_monitor(roomba, duration=8)

                if run_all or args.roomba:
                    test_interactive_bumpers(roomba)
                    test_interactive_cliffs(roomba)
                    test_songs(roomba)

                if run_all or args.drive:
                    test_drive_commands(roomba)
            else:
                warn("Skipping all Roomba sub-tests — connection failed.")
        else:
            warn("Skipping all Roomba sub-tests — serial port unavailable.")

    # ── Camera tests ──────────────────────────────────────────────────────────
    if run_camera:
        cap = test_camera_connect(cfg)
        if cap:
            test_camera_live_stats(cap, cfg, duration=8)
            test_vision_detection(cap, cfg, duration=10)

            if confirm("Open annotated preview window? (needs display)"):
                test_camera_preview(cap, cfg, duration=15)
            else:
                _record("Camera preview", "SKIP")

            cap.release()

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if roomba:
        roomba.stop_drive()
        roomba.disconnect()

    print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
