# ⬡ Droid Firmware — B2EMO Edition

Python firmware for a Roomba 650-based droid with a B2EMO personality,
threaded vision, and a local Ollama AI brain.

---

## Hardware Requirements

| Component | Spec |
|---|---|
| Platform | iRobot Roomba 650 |
| Serial | USB-UART adapter → Mini-DIN 7-pin connector |
| Logic Level | **3.3V** (NOT 5V — Roomba serial is sensitive) |
| Wiring | TXD→Pin3, RXD→Pin4, RTS→Pin5 (Device Detect), GND→Pin6 |
| Camera | USB webcam or CSI camera, index 0 |
| Brain | Local [Ollama](https://ollama.com) API (`localhost:11434`) |

> ⚠️ **Critical Hardware Note:** Always use a 3.3V logic USB-UART adapter
> (CH340, CP2102, or FTDI in 3.3V mode). Feeding 5V into the Roomba's
> Mini-DIN Pin 3 (RX) risks permanent damage to the serial receiver circuit.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama

```bash
# Install: https://ollama.com
ollama pull llama3.2
ollama serve          # Keep this running in a separate terminal
```

### 3. Configure hardware

Edit `config.yaml`:

```yaml
system:
  port: "COM7"        # Windows: COMx  |  Linux: /dev/ttyUSB0  |  macOS: /dev/cu.usbserial-*
  baud: 115200
```

### 4. Run the firmware

```bash
python main.py
```

The firmware will run a **Pre-Flight Diagnostic** first, then start the
main control loop. Type messages in the terminal to chat with B2EMO.
Press `Ctrl+C` to exit gracefully.

---

## Directory Structure

```
/Droid2
│   main.py               # Orchestrator — threading, drive state machine
│   config.yaml           # All settings: hardware, personality, features
│   requirements.txt
│
├── /modules
│   ├── roomba_io.py      # Roomba 650 Open Interface (UART commands)
│   ├── vision_engine.py  # Threaded OpenCV — HOG persons + obstacle detection
│   ├── droid_brain.py    # Ollama AI integration + sensor context injection
│   └── personality.py    # Sensor→speech mapping, cooldowns, B2EMO reactions
│
├── /utils
│   ├── logger.py         # ANSI-colored terminal logger
│   └── diag_manager.py   # Pre-flight checks → /diagnostics/*.diag files
│
└── /diagnostics          # Runtime-generated diagnostic reports and session logs
```

---

## Changing the Personality

**No Python code changes required.** Edit `config.yaml`:

```yaml
personality:
  system_prompt: |
    You are R2-D2 ...   # ← Change this block entirely
```

Three presets are included in `config.yaml` as commented-out blocks:
- **B2EMO** (Andor) — stuttering, anxious, loyal *(default)*
- **R2-D2** — beeps/whistles in brackets with translations
- **Imperial Sentry Droid ID-9** — cold, clipped, militaristic

---

## Feature Flags

Toggle subsystems in `config.yaml` without touching code:

```yaml
features:
  enable_drive:  true   # false = sensor-only / static demo
  enable_vision: true   # false = no camera required
  enable_chat:   true   # false = no Ollama required
  enable_songs:  true   # false = silent Roomba
```

---

## Diagnostics

Every run generates two files in `/diagnostics`:

| File | Contents |
|---|---|
| `preflight_YYYYMMDD_HHMMSS.diag` | Pre-flight check results (serial, camera, Ollama) |
| `session_YYYYMMDD_HHMMSS.log` | Full timestamped session log |

Example `.diag` file:
```
============================================================
  DROID FIRMWARE — PRE-FLIGHT DIAGNOSTIC REPORT
============================================================
  Generated : 2026-05-01T14:32:11.204
  Host      : DROID-PC

SUMMARY
────────────────────────────────────────
  Roomba Serial  : ✔
  Camera (CV2)   : ✔
  Ollama API     : ✔
  Overall        : PASS ✔
```

---

## Architecture: Threading Model

```
┌──────────────────────────────────────────────────────────────┐
│  main thread        │ VisionCapture │ RoombaSensor │ BrainWorker │
│  (drive loop 20Hz)  │ (daemon 15Hz) │ (daemon 10Hz)│ (on-demand) │
│  personality logic  │ HOG + MOG2    │ UART poll    │ Ollama API  │
└──────────────────────────────────────────────────────────────┘
```

- **Drive loop** never blocks on AI or vision.
- **Vision** and **sensor** threads update shared state via thread-safe getters.
- **Brain** runs in a fire-and-forget worker thread; results polled each tick.

---

## Roomba 650 Open Interface Reference

| Opcode | Name | Notes |
|---|---|---|
| 128 | START | Begin OI session |
| 131 | SAFE | Full control, cliff/bump safety active |
| 132 | FULL | Full control, **safety disabled** |
| 137 | DRIVE | velocity(mm/s) + radius(mm) |
| 140 | SONG | Define song (up to 16 notes, 5 slots) |
| 141 | PLAY | Play defined song |
| 149 | QUERY_LIST | Request multiple sensor packets at once |

---

## Upgrade Path

| Feature | Current | Upgrade |
|---|---|---|
| Person detection | OpenCV HOG (CPU) | YOLOv8 via `ultralytics` |
| Obstacle detection | MOG2 background subtraction | Depth camera (Intel RealSense) |
| AI model | Ollama (local) | Any OpenAI-compatible API |
| Voice output | Terminal text | `pyttsx3` TTS (offline) |

---

*"I-I'll try to remember this mission. If my memory coils hold."* — B2EMO
