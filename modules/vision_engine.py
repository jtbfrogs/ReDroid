"""
modules/vision_engine.py
────────────────────────
Threaded computer vision engine for the Droid firmware.

Architecture:
  A single background daemon thread continuously captures frames from the
  camera, runs detection pipelines, and stores the latest DetectionResult
  in a thread-safe slot. The main loop calls get_latest() at any time without
  blocking or experiencing camera frame-lag.

Detection Pipelines:
  1. Person Detection  — OpenCV HOG Descriptor with pre-trained SVM.
                         Reliable CPU-only detector, ~5-10 FPS on a modern PC.
  2. Obstacle Detection— Background subtraction (MOG2) + contour analysis
                         on the lower third of the frame.
                         Flags large, stable foreground objects as obstacles.

Upgrade path:
  Replace _detect_persons() with a YOLO/ONNX model call for GPU-accelerated
  detection. The rest of the pipeline is model-agnostic.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from utils.logger import get_logger

log = get_logger("vision")


# ── Detection Result ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    person_detected:   bool  = False
    obstacle_detected: bool  = False
    person_count:      int   = 0
    obstacle_count:    int   = 0
    person_boxes:      list  = field(default_factory=list)   # List[Tuple[x,y,w,h]]
    obstacle_boxes:    list  = field(default_factory=list)
    frame_width:       int   = 0
    frame_height:      int   = 0
    fps:               float = 0.0
    timestamp:         float = 0.0
    # Annotated frame for optional display (None if show_preview=False)
    annotated_frame: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    def to_context_string(self) -> str:
        """Human-readable summary for use as AI sensor context."""
        parts = []
        if self.person_detected:
            parts.append(f"{self.person_count} person(s) visible on camera")
        if self.obstacle_detected:
            parts.append(f"{self.obstacle_count} obstacle(s) detected ahead")
        if not parts:
            parts.append("visual field clear — no persons or obstacles")
        return "; ".join(parts) + f" [camera {self.frame_width}×{self.frame_height} @ {self.fps:.1f}fps]"


# ── Vision Engine ─────────────────────────────────────────────────────────────

class VisionEngine:
    """
    Threaded OpenCV vision pipeline.

    Args:
        config: Full parsed config dict (uses 'features' and 'system' sub-dicts).
        show_preview: If True, renders an annotated OpenCV window (for debug).
    """

    # HOG detector constants
    _HOG_WIN_STRIDE  = (8, 8)
    _HOG_PADDING     = (4, 4)
    _HOG_SCALE       = 1.05
    _HOG_HIT_THRESH  = 0.0    # Lower = more detections, higher = fewer false positives

    # Obstacle detection constants
    _OBSTACLE_MIN_AREA    = 4000  # px² — smaller contours ignored as noise
    _OBSTACLE_ZONE_FRAC   = 0.50  # Use bottom 50% of frame for obstacle detection
    _BG_HISTORY           = 200   # MOG2 history length
    _BG_VAR_THRESHOLD     = 40    # MOG2 variance threshold
    _MORPH_KERNEL_SIZE    = (5, 5)

    def __init__(self, config: dict, show_preview: bool = False):
        feat = config.get("features", {})
        self._cam_index  = feat.get("camera_index", 0)
        self._res        = tuple(feat.get("vision_resolution", [640, 480]))
        self._fps_cap    = feat.get("vision_fps_cap", 15)
        self._enabled    = feat.get("enable_vision", True)
        self._show_preview = show_preview

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock   = threading.Lock()
        self._latest: DetectionResult = DetectionResult()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # HOG person detector (CPU, no GPU dependency)
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        log.debug("HOG person detector initialized.")

        # Background subtractor for obstacle detection
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._BG_HISTORY,
            varThreshold=self._BG_VAR_THRESHOLD,
            detectShadows=False,
        )
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self._MORPH_KERNEL_SIZE
        )

        # FPS tracking
        self._fps_counter = 0
        self._fps_timer   = time.monotonic()
        self._current_fps = 0.0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Open the camera and launch the background capture thread."""
        if not self._enabled:
            log.info("Vision engine disabled in config — not starting.")
            return False

        log.info(f"Opening camera index {self._cam_index} at {self._res[0]}×{self._res[1]} …")
        self._cap = cv2.VideoCapture(self._cam_index)
        if not self._cap.isOpened():
            log.error(f"Failed to open camera index {self._cam_index}.")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._res[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._res[1])
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Camera opened — actual resolution: {actual_w}×{actual_h}")

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="VisionCapture",
            daemon=True,
        )
        self._thread.start()
        self._running = True
        log.info("Vision capture thread started.")
        return True

    def stop(self) -> None:
        """Signal the capture thread to stop and release camera resources."""
        log.info("Stopping vision engine …")
        self._stop_event.set()
        self._running = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        if self._show_preview:
            cv2.destroyAllWindows()

        log.info("Vision engine stopped.")

    # ── Capture Loop (runs in daemon thread) ──────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Continuously read frames, run detectors, and store DetectionResult.
        Respects fps_cap by sleeping between frames.
        """
        log.debug("Capture loop running …")
        frame_delay = 1.0 / max(1, self._fps_cap)

        while not self._stop_event.is_set():
            t_frame_start = time.monotonic()

            if self._cap is None or not self._cap.isOpened():
                log.warning("Camera lost — attempting reopen in 2s …")
                time.sleep(2.0)
                self._cap = cv2.VideoCapture(self._cam_index)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                log.debug("Empty frame received — skipping.")
                time.sleep(0.05)
                continue

            # Update FPS counter
            self._fps_counter += 1
            elapsed = time.monotonic() - self._fps_timer
            if elapsed >= 1.0:
                self._current_fps = self._fps_counter / elapsed
                self._fps_counter = 0
                self._fps_timer   = time.monotonic()

            # Run detection pipelines
            result = self._process_frame(frame)
            result.fps       = round(self._current_fps, 1)
            result.timestamp = time.monotonic()
            result.frame_width, result.frame_height = frame.shape[1], frame.shape[0]

            # Annotate frame if preview is enabled
            if self._show_preview:
                annotated = self._annotate_frame(frame.copy(), result)
                result.annotated_frame = annotated
                cv2.imshow("Droid Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Vision preview window closed by user.")
                    self._stop_event.set()
                    break

            # Atomic update of latest result
            with self._lock:
                self._latest = result

            # FPS cap: sleep remainder of frame budget
            elapsed_frame = time.monotonic() - t_frame_start
            sleep_time = frame_delay - elapsed_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

        log.debug("Capture loop terminated.")

    # ── Detection Pipelines ───────────────────────────────────────────────────

    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Run all detectors on a single frame and return a DetectionResult."""
        result = DetectionResult()

        # --- Person Detection (HOG) ---
        person_boxes = self._detect_persons(frame)
        result.person_detected = len(person_boxes) > 0
        result.person_count    = len(person_boxes)
        result.person_boxes    = person_boxes

        # --- Obstacle Detection (background subtraction) ---
        obstacle_boxes = self._detect_obstacles(frame)
        result.obstacle_detected = len(obstacle_boxes) > 0
        result.obstacle_count    = len(obstacle_boxes)
        result.obstacle_boxes    = obstacle_boxes

        return result

    def _detect_persons(self, frame: np.ndarray) -> list:
        """
        Detect persons using HOG + pre-trained SVM.

        Resizes frame to a smaller intermediate size for speed, then
        scales bounding boxes back to original dimensions.

        Returns:
            List of (x, y, w, h) bounding box tuples.
        """
        h, w = frame.shape[:2]
        # Resize to max 320px wide for speed — HOG is O(n) in pixel count
        scale = min(1.0, 320.0 / w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        rects, weights = self._hog.detectMultiScale(
            gray,
            winStride=self._HOG_WIN_STRIDE,
            padding=self._HOG_PADDING,
            scale=self._HOG_SCALE,
            hitThreshold=self._HOG_HIT_THRESH,
        )

        if len(rects) == 0:
            return []

        # Scale boxes back to original resolution
        inv = 1.0 / scale
        boxes = []
        for (rx, ry, rw, rh) in rects:
            boxes.append((
                int(rx * inv), int(ry * inv),
                int(rw * inv), int(rh * inv),
            ))

        # Non-maximum suppression to remove overlapping boxes
        boxes = self._nms(boxes, overlap_thresh=0.65)
        return boxes

    def _detect_obstacles(self, frame: np.ndarray) -> list:
        """
        Detect static and slow-moving obstacles using background subtraction
        applied to the lower `_OBSTACLE_ZONE_FRAC` of the frame.

        Returns:
            List of (x, y, w, h) bounding box tuples.
        """
        h, w = frame.shape[:2]
        zone_y = int(h * (1.0 - self._OBSTACLE_ZONE_FRAC))
        roi    = frame[zone_y:h, 0:w]

        # Apply background subtraction
        fg_mask = self._bg_subtractor.apply(roi)

        # Morphological close to fill holes, then erode noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE,  self._morph_kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_ERODE,  self._morph_kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, self._morph_kernel, iterations=2)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._OBSTACLE_MIN_AREA:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            # Translate ROI-relative Y back to full-frame coordinates
            boxes.append((rx, ry + zone_y, rw, rh))

        return boxes

    # ── Annotation ────────────────────────────────────────────────────────────

    def _annotate_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and HUD overlay on a copy of the frame."""
        # Person boxes — green
        for (x, y, w, h) in result.person_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            cv2.putText(frame, "PERSON", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

        # Obstacle boxes — orange
        for (x, y, w, h) in result.obstacle_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)
            cv2.putText(frame, "OBSTACLE", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        # HUD: FPS + status bar at top
        h_frame = frame.shape[0]
        hud = f"FPS: {result.fps:.1f}  |  Persons: {result.person_count}  |  Obstacles: {result.obstacle_count}"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (20, 20, 20), -1)
        cv2.putText(frame, hud, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Obstacle zone boundary line
        zone_y = int(h_frame * (1.0 - self._OBSTACLE_ZONE_FRAC))
        cv2.line(frame, (0, zone_y), (frame.shape[1], zone_y), (80, 80, 200), 1)

        return frame

    # ── NMS Helper ────────────────────────────────────────────────────────────

    @staticmethod
    def _nms(boxes: list, overlap_thresh: float = 0.65) -> list:
        """
        Simple non-maximum suppression for bounding boxes.
        Removes boxes that overlap by more than `overlap_thresh` IoU.
        """
        if len(boxes) == 0:
            return []

        boxes_arr = np.array(boxes, dtype=float)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 0] + boxes_arr[:, 2]
        y2 = boxes_arr[:, 1] + boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = areas.argsort()[::-1]
        keep  = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter_area = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-6)
            inds = np.where(iou <= overlap_thresh)[0]
            order = order[inds + 1]

        return [boxes[k] for k in keep]

    # ── Public Interface ──────────────────────────────────────────────────────

    def get_latest(self) -> DetectionResult:
        """
        Return the most recent DetectionResult (thread-safe, non-blocking).
        Returns an empty result if the engine hasn't processed any frames yet.
        """
        with self._lock:
            return self._latest

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()
