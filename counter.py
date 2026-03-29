"""
SmartSpace - People Counter (Computer Vision)

Install:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Run (recommended via Flask app):
  python app.py

Or run standalone to see the video window:
  python counter.py

Notes:
- Uses OpenCV for camera frames.
- Uses Ultralytics YOLO with built-in tracking IDs.
- Counts entry/exit based on crossing a vertical "doorway" line:
    entry: left -> right
    exit : right -> left
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TrackState:
    """Per-track memory needed to count line crossings once per crossing."""

    last_side: Optional[str] = None  # "L" or "R"
    cooldown: bool = False  # after counting, must move away from line to re-arm
    last_seen_ts: float = 0.0


def _side_of_line(cx: float, line_x: int) -> str:
    return "L" if cx < line_x else "R"


def _box_centroid_xyxy(xyxy: np.ndarray) -> Tuple[float, float]:
    x1, y1, x2, y2 = xyxy.tolist()
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def run_counter(
    shared_state: dict,
    lock,
    room_name: str = "Room 1210",
    camera_source: Union[int, str] = 0,
    model_name: str = "yolov8n.pt",
    line_x_ratio: float = 0.5,
    show_video: bool = True,
) -> None:
    """
    Main loop: capture frames, detect+track persons, count line crossings, update shared state.

    - shared_state: dict-like with keys occupancy, entries, exits, room_name
    - lock: threading.Lock (or compatible) guarding shared_state
    """

    # Load a lightweight model. Ultralytics will auto-download weights on first run.
    model = YOLO(model_name)

    # camera_source can be:
    # - local webcam index (e.g. 0)
    # - phone stream URL (e.g. http://192.168.1.10:8080/video)
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera source: {camera_source}. "
            "Use CAMERA_SOURCE=0 for webcam, or set CAMERA_SOURCE to your phone stream URL."
        )

    tracks: Dict[int, TrackState] = {}
    active_ids: Dict[int, float] = {}

    # Tuning knobs
    away_margin_px = 60  # must move this far from the line to re-arm after a count
    stale_track_seconds = 2.0  # forget tracks not seen recently (helps keep memory clean)

    # Initialize shared state
    with lock:
        shared_state["room_name"] = room_name
        shared_state.setdefault("occupancy", 0)
        shared_state.setdefault("entries", 0)
        shared_state.setdefault("exits", 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue

        h, w = frame.shape[:2]
        line_x = int(w * line_x_ratio)

        # YOLO track call returns track IDs in boxes.id (when tracking succeeds).
        # Restrict to "person" class only: class 0 in COCO.
        results = model.track(
            frame,
            persist=True,
            classes=[0],
            verbose=False,
            conf=0.35,
            iou=0.5,
        )

        now = time.time()
        current_ids = set()

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            # xyxy: (N,4), id: (N,) (may be None for some frames)
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id
            ids_np = ids.cpu().numpy().astype(int) if ids is not None else None

            for i in range(xyxy.shape[0]):
                if ids_np is None:
                    # Without stable IDs we can't reliably prevent double counting.
                    # Skip counting, but still allow display.
                    track_id = None
                else:
                    track_id = int(ids_np[i])

                cx, cy = _box_centroid_xyxy(xyxy[i])
                current_side = _side_of_line(cx, line_x)

                # Draw box + ID for visibility
                x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                label = f"ID {track_id}" if track_id is not None else "person"
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )
                cv2.circle(frame, (int(cx), int(cy)), 3, (255, 255, 255), -1)

                if track_id is None:
                    continue

                current_ids.add(track_id)
                active_ids[track_id] = now

                st = tracks.get(track_id)
                if st is None:
                    st = TrackState(last_side=current_side, cooldown=False, last_seen_ts=now)
                    tracks[track_id] = st
                    continue

                st.last_seen_ts = now

                # Cooldown logic: after we count a crossing, require the person to move
                # away from the line to avoid double counting due to jitter/oscillation.
                if st.cooldown:
                    if abs(cx - line_x) > away_margin_px:
                        st.cooldown = False
                    st.last_side = current_side
                    continue

                # Line crossing logic:
                # - Determine if the tracked centroid changed sides relative to the line.
                # - Count entry/exit only on a side-change event.
                if st.last_side != current_side:
                    if st.last_side == "L" and current_side == "R":
                        with lock:
                            shared_state["entries"] += 1
                            shared_state["occupancy"] += 1
                        st.cooldown = True
                    elif st.last_side == "R" and current_side == "L":
                        with lock:
                            shared_state["exits"] += 1
                            shared_state["occupancy"] = max(0, shared_state["occupancy"] - 1)
                        st.cooldown = True

                st.last_side = current_side

        # Forget stale tracks so IDs don't grow forever
        for tid, last_seen in list(active_ids.items()):
            if now - last_seen > stale_track_seconds:
                active_ids.pop(tid, None)
                tracks.pop(tid, None)

        # Draw doorway line + overlay counts
        cv2.line(frame, (line_x, 0), (line_x, h), (255, 0, 0), 3)
        with lock:
            occ = int(shared_state.get("occupancy", 0))
            ent = int(shared_state.get("entries", 0))
            ex = int(shared_state.get("exits", 0))
        overlay = f"Occupancy: {occ}  Entries: {ent}  Exits: {ex}"
        cv2.putText(
            frame,
            overlay,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if occ == 0 else (0, 0, 255),
            2,
        )

        if show_video:
            try:
                cv2.imshow("SmartSpace - People Counter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            except cv2.error:
                # On some systems (notably macOS), OpenCV GUI calls can fail when
                # invoked from non-main threads. If that happens, fall back to
                # headless mode instead of crashing the counter loop.
                show_video = False
        else:
            # If running headless, keep loop from pegging CPU too hard.
            time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Standalone run mode (local testing). This does not start Flask.
    import threading

    shared = {"room_name": "Room 1210", "occupancy": 0, "entries": 0, "exits": 0}
    state_lock = threading.Lock()
    run_counter(shared, state_lock, camera_source=0, show_video=True)
