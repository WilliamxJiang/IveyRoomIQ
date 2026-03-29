"""
AI SmartSpace - Flask API + Dashboard

Install:
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Run:
  python app.py

Then open:
  http://127.0.0.1:5000/

API:
  GET /api/status
    -> { room_name, occupancy, entries, exits, status }
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from counter import run_counter


app = Flask(__name__)

# Shared in-memory state (no DB). Guard with a lock since CV thread updates it.
STATE_LOCK = threading.Lock()
ROOM_ID = os.environ.get("ROOM_ID", "room-1210")
ROOM_NAME = os.environ.get("SMARTSPACE_ROOM_NAME", "Room 1210")

STATE = {
    "room_name": ROOM_NAME,
    "occupancy": 0,
    "entries": 0,
    "exits": 0,
}

# Multi-room occupancy state for the row-based dashboard view.
# The active camera feed updates ROOM_ID in real time, while other rooms stay available
# for manual event simulation endpoints during MVP demos.
ROOMS = {
    "room-1226": {"room_name": "1226", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1224": {"room_name": "1224", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1220": {"room_name": "1220", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1218": {"room_name": "1218", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1212": {"room_name": "1212", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1210": {"room_name": "1210", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1206": {"room_name": "1206", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1204": {"room_name": "1204", "occupancy": 0, "entries": 0, "exits": 0},
    "room-1200": {"room_name": "1200", "occupancy": 0, "entries": 0, "exits": 0},
}
ROOMS.setdefault(ROOM_ID, {"room_name": ROOM_NAME, "occupancy": 0, "entries": 0, "exits": 0})

FLOORPLAN_PATH = Path(
    os.environ.get(
        "FLOORPLAN_PATH",
        "/Users/wj/.cursor/projects/Users-wj-Downloads-Lit-Prototype/assets/image-53432cb6-b982-43f1-8f17-5b33e8325a80.png",
    )
)
LOGO_PATH = Path(
    os.environ.get(
        "IVEY_LOGO_PATH",
        "/Users/wj/.cursor/projects/Users-wj-Downloads-Lit-Prototype/assets/image-3b937889-6685-450c-aad4-221eee411f3a.png",
    )
)
PROFILE_PHOTO_PATH = Path(
    os.environ.get(
        "PROFILE_PHOTO_PATH",
        "/Users/wj/.cursor/projects/Users-wj-Downloads-Lit-Prototype/assets/image-d6cdaa88-d786-40a6-b21a-5fb0e0eefc41.png",
    )
)


def _start_counter_thread() -> None:
    """
    Starts the CV loop in a background thread.

    Tip: set SHOW_VIDEO=0 to run without the OpenCV window.
    """

    camera_source_raw = os.environ.get("CAMERA_SOURCE", os.environ.get("CAMERA_INDEX", "0"))
    camera_source = int(camera_source_raw) if str(camera_source_raw).isdigit() else camera_source_raw
    # IMPORTANT (macOS/OpenCV): cv2.imshow() is unreliable from background threads.
    # Since we run the counter in a daemon thread, default to headless mode unless
    # the user explicitly enables video.
    show_video = os.environ.get("SHOW_VIDEO", "0") not in ("0", "false", "False")
    model_name = os.environ.get("YOLO_MODEL", "yolov8n.pt")
    line_x_ratio = float(os.environ.get("LINE_X_RATIO", "0.5"))

    t = threading.Thread(
        target=run_counter,
        kwargs=dict(
            shared_state=STATE,
            lock=STATE_LOCK,
            room_name=STATE["room_name"],
            camera_source=camera_source,
            model_name=model_name,
            line_x_ratio=line_x_ratio,
            show_video=show_video,
        ),
        daemon=True,
        name="smartspace-counter",
    )
    t.start()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    with STATE_LOCK:
        room_name = str(STATE.get("room_name", "AI SmartSpace"))
        occupancy = int(STATE.get("occupancy", 0))
        entries = int(STATE.get("entries", 0))
        exits = int(STATE.get("exits", 0))

    return jsonify(
        {
            "room_name": room_name,
            "occupancy": occupancy,
            "entries": entries,
            "exits": exits,
            "status": "Available" if occupancy == 0 else "Occupied",
        }
    )


@app.get("/api/rooms")
def api_rooms():
    with STATE_LOCK:
        # Keep ROOMS in sync with the live single-room counter state.
        ROOMS[ROOM_ID]["room_name"] = str(STATE.get("room_name", ROOM_NAME))
        ROOMS[ROOM_ID]["occupancy"] = int(STATE.get("occupancy", 0))
        ROOMS[ROOM_ID]["entries"] = int(STATE.get("entries", 0))
        ROOMS[ROOM_ID]["exits"] = int(STATE.get("exits", 0))

        rooms = []
        total_occupancy = 0
        total_entries = 0
        total_exits = 0
        for room_id, room in ROOMS.items():
            occupancy = int(room.get("occupancy", 0))
            entries = int(room.get("entries", 0))
            exits = int(room.get("exits", 0))
            total_occupancy += occupancy
            total_entries += entries
            total_exits += exits
            rooms.append(
                {
                    "room_id": room_id,
                    "room_name": str(room.get("room_name", room_id)),
                    "occupancy": occupancy,
                    "entries": entries,
                    "exits": exits,
                    "status": "Available" if occupancy == 0 else "Occupied",
                }
            )

    return jsonify(
        {
            "building_name": "Ivey Row 1220-1210-1200",
            "occupancy": total_occupancy,
            "entries": total_entries,
            "exits": total_exits,
            "rooms": rooms,
        }
    )


@app.post("/api/rooms/<room_id>/event")
def api_room_event(room_id: str):
    """
    Update a room counter with crossing events.
    JSON payload:
      {"direction": "entry"} or {"direction": "exit"}
    """
    payload = request.get_json(silent=True) or {}
    direction = str(payload.get("direction", "")).strip().lower()
    if direction not in {"entry", "exit"}:
        return jsonify({"error": "direction must be 'entry' or 'exit'"}), 400

    with STATE_LOCK:
        if room_id not in ROOMS:
            return jsonify({"error": f"Unknown room_id: {room_id}"}), 404
        room = ROOMS[room_id]
        if direction == "entry":
            room["entries"] += 1
            room["occupancy"] += 1
        else:
            room["exits"] += 1
            room["occupancy"] = max(0, room["occupancy"] - 1)

        updated = {
            "room_id": room_id,
            "room_name": room["room_name"],
            "occupancy": int(room["occupancy"]),
            "entries": int(room["entries"]),
            "exits": int(room["exits"]),
            "status": "Available" if int(room["occupancy"]) == 0 else "Occupied",
        }
    return jsonify(updated)


@app.get("/floorplan.png")
def floorplan_image():
    if not FLOORPLAN_PATH.exists():
        abort(404)
    return send_file(str(FLOORPLAN_PATH), mimetype="image/png")


@app.get("/logo.png")
def logo_image():
    if not LOGO_PATH.exists():
        abort(404)
    return send_file(str(LOGO_PATH), mimetype="image/png")


@app.get("/profile.png")
def profile_photo():
    if not PROFILE_PHOTO_PATH.exists():
        abort(404)
    return send_file(str(PROFILE_PHOTO_PATH), mimetype="image/png")


if __name__ == "__main__":
    _start_counter_thread()
    # Use threaded server so the dashboard + API stay responsive.
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)

