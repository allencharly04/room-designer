"""
Room Designer v2 — Smart Layout Planner
Pipeline: photos → depth estimation → furniture detection (YOLO) →
          clean 3D room model → Ollama layout suggestions
"""

import os, io, json, base64, math, logging, shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import httpx

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Room Designer v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

BASE_DIR  = Path(__file__).parent
SCANS_DIR = BASE_DIR / "scans"
SCANS_DIR.mkdir(exist_ok=True)

# ── Ollama config ─────────────────────────────
OLLAMA_URL   = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")


# ══════════════════════════════════════════════════════════
# ML MODEL LOADERS (lazy)
# ══════════════════════════════════════════════════════════
_depth_model = None
_yolo_model  = None

def get_depth_model():
    global _depth_model
    if _depth_model is not None:
        return _depth_model
    try:
        logger.info("Loading Depth Anything V2...")
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        _depth_model = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=device
        )
        logger.info(f"Depth model loaded ({'GPU' if device==0 else 'CPU'})")
    except Exception as e:
        logger.warning(f"Depth model failed: {e}")
        _depth_model = "mock"
    return _depth_model


def get_yolo_model():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        logger.info("Loading YOLOv8...")
        from ultralytics import YOLO
        _yolo_model = YOLO("yolov8n.pt")  # smallest, fastest variant
        logger.info("YOLO loaded")
    except Exception as e:
        logger.warning(f"YOLO unavailable: {e}")
        _yolo_model = "mock"
    return _yolo_model


# ══════════════════════════════════════════════════════════
# YOLO CLASS → FURNITURE MAPPING
# COCO dataset class IDs that represent room furniture/objects
# ══════════════════════════════════════════════════════════
FURNITURE_CLASSES = {
    56: {"name": "chair",     "def_w": 60,  "def_d": 60,  "def_h": 90,  "color": 0xff6b6b},
    57: {"name": "sofa",      "def_w": 200, "def_d": 90,  "def_h": 80,  "color": 0xff9d4d},
    58: {"name": "plant",     "def_w": 40,  "def_d": 40,  "def_h": 80,  "color": 0x4fffb0},
    59: {"name": "bed",       "def_w": 200, "def_d": 200, "def_h": 55,  "color": 0x7b61ff},
    60: {"name": "table",     "def_w": 120, "def_d": 80,  "def_h": 75,  "color": 0xffd166},
    62: {"name": "tv",        "def_w": 120, "def_d": 15,  "def_h": 70,  "color": 0x00e5ff},
    63: {"name": "laptop",    "def_w": 35,  "def_d": 25,  "def_h": 3,   "color": 0xc8daea},
    64: {"name": "mouse",     "def_w": 10,  "def_d": 6,   "def_h": 3,   "color": 0xc8daea},
    66: {"name": "keyboard",  "def_w": 45,  "def_d": 15,  "def_h": 3,   "color": 0xc8daea},
    67: {"name": "phone",     "def_w": 15,  "def_d": 8,   "def_h": 1,   "color": 0xc8daea},
    72: {"name": "fridge",    "def_w": 70,  "def_d": 70,  "def_h": 180, "color": 0x23d18b},
    73: {"name": "book",      "def_w": 20,  "def_d": 5,   "def_h": 28,  "color": 0xf7c948},
    74: {"name": "clock",     "def_w": 30,  "def_d": 5,   "def_h": 30,  "color": 0xc8daea},
    75: {"name": "vase",      "def_w": 20,  "def_d": 20,  "def_h": 35,  "color": 0x4d9eff},
}

# Detected items that may not be furniture but often appear
OBJECT_CLASSES = {
    0:  "person",
    15: "cat",
    16: "dog",
    39: "bottle",
    41: "cup",
    46: "banana",
    71: "toaster",
}


# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════
class Session:
    def __init__(self, sid: str):
        self.sid         = sid
        self.dir         = SCANS_DIR / sid
        self.dir.mkdir(exist_ok=True)
        self.images_dir  = self.dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.photos      : list[str] = []
        self.user_height : float     = 175.0
        self.room_type   : str       = "bedroom"
        self.status      : str       = "capturing"
        self.stage       : str       = "Ready"
        self.progress    : int       = 0
        self.room_model  : dict      = {}
        self.error       : str       = ""

    def to_dict(self):
        return {
            "sid":        self.sid,
            "photos":     len(self.photos),
            "status":     self.status,
            "stage":      self.stage,
            "progress":   self.progress,
            "room_model": self.room_model,
            "error":      self.error,
        }

_sessions: dict[str, Session] = {}

def new_sid() -> str:
    return datetime.now().strftime("scan_%Y%m%d_%H%M%S")

def get_session(sid: str) -> Session:
    if sid not in _sessions:
        _sessions[sid] = Session(sid)
    return _sessions[sid]


# ══════════════════════════════════════════════════════════
# CORE PIPELINE
# ══════════════════════════════════════════════════════════
def calibrate_depth_scale(depth_map: np.ndarray, camera_height_cm: float, fov_deg: float) -> float:
    """
    Use bottom-centre pixels (floor) + camera height to find metric scale.
    Returns: scale factor such that z_cm = depth_value * scale.
    """
    h, w  = depth_map.shape
    fov_v = math.radians(fov_deg) * (h / w)
    fy    = (h / 2) / math.tan(fov_v / 2)

    floor_rows = np.arange(int(h * 0.85), h)
    floor_cols = np.arange(int(w * 0.35), int(w * 0.65))
    fc, fr     = np.meshgrid(floor_cols, floor_rows)
    floor_d    = depth_map[fr, fc].ravel()
    floor_d    = floor_d[(floor_d > 0.08) & (floor_d < 0.94)]

    if len(floor_d) > 10:
        d_floor   = float(np.median(floor_d))
        floor_row = float(h * 0.90)
        z_floor   = camera_height_cm * fy / max(floor_row - h / 2, 1.0)
        return float(np.clip(z_floor / max(d_floor, 0.01), 80.0, 1800.0))
    return 600.0


def run_depth(pil_img: Image.Image) -> np.ndarray:
    """Get normalised depth map (0=far, 1=near)."""
    model = get_depth_model()
    if model == "mock":
        w, h = pil_img.size
        y, _ = np.mgrid[0:h, 0:w]
        return np.clip(1.0 - y/h * 0.6 + np.random.normal(0, 0.02, (h, w)), 0, 1).astype(np.float32)
    result = model(pil_img)
    d = np.array(result["depth"], dtype=np.float32)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)


def detect_furniture(pil_img: Image.Image) -> list:
    """Run YOLO, return list of detections with bbox + class."""
    model = get_yolo_model()
    if model == "mock":
        return []
    results = model(pil_img, verbose=False, conf=0.35)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls  = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            if cls in FURNITURE_CLASSES:
                info = FURNITURE_CLASSES[cls]
                dets.append({
                    "class_id":   cls,
                    "name":       info["name"],
                    "conf":       round(conf, 2),
                    "bbox":       [round(v, 1) for v in xyxy],
                    "def_w":      info["def_w"],
                    "def_d":      info["def_d"],
                    "def_h":      info["def_h"],
                    "color":      info["color"],
                })
    return dets


def estimate_object_position(
    detection: dict,
    depth_map: np.ndarray,
    scale: float,
    fov_deg: float,
    camera_angle_deg: float,
    img_w: int,
    img_h: int,
) -> dict:
    """
    From a 2D bounding box + depth map + camera angle,
    estimate the object's 3D world position.
    """
    x1, y1, x2, y2 = detection["bbox"]
    cx_px = (x1 + x2) / 2
    cy_px = (y1 + y2) / 2

    # Sample depth at centre of bbox (with small window)
    s       = 5
    x_lo    = max(0, int(cx_px) - s)
    x_hi    = min(img_w - 1, int(cx_px) + s)
    y_lo    = max(0, int(cy_px) - s)
    y_hi    = min(img_h - 1, int(cy_px) + s)
    patch   = depth_map[y_lo:y_hi, x_lo:x_hi]
    patch   = patch[patch > 0.05]
    if len(patch) < 2:
        d_val = 0.4
    else:
        d_val = float(np.median(patch))

    z_cm  = d_val * scale

    # Horizontal offset from image centre → world x
    fov_h = math.radians(fov_deg)
    fx    = (img_w / 2) / math.tan(fov_h / 2)
    x_rel = (cx_px - img_w / 2) / fx * z_cm

    # Rotate by camera angle (yaw)
    a = math.radians(camera_angle_deg)
    world_x = x_rel * math.cos(a) + z_cm * math.sin(a)
    world_z = -x_rel * math.sin(a) + z_cm * math.cos(a)

    return {
        "x": round(float(world_x), 1),
        "z": round(float(world_z), 1),
        "dist_cm":  round(z_cm, 1),
    }


def build_room_model(session: Session) -> dict:
    """
    Main pipeline: process all photos, measure room, detect furniture.
    Each photo is tagged with its camera angle (0°, 45°, 90°, ...).
    """
    photos = sorted(session.images_dir.glob("*.jpg"))
    if len(photos) < 3:
        raise ValueError(f"Need at least 3 photos, got {len(photos)}")

    session.stage    = "Loading models..."
    session.progress = 5
    _ = get_depth_model()
    _ = get_yolo_model()

    camera_height_cm = session.user_height * 0.62
    fov_deg          = 70.0  # OnePlus main camera
    # Assume photos are taken at evenly-spaced angles around the room
    angle_step       = 360.0 / len(photos)

    all_detections   = []
    max_distances    = []

    session.stage = "Analysing photos..."
    for i, photo_path in enumerate(photos):
        session.progress = 10 + int((i / len(photos)) * 70)
        session.stage    = f"Analysing photo {i+1}/{len(photos)}"

        pil_img = Image.open(photo_path).convert("RGB")
        img_w, img_h = pil_img.size

        # Resize for faster processing
        if img_w > 640:
            ratio  = 640 / img_w
            new_wh = (640, int(img_h * ratio))
            pil_img = pil_img.resize(new_wh)
            img_w, img_h = new_wh

        # 1. Depth
        depth = run_depth(pil_img)
        scale = calibrate_depth_scale(depth, camera_height_cm, fov_deg)

        # 2. Max distance in this photo (for room size estimate)
        mid_d = np.median(depth[depth > 0.08])
        max_distances.append(float(mid_d) * scale)

        # 3. YOLO detection
        camera_angle = i * angle_step
        dets = detect_furniture(pil_img)
        for det in dets:
            pos = estimate_object_position(
                det, depth, scale, fov_deg, camera_angle, img_w, img_h
            )
            det["position"] = pos
            det["from_photo"] = i
            all_detections.append(det)

        logger.info(f"Photo {i+1}: scale={scale:.0f} mid_dist={mid_d*scale:.0f}cm · {len(dets)} objects")

    # 4. Deduplicate detections (same object seen in multiple photos)
    session.stage    = "Merging detections..."
    session.progress = 85
    merged = deduplicate_detections(all_detections)

    # 5. Estimate room dimensions
    # Use the detections + depth estimates to infer room bounds
    xs = [d["position"]["x"] for d in merged]
    zs = [d["position"]["z"] for d in merged]
    if len(xs) >= 2:
        room_w = max(abs(min(xs)), abs(max(xs))) * 2 * 1.2
        room_d = max(abs(min(zs)), abs(max(zs))) * 2 * 1.2
    else:
        # Fallback: use max depth estimates
        room_w = np.median(max_distances) * 1.3 if max_distances else 400
        room_d = np.median(max_distances) * 1.3 if max_distances else 400

    # Cap to realistic range (2-10m)
    room_w = float(np.clip(room_w, 200, 1000))
    room_d = float(np.clip(room_d, 200, 1000))
    room_h = float(camera_height_cm * 2.3)   # typical ceiling ~2.4m if camera 1.1m

    session.stage    = "Building 3D model..."
    session.progress = 95

    model = {
        "room": {
            "width_cm":  round(room_w, 0),
            "depth_cm":  round(room_d, 0),
            "height_cm": round(room_h, 0),
            "width_m":   round(room_w / 100, 2),
            "depth_m":   round(room_d / 100, 2),
            "height_m":  round(room_h / 100, 2),
        },
        "furniture": merged,
        "meta": {
            "photos":       len(photos),
            "total_raw":    len(all_detections),
            "room_type":    session.room_type,
            "camera_height_cm": round(camera_height_cm, 1),
        }
    }
    return model


def deduplicate_detections(dets: list, dist_threshold_cm: float = 80.0) -> list:
    """Merge detections of same class within threshold distance."""
    merged = []
    for det in dets:
        matched = False
        for m in merged:
            if m["name"] == det["name"]:
                dx = m["position"]["x"] - det["position"]["x"]
                dz = m["position"]["z"] - det["position"]["z"]
                if math.sqrt(dx * dx + dz * dz) < dist_threshold_cm:
                    # Same object — use higher confidence
                    if det["conf"] > m["conf"]:
                        m.update({k: det[k] for k in ["conf", "position"]})
                    matched = True
                    break
        if not matched:
            merged.append({
                "name":     det["name"],
                "class_id": det["class_id"],
                "conf":     det["conf"],
                "def_w":    det["def_w"],
                "def_d":    det["def_d"],
                "def_h":    det["def_h"],
                "color":    det["color"],
                "position": det["position"],
                "rotation_deg": 0,
            })
    # Assign unique IDs
    for i, m in enumerate(merged):
        m["id"] = f"obj_{i}"
    return merged


# ══════════════════════════════════════════════════════════
# OLLAMA INTEGRATION
# ══════════════════════════════════════════════════════════
async def ollama_suggest_layout(room_model: dict, goal: str) -> dict:
    """Send room to Ollama, get layout suggestions."""
    prompt = f"""You are an interior design AI. Given this room layout in JSON, suggest an improved arrangement.

GOAL: {goal}

CURRENT ROOM:
{json.dumps(room_model, indent=2)}

Respond with ONLY a JSON object matching this schema (no markdown, no explanation):
{{
  "reasoning": "brief explanation of your changes",
  "new_positions": [
    {{"id": "obj_0", "x": 100, "z": -50, "rotation_deg": 90, "reason": "why moved"}}
  ]
}}

Rules:
- Keep x within ±{round(room_model['room']['width_cm']/2)} cm
- Keep z within ±{round(room_model['room']['depth_cm']/2)} cm
- Don't overlap furniture
- Sofa should face TV if both present
- Bed should have headboard against a wall
- Leave walking space (80cm min between objects)
"""

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
            )
            r.raise_for_status()
            data     = r.json()
            response = data.get("response", "{}")
            parsed   = json.loads(response)
            return parsed
        except httpx.HTTPError as e:
            raise HTTPException(500, f"Ollama error: {e}")
        except json.JSONDecodeError:
            return {"reasoning": "Could not parse response", "new_positions": []}


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def root():
    p = Path("static/index.html")
    return HTMLResponse(content=p.read_text(encoding="utf-8")) if p.exists() \
        else HTMLResponse("<h1>Room Designer v2</h1>")


@app.get("/api/health")
async def health():
    # Check Ollama
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "status":        "ok",
        "depth_model":   "loaded" if _depth_model and _depth_model != "mock" else "not loaded",
        "yolo_model":    "loaded" if _yolo_model  and _yolo_model  != "mock" else "not loaded",
        "ollama":        "ok" if ollama_ok else "unavailable",
        "ollama_url":    OLLAMA_URL,
        "ollama_model":  OLLAMA_MODEL,
    }


@app.post("/api/session/new")
async def session_new(body: dict):
    sid = new_sid()
    s   = get_session(sid)
    s.user_height = float(body.get("height_cm", 175))
    s.room_type   = body.get("room_type", "bedroom")
    logger.info(f"New session {sid} · height {s.user_height}cm · {s.room_type}")
    return {"sid": sid}


@app.get("/api/session/{sid}")
async def session_get(sid: str):
    if sid not in _sessions:
        raise HTTPException(404, "Session not found")
    return _sessions[sid].to_dict()


@app.post("/api/session/{sid}/photo")
async def upload_photo(sid: str, file: UploadFile = File(...)):
    s   = get_session(sid)
    idx = len(s.photos)
    dest = s.images_dir / f"photo_{idx:03d}.jpg"
    data = await file.read()
    dest.write_bytes(data)
    s.photos.append(dest.name)
    logger.info(f"[{sid}] Photo {idx+1} ({len(data)//1024}KB)")
    return {"idx": idx, "total": len(s.photos)}


@app.post("/api/session/{sid}/process")
async def process(sid: str):
    """Run pipeline synchronously (small enough to not need threading)."""
    if sid not in _sessions:
        raise HTTPException(404, "Session not found")
    s = _sessions[sid]
    if len(s.photos) < 3:
        raise HTTPException(400, f"Need at least 3 photos, got {len(s.photos)}")

    s.status = "processing"
    try:
        model = build_room_model(s)
        s.room_model = model
        s.status     = "done"
        s.stage      = f"Done! Found {len(model['furniture'])} objects"
        s.progress   = 100
        logger.info(f"[{sid}] Pipeline complete: {len(model['furniture'])} objects detected")
        return {"status": "done", "room_model": model}
    except Exception as e:
        logger.exception("Pipeline error")
        s.status = "error"
        s.error  = str(e)
        s.stage  = "Error"
        raise HTTPException(500, str(e))


@app.post("/api/session/{sid}/ask_ai")
async def ask_ai(sid: str, body: dict):
    """Ask Ollama to suggest a new layout."""
    if sid not in _sessions:
        raise HTTPException(404, "Session not found")
    s    = _sessions[sid]
    goal = body.get("goal", "Arrange the room for maximum comfort and good flow")

    # Use current room_model (may have been edited client-side)
    current_model = body.get("room_model", s.room_model)
    if not current_model:
        raise HTTPException(400, "No room model available")

    result = await ollama_suggest_layout(current_model, goal)
    return result


@app.delete("/api/session/{sid}")
async def session_delete(sid: str):
    if sid in _sessions:
        s = _sessions.pop(sid)
        if s.dir.exists():
            shutil.rmtree(s.dir)
    return {"deleted": sid}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Ollama: {OLLAMA_URL} model={OLLAMA_MODEL}")
    logger.info(f"Scans: {SCANS_DIR}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False,
                ws_ping_interval=60, ws_ping_timeout=120)
