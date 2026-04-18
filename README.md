<div align="center">

# 🏠 RoomDesigner

### AI-powered room scanner & 3D furniture arranger — runs on your phone, processed on your laptop

[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?style=flat-square&logo=three.js&logoColor=white)](https://threejs.org/)
[![Depth Anything V2](https://img.shields.io/badge/Depth_Anything-V2-ff6b6b?style=flat-square)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-00e5ff?style=flat-square)](https://github.com/ultralytics/ultralytics)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.1-7b61ff?style=flat-square)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-4fffb0?style=flat-square)](LICENSE)

**Scan your room with 8 phone photos → auto-detect walls, furniture & dimensions → design in 3D → ask AI to suggest better layouts**

</div>

---

## ✨ What it does

Take 8 photos of your room by rotating in place. The laptop runs a hybrid AI pipeline that measures your room's real dimensions, detects every piece of furniture, and builds a clean interactive 3D scene you can rearrange — then asks a local LLM for layout suggestions.

**No tape measure. No cloud services. No paid APIs. No LiDAR hardware needed.**

```
     📱                           💻                          📱
 Capture 8     ───────────→    Hybrid AI       ───────────→  Interactive
 photos                         Pipeline                       3D Designer
                                                                   │
                                                                   ▼
                                                              ✨ Ollama AI
                                                              suggests
                                                              better layouts
```

---

## 🎯 Key Features

- **📐 Auto-measurement** — uses your height as an anchor to measure walls, objects, and room dimensions in real centimetres. No manual input beyond your own height.
- **🪑 Furniture auto-detection** — YOLOv8 recognises beds, sofas, chairs, desks, tables, TVs, wardrobes, shelves, plants and more, then places them in 3D at their estimated positions.
- **🎨 Clean editable 3D scene** — drag furniture on the floor, rotate, delete, or add new items from a palette. Real-time top-down floor plan view.
- **🧠 Local AI layout suggestions** — Ollama (llama3.1) suggests new arrangements based on your goals ("maximise comfort flow", "work-focused setup", "minimalist", etc.)
- **💾 Export** — save your layout as PNG floor plan or JSON for later use.
- **🆓 100% free** — no paid APIs, no cloud processing, runs entirely on your local network.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────┐       ┌──────────────────────────────┐
│  📱 PHONE (browser)                    │       │  💻 LAPTOP SERVER (FastAPI)  │
│                                        │       │                              │
│  ① Set height + room type              │       │                              │
│  ② Capture 8 photos at 45° intervals   │────▶ │  ③ Depth Anything V2         │
│                                        │       │     → metric depth maps      │
│                                        │       │  ④ Floor-anchor calibration  │
│                                        │       │     → cm scale per frame     │
│                                        │       │  ⑤ YOLOv8 object detection   │
│                                        │       │     → furniture bounding box │
│                                        │       │  ⑥ Triangulate 3D positions  │
│                                        │◀────  │  ⑦ Build JSON room model     │
│  ⑧ Three.js interactive 3D scene       │       │                              │
│     • Room walls with real dimensions  │       │                              │
│     • Detected furniture as movable    │       │                              │
│       colored boxes                    │       │                              │
│     • Drag / rotate / add / delete     │       │                              │
│                                        │       │                              │
│  ⑨ "Ask AI" button                     │────▶ │  ⑩ Ollama llama3.1           │
│                                        │       │     → suggested new layout   │
│  ⑪ Apply AI suggestions ◀──────────────│       │                              │
└────────────────────────────────────────┘       └──────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with Anaconda recommended
- **Ollama** installed with `llama3.1:latest` pulled — [ollama.com](https://ollama.com/)
- **Phone on the same WiFi** as the laptop

### Setup

```bash
# Clone
git clone https://github.com/allencharly04/room-designer.git
cd room-designer

# Create conda environment
conda create -n room-designer python=3.10 -y
conda activate room-designer

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running (separate terminal)
ollama pull llama3.1:latest
# ollama serve  # only if not already running

# Run the server
python app.py
```

### Open on your phone

```
http://<your-laptop-ip>:8000
```

Find your laptop's IP with `ipconfig` (Windows) or `ifconfig` (Mac/Linux). You want the IPv4 under your WiFi adapter.

> ⚠️ **Windows firewall** may block port 8000 — allow it in Inbound Rules, or run:
> ```bash
> netsh advfirewall firewall add rule name="RoomDesigner" protocol=TCP dir=in localport=8000 action=allow
> ```

---

## 📸 Using It

| Step | What happens |
|---|---|
| **① Setup** | Slide to your height (used for depth calibration). Pick room type. |
| **② Capture** | Stand in the centre of your room. Tap shutter, rotate ~45°, tap again. 8 photos = one full circle. |
| **③ Processing** | The laptop runs depth AI + YOLO detection + triangulation. ~30 sec on GPU, ~2 min on CPU. |
| **④ Design** | Your room appears as walls + floor grid with labeled furniture boxes. Drag to rearrange, tap to select & rotate. |
| **⑤ AI help** | Tap ✨ AI, pick a goal, Ollama suggests new positions with reasoning. One tap to apply. |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Vanilla JS + Three.js r128 (no build step, runs in phone browser) |
| **Backend** | FastAPI + Uvicorn |
| **Depth estimation** | [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) |
| **Object detection** | [YOLOv8n](https://github.com/ultralytics/ultralytics) (~6MB) |
| **LLM** | Ollama + llama3.1:latest (local inference) |
| **3D rendering** | Three.js (WebGL) |

---

## 🧪 How the Auto-Measurement Works

Monocular depth estimation gives only **relative** distances. To get real centimetres, we use your body as the measurement anchor:

```python
user_height_cm = 175                      # user input
camera_height_cm = 175 * 0.62             # ≈ 109cm, typical phone-hold position

# In each photo, sample floor pixels (bottom-centre strip)
d_floor = median(depth_map[bottom_rows, centre_cols])

# Geometry: at pixel row 90% of image height, floor is exactly camera_height below
z_floor_cm = camera_height_cm * fy / (floor_row - image_centre_y)

# Solve for scale
metric_scale = z_floor_cm / d_floor

# Every other pixel's depth is now in real cm
z_world_cm = depth_value * metric_scale
```

This gives **±10–20cm accuracy** at room scale — good enough for practical furniture arrangement.

---

## 🎨 Screenshots

> Add screenshots here after capturing them from the live app:
> - `docs/home.png` — setup screen with height/room selection
> - `docs/capture.png` — camera view with progress ring
> - `docs/designer.png` — 3D room with detected furniture
> - `docs/floorplan.png` — top-down floor plan
> - `docs/ai-modal.png` — AI layout suggestions

---

## 🔮 Future Ideas

- [ ] Save/load multiple room layouts per user
- [ ] Door / window detection
- [ ] Wall colour / texture swap
- [ ] Real 3D furniture meshes (GLB) instead of boxes
- [ ] Multi-room apartment support
- [ ] Feng shui / Vastu analysis via Ollama
- [ ] Export to Sketchup / Blender format
- [ ] iOS DeviceOrientation permission handling for gyro-assisted stitching

---

## 📦 Project Structure

```
room-designer/
├── app.py                    # FastAPI server + AI pipeline
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
├── static/
│   └── index.html            # Phone UI (capture → process → design)
└── scans/                    # Session data (gitignored)
```

---

## 📜 License

MIT — use it, fork it, learn from it.

---

## 🙏 Credits

- [Depth Anything V2](https://depth-anything-v2.github.io/) — monocular depth estimation
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [Ollama](https://ollama.com/) — local LLM runtime
- [Three.js](https://threejs.org/) — 3D rendering

---

<div align="center">

**Built by [@allencharly04](https://github.com/allencharly04)** · Part of the [Charnel Ally](https://www.tiktok.com/@charnelally) AI agent series

⭐ Star this repo if you find it useful!

</div>
