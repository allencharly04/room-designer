# 🏠 Room Designer v2 — Hybrid AI Layout Planner

Scan your room with 8 photos → get accurate dimensions + furniture detection → arrange in 3D → ask Ollama for layout suggestions.

## Architecture

```
Phone (browser)                        Laptop (server)
───────────────                        ───────────────
1. Setup: height + room type
2. Take 8 photos (one every 45°)  →    Upload via WiFi
                                       ↓
                                  3. Depth Anything V2 → distances
                                  4. YOLOv8 → detect furniture
                                  5. Triangulate 3D positions
                                  6. Build clean room model (JSON)
7. Receive model  ←                    
8. Clean 3D scene with boxes
9. Drag to rearrange
10. "Ask AI"  →                        11. Ollama suggests new layout
12. Apply ←                             (llama3.1:latest)
```

## Setup

### 1. Folder structure
Put this in `D:\allen\Agents\12. room designer v2\` (or wherever you want).
Keep `static\index.html` inside the `static` subfolder.

### 2. Install dependencies
```bash
conda activate room-scanner
cd "D:\allen\Agents\12. room designer v2"
pip install -r requirements.txt
```

This installs:
- FastAPI + uvicorn (server)
- Depth Anything V2 (distance estimation)
- **YOLOv8n** (~6MB, downloads on first use)
- Ollama client (httpx)

### 3. Start Ollama (separate terminal)
```bash
ollama serve
# make sure you have llama3.1:latest:
ollama pull llama3.1:latest
```

### 4. Run the server
```bash
python app.py
```

### 5. Open on phone
```
http://<laptop-IP>:8000
```

## Usage

1. **Home screen** — set your height (used for depth calibration) and pick room type. Check the server status bar — all 4 should say OK/loaded.

2. **Capture** — Stand in the centre of your room. Tap shutter, rotate ~45° (about 1/8 of a full circle), tap again. Repeat 8 times. The progress ring fills up. Hit Done.

3. **Processing** — Watch the pipeline run. Takes ~30 seconds on GPU, ~2 min on CPU.

4. **Designer** — Your room appears as walls + floor grid. Detected furniture appears as colored boxes labeled with their names. 
   - **Drag** any box to move it on the floor
   - **Tap** a box to select → use rotation slider or delete
   - **Add Furniture** drawer at bottom for more items
   - **Floor Plan** tab shows top-down view
   - Export as PNG or JSON

5. **Ask AI** — Tap the ✨ AI button in the header. Pick a goal (Comfort, Light, Work, etc.) or type your own. Ollama returns new positions + reasoning. Tap Apply to rearrange automatically.

## Tips

- **Good lighting matters** — turn on all lights for better YOLO detection
- **Slow rotation** — hold steady, don't blur
- **Photo 1 = your "front"** — all positions are relative to where you started
- **3-8 photos works** — 8 is ideal but 3 is enough for basic scanning

## Troubleshooting

- **YOLO says "not loaded"** → run `pip install ultralytics` manually
- **Ollama says "unavailable"** → start `ollama serve` in another terminal
- **Room dimensions way off** → take more photos, ensure phone held at chest level
- **No furniture detected** → YOLO only detects COCO classes (bed/chair/sofa/desk/table/TV/etc). Specific items like "lamp" or "mirror" aren't in COCO.
