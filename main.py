# main.py — FastAPI server (works with: uvicorn main:app --host 0.0.0.0 --port 10000)
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageIn(BaseModel):
    image: str
    filename: Optional[str] = None

def _to_bgr(image_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _water_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # blue–cyan/teal bands to find likely water
    lower1 = np.array([85, 20, 40]);  upper1 = np.array([120, 255, 255])
    lower2 = np.array([70, 10, 30]);  upper2 = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def _reflection_score(bgr: np.ndarray, water_mask: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    vals = gray[water_mask > 0]
    if vals.size == 0: return 0.0
    thresh = np.percentile(vals, 95)
    return float(np.mean(vals >= thresh))  # 0..1

def _turbidity_score(bgr: np.ndarray, water_mask: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    vals = lap[water_mask > 0]
    if vals.size == 0: return 0.0
    var = float(np.var(vals))
    var = max(0.0, min(var, 2000.0))
    return 1.0 - (var / 2000.0)  # 0=crisp, 1=hazy

def _green_ratio_in_water(bgr: np.ndarray, water_mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_g = np.array([40, 25, 25]); upper_g = np.array([85, 255, 255])
    green = cv2.inRange(hsv, lower_g, upper_g)
    green_in_water = green[water_mask > 0]
    water_pix = int(np.sum(water_mask > 0))
    if water_pix == 0: return 0.0
    return float(np.sum(green_in_water > 0)) / float(water_pix)

def _palette_hint(bgr: np.ndarray, water_mask: np.ndarray) -> str:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ws = hsv[water_mask > 0]
    if ws.size == 0: return "unknown"
    h = ws[:, 0].astype(np.float32)
    teal_ratio = float(np.mean((h >= 75) & (h <= 95)))
    if teal_ratio > 0.45: return "pebble-teal"
    blue_ratio = float(np.mean((h > 95) & (h <= 120)))
    if blue_ratio > 0.45: return "blue"
    return "mixed"

@app.post("/analyze-pool-photo")
def analyze_pool_photo(inp: ImageIn):
    bgr = _to_bgr(inp.image)
    if bgr is None:
        return {"status": "unknown", "confidence": 0.0}

    # speed resize
    h, w = bgr.shape[:2]
    target_w = 900
    if w > target_w:
        scale = target_w / w
        bgr = cv2.resize(bgr, (target_w, int(h * scale)))

    mask = _water_mask_hsv(bgr)
    water_pix = int(np.sum(mask > 0))
    water_conf = min(1.0, water_pix / (0.08 * bgr.shape[0] * bgr.shape[1] + 1e-6))

    refl = _reflection_score(bgr, mask)
    turb = _turbidity_score(bgr, mask)
    green = _green_ratio_in_water(bgr, mask)
    palette = _palette_hint(bgr, mask)

    # finish-aware decision
    status = "clean"
    conf = 0.0
    if water_conf >= 0.5:
        dirty_signal = max(turb, green)
        clean_blockers = (refl >= 0.7) or (palette == "pebble-teal" and turb < 0.45)
        if dirty_signal >= 0.55 and not clean_blockers:
            status = "dirty"
            conf = float(0.5*dirty_signal + 0.3*(1.0 - refl) + 0.2*water_conf)
        else:
            status = "clean"
            conf = float(max(1.0 - dirty_signal, refl))
    else:
        status = "clean"
        conf = 0.5

    conf = max(0.0, min(conf, 1.0))
    return {
        "status": status,
        "confidence": round(conf, 3),
        "water_confidence": round(float(water_conf), 3),
        "turbidity": round(float(turb), 3),
        "green_ratio": round(float(green), 3),
        "reflection_score": round(float(refl), 3),
        "palette": palette,
        "filename": inp.filename or ""
    }

@app.get("/health")
def health():
    return {"status": "ok"}
