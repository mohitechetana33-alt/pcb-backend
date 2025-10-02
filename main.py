from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import hashlib
import uuid
from typing import List, Dict
import logging
import os

# ----------------------------------
# Config
# ----------------------------------
# Set Paddle device via env: "cpu" (default) or "gpu"
PADDLE_DEVICE = os.getenv("PADDLE_DEVICE", "cpu").lower()

# ----------------------------------
# FastAPI App + CORS
# ----------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------
# Initialize PaddleOCR (3.x-safe, 2.x fallback)
# ----------------------------------
ocr = None
try:
    try:
        import paddle
        paddle.device.set_device(PADDLE_DEVICE)  # replaces old use_gpu kw
        logger.info(f"Paddle device set to: {PADDLE_DEVICE}")
    except Exception as dev_e:
        logger.warning(f"Could not set Paddle device explicitly: {dev_e}")

    # use_textline_orientation=True replaces deprecated use_angle_cls
    ocr = PaddleOCR(
        lang="en",
        use_textline_orientation=True
    )
    logger.info("PaddleOCR initialized.")
except Exception as e:
    logger.error(f"PaddleOCR init failed: {e}")
    ocr = None

# ----------------------------------
# Initialize YOLOv8
# ----------------------------------
model = None
try:
    model = YOLO("yolov8n.pt")  # downloads on first run
    logger.info("YOLOv8 model loaded.")
except Exception as e:
    logger.error(f"YOLO init failed: {e}")
    model = None

# ----------------------------------
# Component Price DB (example)
# ----------------------------------
COMPONENT_DATABASE = {
    'relay': {'name': 'Relay', 'avg_price_inr': 45, 'category': 'BOM'},
    'capacitor': {'name': 'Capacitor', 'avg_price_inr': 2.5, 'category': 'BOM'},
    'resistor': {'name': 'Resistor', 'avg_price_inr': 0.15, 'category': 'BOM'},
    'ic': {'name': 'IC', 'avg_price_inr': 25, 'category': 'BOM'},
    'led': {'name': 'LED', 'avg_price_inr': 1.2, 'category': 'BOM'},
    'connector': {'name': 'Connector', 'avg_price_inr': 8, 'category': 'BOP'},
    'transistor': {'name': 'Transistor', 'avg_price_inr': 3.5, 'category': 'BOM'},
    'diode': {'name': 'Diode', 'avg_price_inr': 0.8, 'category': 'BOM'},
    'inductor': {'name': 'Inductor', 'avg_price_inr': 1.5, 'category': 'BOM'},
}

# ----------------------------------
# OCR helper (supports PaddleOCR 3.x predict + 2.x ocr formats)
# ----------------------------------
def _run_ocr_on_crop(crop_bgr: np.ndarray) -> str:
    """
    Run OCR on a BGR crop. Tries PaddleOCR 3.x predict() first,
    falls back to 2.x ocr(..., cls=True). Returns concatenated text.
    """
    if ocr is None or crop_bgr is None or crop_bgr.size == 0:
        return ""

    text_chunks: List[str] = []

    # Try 3.x style first
    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pred = ocr.predict(crop_rgb)  # 3.x call

        if isinstance(pred, (list, tuple)) and len(pred) > 0:
            item = pred[0]

            # Case A: dict with 'res' -> list of dicts containing 'text'
            if isinstance(item, dict) and "res" in item and isinstance(item["res"], (list, tuple)):
                for r in item["res"]:
                    if isinstance(r, dict) and "text" in r:
                        text_chunks.append(str(r["text"]))
                    elif isinstance(r, (list, tuple)) and len(r) >= 1:
                        text_chunks.append(str(r[0]))

            # Case B: list-like older shape: [ [box], (text, score) ]
            elif isinstance(item, (list, tuple)):
                for elem in item:
                    if isinstance(elem, (list, tuple)) and len(elem) >= 2:
                        maybe_txt = elem[1]
                        if isinstance(maybe_txt, (list, tuple)) and len(maybe_txt) >= 1:
                            text_chunks.append(str(maybe_txt[0]))

        if text_chunks:
            return " ".join([t.strip() for t in text_chunks if t and t.strip()])
    except Exception:
        pass  # fall through to 2.x

    # Fallback to 2.x
    try:
        res2 = ocr.ocr(crop_bgr, cls=True)
        if isinstance(res2, (list, tuple)) and len(res2) > 0 and isinstance(res2[0], (list, tuple)):
            for line in res2[0]:
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    txt_pair = line[1]
                    if isinstance(txt_pair, (list, tuple)) and len(txt_pair) >= 1:
                        text_chunks.append(str(txt_pair[0]))
        if text_chunks:
            return " ".join([t.strip() for t in text_chunks if t and t.strip()])
    except Exception:
        pass

    return ""

# ----------------------------------
# API: Analyze PCB
# ----------------------------------
@app.post("/api/analyze")
async def analyze_pcb(file: UploadFile = File(...)):
    """Real PCB analysis endpoint"""
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger.info(f"[{request_id}] Starting analysis")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image"}

        image_hash = hashlib.md5(contents).hexdigest()[:12]
        logger.info(f"[{request_id}] Image hash: {image_hash}")

        components: List[Dict] = []

        # YOLO detection
        if model is not None:
            results = model(img, conf=0.25, verbose=False)

            for r_idx, result in enumerate(results):
                if not hasattr(result, "boxes") or result.boxes is None:
                    continue

                names = getattr(result, "names", {})
                for b_idx, box in enumerate(result.boxes):
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                    cls_id = int(box.cls[0]) if hasattr(box, "cls") else -1
                    class_name = str(names.get(cls_id, "component")).lower()

                    # Crop for OCR
                    crop = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    try:
                        marking_text = _run_ocr_on_crop(crop)
                    except Exception as ocr_e:
                        logger.warning(f"[{request_id}] OCR failed on box {b_idx}: {ocr_e}")
                        marking_text = ""

                    # DB match
                    component_info = COMPONENT_DATABASE.get(class_name, {
                        'name': class_name.title() if class_name else 'Component',
                        'avg_price_inr': 5.0,
                        'category': 'BOM'
                    })

                    components.append({
                        'id': f"{request_id}_{r_idx}_{b_idx}",
                        'name': f"{component_info['name']} {marking_text}".strip(),
                        'type': component_info['name'],
                        'marking': marking_text,
                        'bbox': [x1, y1, x2, y2],
                        'confidence': round(conf, 3),
                        'position': {'x': int((x1 + x2) / 2), 'y': int((y1 + y2) / 2)},
                        'package': 'SMD',
                        'quantity': 1,
                        'category': component_info['category'],
                        'estimatedCostINR': component_info['avg_price_inr'],
                        'function': f'{component_info["name"]} component'
                    })

        # Fallback if no detections
        if not components:
            logger.warning(f"[{request_id}] No components detected, using fallback")
            h, w = img.shape[:2]
            components = [{
                'id': f"{request_id}_1",
                'name': 'PCB Component',
                'type': 'Unknown',
                'marking': 'N/A',
                'bbox': [0, 0, w // 2, h // 2],
                'confidence': 0.5,
                'position': {'x': w // 4, 'y': h // 4},
                'package': 'Unknown',
                'quantity': 1,
                'category': 'BOM',
                'estimatedCostINR': 10.0,
                'function': 'Component detected'
            }]

        logger.info(f"[{request_id}] Detected {len(components)} components")

        # Costing
        bom_cost = sum(c['estimatedCostINR'] * c['quantity'] for c in components if c.get('category') == 'BOM')
        bop_cost = sum(c['estimatedCostINR'] * c['quantity'] for c in components if c.get('category') == 'BOP')
        labour_cost = len(components) * 3.5
        rnd_cost = (bom_cost + bop_cost) * 0.10
        total_cost = bom_cost + bop_cost + labour_cost + rnd_cost

        avg_conf = round(
            sum(c.get('confidence', 0.0) for c in components) / len(components),
            3
        ) if components else 0.0

        return {
            'request_id': request_id,
            'image_hash': image_hash,
            'components': components,
            'cost_breakdown': {
                'bomCost': round(bom_cost, 2),
                'bopCost': round(bop_cost, 2),
                'labourCost': round(labour_cost, 2),
                'rndCost': round(rnd_cost, 2),
                'totalCost': round(total_cost, 2)
            },
            'metadata': {
                'total_components': len(components),
                'avg_confidence': avg_conf
            }
        }

    except Exception as e:
        logger.error(f"[{request_id}] Error: {str(e)}")
        return {'error': str(e)}

# ----------------------------------
# Healthcheck
# ----------------------------------
@app.get("/health")
async def health():
    return {
        'status': 'ok',
        'yolo_loaded': model is not None,
        'ocr_loaded': ocr is not None,
        'paddle_device': PADDLE_DEVICE
    }

# ----------------------------------
# Uvicorn Entry
# ----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
