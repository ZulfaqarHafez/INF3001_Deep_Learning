# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from PIL import Image
import io, json

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CKPT_PATH = MODELS_DIR / "helmet_classifier.pth"
MAP_PATH  = MODELS_DIR / "class_mapping.json"

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# Class mapping
# ----------------------------
with open(MAP_PATH, "r", encoding="utf-8") as f:
    class_mapping = json.load(f)

# ensure deterministic order 0..N-1
labels = [class_mapping[str(i)] for i in range(len(class_mapping))]
num_classes = len(labels)


# ----------------------------
# Build model (two head variants)
# ----------------------------
def build_resnet18(num_classes: int, head: str, pretrained: bool = True):
    # handle both modern & older torchvision
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=pretrained)

    if head == "linear":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif head == "dropout-linear":
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(model.fc.in_features, num_classes),
        )
    else:
        raise ValueError(f"Unknown head type: {head}")
    return model


# ----------------------------
# Load checkpoint smartly
# ----------------------------
state_dict = torch.load(CKPT_PATH, map_location="cpu")

# Decide which head the checkpoint used
if any(k.startswith("fc.1.") for k in state_dict.keys()):
    head_type = "dropout-linear"  # checkpoint expects Dropout + Linear
elif "fc.weight" in state_dict and "fc.bias" in state_dict:
    head_type = "linear"          # checkpoint expects plain Linear
else:
    # Default fallback: try linear first
    head_type = "linear"

model = build_resnet18(num_classes=num_classes, head=head_type, pretrained=True)

# Try strict load; if it fails due to only-head differences, attempt the other head once.
try:
    model.load_state_dict(state_dict, strict=True)
except RuntimeError as e:
    # If we guessed wrong, try the alternative head
    alt_head = "dropout-linear" if head_type == "linear" else "linear"
    model = build_resnet18(num_classes=num_classes, head=alt_head, pretrained=True)
    model.load_state_dict(state_dict, strict=True)

model.eval()


# ----------------------------
# Preprocessing
# ----------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ----------------------------
# Inference endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        x = transform(img).unsqueeze(0)  # [1,3,224,224]

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

        results = [
            {"label": labels[i], "probability": round(float(p), 4)}
            for i, p in enumerate(probs)
        ]
        # sort high→low for convenience
        results.sort(key=lambda d: d["probability"], reverse=True)

        return JSONResponse({"predictions": results})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
