import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from rembg import remove
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import io
import logging
import os
import sys

# ----------------------------------------------------------------
# Cấu hình stdout để in UTF-8 an toàn trên Windows
# ----------------------------------------------------------------
sys.stdout.reconfigure(encoding='utf-8')

# ----------------- FastAPI setup -----------------
app = FastAPI()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ======================================================================
# 🔹 MODEL LOADING & PREDICT FUNCTIONS
# ======================================================================
def get_model(model_name: str, num_classes: int):
    if model_name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def load_single_model(model_architecture, model_path, num_classes, device):
    model = get_model(model_architecture, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    classes = checkpoint["classes"]
    model = model.to(device)
    model.eval()
    return model, classes


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(model, classes, img, device):
    try:
        img = remove(img).convert("RGB")
    except Exception as e:
        logging.error(f"Lỗi xử lý ảnh: {e}")
        return "error", {}

    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]
        idx = torch.argmax(probs).item()

    pred_label = classes[idx]
    probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return pred_label, probabilities


# ======================================================================
# 🔹 RULE-BASED LOGIC (HẬU XỬ LÝ)
# ======================================================================
def apply_prediction_rules(pred_label, probabilities):
    prob_good = probabilities.get('good', 0)
    prob_reject = probabilities.get('reject', 0)
    prob_immature = probabilities.get('immature', 0)
    prob_not_df = probabilities.get('not_dragonfruit', 0)
    
    max_prob = max(prob_good, prob_reject, prob_immature, prob_not_df)

    if pred_label in ['good', 'reject', 'immature'] and max_prob < 0.70 and prob_not_df > 0.20:
        return 'not_dragonfruit', "Rule 1: Low confidence -> not_dragonfruit"
        
    if pred_label == 'good':
        if prob_good < 0.98 and prob_reject > 0.01:
            return 'reject', "Rule 2: good -> reject"
            
    return pred_label, "No rule applied"

# ======================================================================
# 🔹 LOAD MODEL
# ======================================================================
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

logging.info(f"Using device: {device}")

MODEL_ARCHITECTURE = "efficientnet_v2_s"
MODEL_PATH = os.path.join(BASE_DIR,"model", "EfficientNetV2", "model", f"{MODEL_ARCHITECTURE}.pth")

model, classes = load_single_model(MODEL_ARCHITECTURE, MODEL_PATH, 4, device)
print(f"[OK] Model loaded from {MODEL_PATH}")

# ======================================================================
# 🔹 API: Predict endpoint
# ======================================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return {"error": "File upload không phải ảnh hợp lệ"}

        pred_label, probs = predict_image(model, classes, image, device)
        if pred_label == "error":
            return {"error": "Không thể xử lý ảnh"}

        final_label, rule_used = apply_prediction_rules(pred_label, probs)

        return {
            "raw_result": pred_label,
            "final_result": final_label,
            "probabilities": probs,
            "rule_applied": rule_used
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}

# ======================================================================
# 🔹 Serve Frontend
# ======================================================================
app.mount("/static", StaticFiles(directory="dist/assets"), name="static")

@app.get("/")
async def root():
    return FileResponse("dist/index.html")
