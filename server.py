import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import logging

# ----------------- FastAPI setup -----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cho phép React/Frontend bất kỳ gọi API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ----------------- Utils -----------------
def load_resnet(model_path, num_classes, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    classes = checkpoint["classes"]
    model = model.to(device)
    model.eval()
    return model, classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(model, classes, img, device):
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]
        idx = torch.argmax(probs).item()
    return classes[idx], {classes[i]: float(probs[i]) for i in range(len(classes))}

# ----------------- Load models -----------------
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

logging.info(f"Using device: {device}")

detector_model, detector_classes = load_resnet(
    "detector_model.pth", num_classes=2, device=device
)
classifier_model, classifier_classes = load_resnet(
    "classifier_model.pth", num_classes=3, device=device
)

# ----------------- API predict -----------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except UnidentifiedImageError:
            return {"error": "File upload không phải ảnh hợp lệ"}

        # Step 1: Detector
        det_label, det_probs = predict_image(detector_model, detector_classes, image, device)
        logging.info(f"Detector: {det_label}, Probs: {det_probs}")

        # Nếu không phải thanh long hoặc xác suất thanh long thấp
        if det_label == "not_dragonfruit" or det_probs.get("dragonfruit", 0) < 0.95:
            return {
                "result": "not_dragonfruit",
                "stage": "detector",
                "probabilities": det_probs
            }

        # Step 2: Classifier
        cls_label, cls_probs = predict_image(classifier_model, classifier_classes, image, device)
        logging.info(f"Classifier: {cls_label}, Probs: {cls_probs}")

        # Rule-based refinement
        if cls_probs.get("good", 0) > 0.97:
            final_label = "good"
        else:
            if cls_probs.get("reject", 0) > cls_probs.get("immature", 0):
                final_label = "reject"
            else:
                if cls_probs.get("immature", 0) > 0.8:
                    final_label = "immature"
                else:
                    if cls_probs.get("reject", 0) > 0.02:
                        final_label = "reject"
                    else:
                        final_label = "immature"

        return {
            "result": final_label,
            "stage": "classifier",
            "detector_probabilities": det_probs,
            "classifier_probabilities": cls_probs
        }

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {"error": str(e)}
    
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount thư mục static (JS, CSS, images sau khi build)
app.mount("/static", StaticFiles(directory="build/static"), name="static")

# Route trả về index.html (giao diện React)
@app.get("/")
async def serve_frontend():
    index_path = os.path.join("build", "index.html")
    return FileResponse(index_path)
