import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from rembg import remove
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

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
        probs = torch.softmax(out, dim=1)[0]
        idx = torch.argmax(probs).item()
    return classes[idx], {classes[i]: float(probs[i]) for i in range(len(classes))}

# ----------------------------
def evaluate_image(image_path, detector_model, detector_classes, classifier_model, classifier_classes, device):
    inp = Image.open(image_path)
    out = remove(inp).convert("RGB") 
    img = out

    # Step 1: Detector
    det_label, det_probs = predict_image(detector_model, detector_classes, img, device)
    print(f"\n🖼️ Image: {os.path.basename(image_path)}")
    print("✅ Detector:", det_label, det_probs)

    if det_label == "not_dragonfruit":
        print("Not a dragonfruit")
        return "not_dragonfruit"
    else:
        if det_probs['dragonfruit'] > 0.8:
        # Step 2: Quality classifier
            cls_label, cls_probs = predict_image(classifier_model, classifier_classes, img, device)
            final_label = cls_label
            if cls_probs['good'] > 0.93: 
                final_label = "good"
            else: 
                if cls_probs['reject'] > cls_probs['immature']: 
                    final_label = "reject"
                else: 
                    #if cls_probs['immature'] > 0.8: 
                    final_label = "immature"
                    #else:
            #             if cls_probs['reject'] > 0.02:
            #                 final_label = "reject"
            #             else:
            #                 final_label = "immature"
        else:
            print("Not a dragonfruit")
            return "not_dragonfruit"
        print(f"✅ Dragonfruit quality: {final_label}")
        print("Probabilities:", cls_probs)
        return final_label
        # else:
        #     print("Not a dragonfruit")
        #     return "not_dragonfruit"

#----------------------------
def evaluate_folder(root_folder, detector_model, detector_classes, classifier_model, classifier_classes, device):
    y_true, y_pred = [], []
    valid_exts = [".jpg", ".jpeg", ".png"]

    for label in os.listdir(root_folder):
        subdir = os.path.join(root_folder, label)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if not any(fname.lower().endswith(ext) for ext in valid_exts):
                continue
            fpath = os.path.join(subdir, fname)

            pred_final = evaluate_image(
                fpath, detector_model, detector_classes, classifier_model, classifier_classes, device
            )

            # Ground truth = folder name
            gt = label.lower()
            y_true.append(gt)
            y_pred.append(pred_final)

    # Metrics
    labels = ["not_dragonfruit", "good", "immature", "reject"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # AUC-ROC
    try:
        y_true_bin = np.zeros((len(y_true), len(labels)))
        y_pred_bin = np.zeros((len(y_pred), len(labels)))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
            y_true_bin[i, label_to_idx[yt]] = 1
            y_pred_bin[i, label_to_idx[yp]] = 1
        auc = roc_auc_score(y_true_bin, y_pred_bin, average="weighted", multi_class="ovo")
    except Exception:
        auc = None

    print("\n📊 Final Metrics:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)
    print("AUC-ROC:", auc)
    print("Confusion Matrix:\n", cm)

    return y_true, y_pred

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    # Load detector
    detector_model, detector_classes = load_resnet(
        "model/detector_model.pth", num_classes=2, device=device
    )

    # Load classifier
    classifier_model, classifier_classes = load_resnet(
        "model/classifier_model2.pth", num_classes=3, device=device
    )

    test_single_image = ""   # image path
    folder_path = "test_image"

    if os.path.isfile(test_single_image):
        print("\nPredicting single image...")
        final_label = evaluate_image(
            test_single_image, detector_model, detector_classes,
            classifier_model, classifier_classes, device
        )
        print(f"\nFinal Prediction: {final_label}")
    else:
        print("\nEvaluating folder...")
        y_true, y_pred = evaluate_folder(
            folder_path, detector_model, detector_classes,
            classifier_model, classifier_classes, device
        )

        labels = ["not_dragonfruit", "good", "immature", "reject"]

        # Tạo báo cáo chi tiết
        report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

        # Chuyển sang DataFrame để dễ hiển thị
        df_report = pd.DataFrame(report).transpose()

        print("\n📊 Bảng 5.2. Kết quả chi tiết theo từng lớp:")
        print(df_report)

