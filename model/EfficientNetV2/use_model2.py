import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from rembg import remove
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

# (Các hàm get_model, load_single_model, transform, predict_single_image giữ nguyên như cũ)
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

def predict_single_image(model, classes, image_path, device):
    try:
        inp = Image.open(image_path)
        img = remove(inp).convert("RGB") 
    except Exception as e:
        print(f"Lỗi xử lý ảnh {os.path.basename(image_path)}: {e}")
        return "error", {}
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        idx = torch.argmax(probs).item()
    pred_label = classes[idx]
    probabilities = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return pred_label, probabilities

# ==============================================================================
# <<< HÀM MỚI: BỘ LUẬT DỰ ĐOÁN ĐÃ ĐƯỢC TINH CHỈNH >>>
# ==============================================================================
def apply_prediction_rules(pred_label, probabilities):
    """
    Áp dụng bộ luật hậu xử lý theo thứ tự ưu tiên để cải thiện kết quả.
    """
    prob_good = probabilities.get('good', 0)
    prob_reject = probabilities.get('reject', 0)
    prob_immature = probabilities.get('immature', 0)
    prob_not_df = probabilities.get('not_dragonfruit', 0)
    
    # Lấy ra xác suất cao nhất
    max_prob = max(prob_good, prob_reject, prob_immature, prob_not_df)

    # Luật 1: Ưu tiên xử lý các trường hợp không chắc chắn, có thể không phải thanh long
    # Nếu dự đoán là một loại quả nhưng độ tin cậy rất thấp VÀ có khả năng là not_dragonfruit
    if pred_label in ['good', 'reject', 'immature'] and max_prob < 0.70 and prob_not_df > 0.20:
        return 'not_dragonfruit', "Rule 1: Low confidence -> not_dragonfruit"
        
    # Luật 2: Xử lý các trường hợp 'reject' bị nhầm thành 'good' một cách cẩn trọng hơn
    # Chỉ áp dụng khi dự đoán là 'good'
    if pred_label == 'good':
        # Điều kiện này chặt hơn: P(good) phải dưới 0.98 VÀ P(reject) phải đủ lớn
        if prob_good < 0.98 and prob_reject > 0.01:
            return 'reject', "Rule 2: good -> reject"
            
    # Nếu không có luật nào được áp dụng, trả về nhãn gốc
    return pred_label, "No rule applied"
# ==============================================================================

def evaluate_folder(root_folder, model, classes, device):
    y_true, y_pred = [], []
    
    for label in sorted(os.listdir(root_folder)):
        subdir = os.path.join(root_folder, label)
        if not os.path.isdir(subdir): continue
        
        for fname in os.listdir(subdir):
            fpath = os.path.join(subdir, fname)
            raw_label, probs = predict_single_image(model, classes, fpath, device)
            
            if raw_label == "error": continue
                
            final_label, rule_applied = apply_prediction_rules(raw_label, probs)
            
            print_msg = f"Image: {fname:<20} | True: {label:<15} | Raw: {raw_label:<15} | Final: {final_label:<15}"
            if raw_label != final_label:
                print_msg += f" (Rule: {rule_applied})"
            print(print_msg)

            y_true.append(label.lower())
            y_pred.append(final_label)

    # (Phần tính toán và in kết quả giữ nguyên)
    print("\n" + "="*50)
    print("📊 KẾT QUẢ ĐÁNH GIÁ (SAU KHI ÁP DỤNG LUẬT MỚI)")
    print("="*50)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nĐộ chính xác tổng thể (Accuracy): {accuracy:.4f}")
    
    all_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    print("\nMa trận nhầm lẫn (Confusion Matrix):")
    print(pd.DataFrame(cm, index=all_labels, columns=all_labels))
    
    print("\nBáo cáo chi tiết (Classification Report):")
    report = classification_report(y_true, y_pred, labels=all_labels, zero_division=0)
    print(report)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    MODEL_ARCHITECTURE = "efficientnet_v2_s"
    MODEL_PATH = f"EfficientNetV2/model/{MODEL_ARCHITECTURE}.pth"
    TEST_FOLDER = "data/test_image"

    model, classes = load_single_model(MODEL_ARCHITECTURE, MODEL_PATH, 4, device)
    print(f"Tải thành công mô hình từ {MODEL_PATH}")

    evaluate_folder(TEST_FOLDER, model, classes, device)