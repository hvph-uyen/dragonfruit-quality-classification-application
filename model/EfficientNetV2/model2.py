import os
import copy
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

def get_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_v2_s(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else: # Default to resnet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_corrects, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)
    return running_loss / total, running_corrects / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, running_corrects, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Val", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)
    return running_loss / total, running_corrects / total

def train_model(model_name, data_dir, model_save_path,
                           num_classes, batch_size=32,
                           stage1_epochs=5, stage2_epochs=15,
                           lr_stage1=1e-3, lr_stage2=1e-4, device="cpu"):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    num_workers = 0 if platform.system() == "Darwin" else 4

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = get_model(model_name, num_classes, pretrained=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_wts = None
    
    # Stage 1
    for param in model.parameters(): param.requires_grad = False
    if model_name == "efficientnet_v2_s":
        for param in model.classifier.parameters(): param.requires_grad = True
    else:
        for param in model.fc.parameters(): param.requires_grad = True
    optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage1)
    
    print(f"--- GIAI ĐOẠN 1: HUẤN LUYỆN LỚP CLASSIFIER ---")
    for epoch in range(1, stage1_epochs + 1):
        # ... (logic epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{stage1_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Stage 2
    for param in model.parameters(): param.requires_grad = True
    optimizer_stage2 = optim.SGD(model.parameters(), lr=lr_stage2, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage2, mode="max", factor=0.5, patience=2)
    
    print(f"--- GIAI ĐOẠN 2: TINH CHỈNH TOÀN BỘ MÔ HÌNH ---")
    for epoch in range(1, stage2_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(f"Epoch {epoch}/{stage2_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    if best_model_wts:
        model.load_state_dict(best_model_wts)
    torch.save({'model_state_dict': model.state_dict(), 'classes': train_ds.classes}, model_save_path)
    print(f"✅ Training complete. Model saved to: {model_save_path}")


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    MODEL_ARCHITECTURE = "efficientnet_v2_s"
    
    print("\n--- TRAINING MODEL ---")
    train_model(
        model_name=MODEL_ARCHITECTURE,
        data_dir="data/dataset",  # <<< THAY ĐỔI QUAN TRỌNG
        model_save_path=f"model/{MODEL_ARCHITECTURE}.pth", # <<< TÊN FILE MỚI
        num_classes=4, # <<< THAY ĐỔI QUAN TRỌNG
        device=device,
        batch_size=32 
    )