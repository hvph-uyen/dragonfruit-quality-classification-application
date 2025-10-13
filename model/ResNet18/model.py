import os
import copy
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


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


def train_two_stage_resnet(data_dir, model_save_path,
                           num_classes, batch_size=32,
                           stage1_epochs=4, stage2_epochs=12,
                           lr_stage1=1e-3, lr_stage2=1e-4, device="cpu"):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    num_workers = 0 if platform.system() == "Darwin" else 4

    # ------------------------
    # Transforms & Data
    # ------------------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ------------------------
    # Model
    # ------------------------
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # ------------------------
    # Stage 1: Freeze backbone
    # ------------------------
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    optimizer_stage1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_stage1)

    best_val_acc_stage1 = 0.0
    for epoch in range(1, stage1_epochs + 1):
        print(f"\nStage1 Epoch {epoch}/{stage1_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage1, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Stage1 | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")
        if val_acc > best_val_acc_stage1:
            best_val_acc_stage1 = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'classes': train_ds.classes},
                       model_save_path.replace(".pth", "_stage1_best.pth"))

    # ------------------------
    # Stage 2: Unfreeze all
    # ------------------------
    for param in model.parameters():
        param.requires_grad = True
    optimizer_stage2 = optim.SGD(model.parameters(), lr=lr_stage2, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_stage2, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0
    best_model_wts = None
    for epoch in range(1, stage2_epochs + 1):
        print(f"\nStage2 Epoch {epoch}/{stage2_epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer_stage2, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Stage2 | Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'model_state_dict': best_model_wts, 'classes': train_ds.classes},
                       model_save_path.replace(".pth", "_best.pth"))

    if best_model_wts:
        model.load_state_dict(best_model_wts)
    torch.save({'model_state_dict': model.state_dict(), 'classes': train_ds.classes}, model_save_path)
    print("✅ Training complete. Final model saved at:", model_save_path)

    return model, train_ds.classes, model_save_path


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    # Example training DETECTOR (binary: dragonfruit vs not_dragonfruit)
    # train_two_stage_resnet(
    #     data_dir="dataset2",
    #     model_save_path="model/detector_model.pth",
    #     num_classes=2,
    #     device=device
    # )

    # Example training CLASSIFIER (3 classes: good/immature/reject)
    train_two_stage_resnet(
        data_dir="dataset",
        model_save_path="model/classifier_model2.pth",
        num_classes=3,
        device=device
    )