import os
import argparse
import random
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# =============================
# ARGUMENTS
# =============================
def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B4 for PCB defect classification")
    parser.add_argument("--data-dir", type=str, default="../outputs/labeled_rois_jpeg", help="Dataset directory")
    parser.add_argument("--model-load-path", type=str, required=True, help="Path to pretrained .pth model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="../models_finetune_v4")
    return parser.parse_args()

# =============================
# SETUP
# =============================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================
# DATASET PREP
# =============================
def prepare_datasets(data_dir, val_split):
    train_tfms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    full_dataset = datasets.ImageFolder(data_dir, transform=train_tfms)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_tfms  # Apply clean transform for val

    print(f"📦 Dataset split -> Train: {train_size} | Val: {val_size}")
    return train_dataset, val_dataset, full_dataset.classes

# =============================
# MODEL SETUP
# =============================
def create_model(num_classes, model_path):
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    print("✅ Pretrained weights loaded.")

    # Freeze all layers except classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    return model

# =============================
# TRAIN LOOP
# =============================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast("cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return val_loss / len(loader.dataset), 100 * correct / total

# =============================
# MAIN
# =============================
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load data
    train_dataset, val_dataset, classes = prepare_datasets(args.data_dir, args.val_split)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model setup
    model = create_model(num_classes=len(classes), model_path=args.model_load_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n🧠 Epoch [{epoch}/{args.epochs}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Gradual unfreezing after 10 epochs
        if epoch == 10:
            print("🔓 Unfreezing last 3 layers of EfficientNet...")
            for name, param in list(model.features[-3:].named_parameters()):
                param.requires_grad = True

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(args.output_dir, f"finetune_v4_best_acc_{best_acc:.2f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Saved new best model: {save_path}")

    print(f"\n🎯 Fine-tuning complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
