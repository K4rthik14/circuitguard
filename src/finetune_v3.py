import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from copy import deepcopy

# ------------------------- CONFIG -------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNetB4 v3 Stable")
    parser.add_argument("--data-dir", type=str, default="../outputs/labeled_rois_jpeg", help="Dataset directory")
    parser.add_argument("--model-load-path", type=str, required=True, help="Path to pretrained model (.pth)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="../models_finetune_v3_fixed")
    return parser.parse_args()

# ------------------------- AUGMENTATIONS -------------------------
def get_transforms():
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(380),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tfms, val_tfms

# ------------------------- EMA HELPER -------------------------
class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema_model = deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k] * (1 - self.decay))

# ------------------------- MIXUP -------------------------
def mixup_data(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y, y, 1
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ------------------------- TRAIN FUNCTION -------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device, epoch, total_epochs, use_mixup=True):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}]")
    for imgs, labels in progress:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)

        with autocast():
            outputs = model(imgs)
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

# ------------------------- VALIDATION -------------------------
def validate(model, loader, criterion, device):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return val_loss / len(loader), 100.0 * correct / total

# ------------------------- MAIN -------------------------
def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    print("📦 Loading dataset...")
    train_tfms, val_tfms = get_transforms()
    train_data = datasets.ImageFolder(os.path.join(args.data_dir, "train"), transform=train_tfms)
    val_data = datasets.ImageFolder(os.path.join(args.data_dir, "val"), transform=val_tfms)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    num_classes = len(train_data.classes)
    print(f"Classes: {train_data.classes}")

    # Model
    model = models.efficientnet_b4(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    print(f"Loading weights from {args.model_load_path} ...")
    ckpt = torch.load(args.model_load_path, map_location="cpu")
    model.load_state_dict(ckpt)
    print("✅ Weights loaded.")

    model = model.to(device)
    ema = ModelEMA(model)

    # Freeze all but classifier for first 10 epochs
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # Gradually unfreeze after epoch 10
        if epoch == 10:
            print("🔓 Unfreezing all layers for deeper fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr / 2, weight_decay=1e-4)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema, device, epoch, args.epochs)
        val_loss, val_acc = validate(ema.ema_model, val_loader, criterion, device)
        scheduler.step()

        print(f"✅ Fine-Tune Epoch [{epoch}/{args.epochs}] - Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, f"finetune_v3_fixed_best_{best_acc:.2f}.pth")
            torch.save(ema.ema_model.state_dict(), best_path)
            print(f"⭐ New best model saved to: {best_path}")

    print(f"\n--- Fine-Tuning Completed ---\nBest Accuracy Achieved: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
