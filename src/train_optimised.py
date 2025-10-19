import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timm import create_model
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from copy import deepcopy

# --- 1. CONFIGURATION ---
DATA_DIR = "outputs/labeled_rois_jpeg"
# IMPORTANT: Path to your best model from the initial training phase
MODEL_LOAD_PATH = "fine_tuned_model_acc_95.71.pth" # Update this path
SAVE_DIR = "../models_optimized"
EPOCHS = 40
BATCH_SIZE = 16
LR = 2e-5 # Start with a low learning rate for fine-tuning
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. ADVANCED AUGMENTATIONS ---
def get_transforms():
    # Augmentations inspired by your finetune_v3 and finetune_v4 scripts
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(380, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tfms, val_tfms

# --- 3. EXPONENTIAL MOVING AVERAGE (EMA) ---
# For more stable and accurate final weights
class ModelEMA:
    def __init__(self, model, decay=0.9998):
        self.ema_model = deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad = False
        self.decay = decay

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k].detach() * (1 - self.decay))

# --- 4. MIXUP AUGMENTATION ---
# A powerful regularization technique
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- 5. TRAINING & VALIDATION LOOPS ---
def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, epoch):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc=f"Train Epoch {epoch}/{EPOCHS}")
    for images, labels in progress:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(images, labels)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA model
        ema.update(model)

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)

def validate(model, loader, criterion):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return val_loss / len(loader), 100.0 * correct / total

# --- 6. MAIN EXECUTION ---
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    print("📦 Loading dataset...")
    train_tfms, val_tfms = get_transforms()
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_tfms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    num_classes = len(train_dataset.classes)
    print(f"Classes: {train_dataset.classes}")

    print("🚀 Setting up model: efficientnet_b4")
    model = create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    
    # Load your best pre-trained weights
    if os.path.exists(MODEL_LOAD_PATH):
        model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=DEVICE))
        print(f"✅ Weights loaded from: {MODEL_LOAD_PATH}")
    else:
        print(f"⚠️ Pretrained weights not found at {MODEL_LOAD_PATH}. Training from scratch.")

    model.to(DEVICE)
    ema_model = ModelEMA(model)

    # Freeze all layers except the classifier for the first few epochs
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    # Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    best_acc = 0.0
    print("--- Starting Optimized Training ---")
    for epoch in range(1, EPOCHS + 1):
        
        # Unfreeze all layers after a few epochs
        if epoch == 10:
            print("🔓 Unfreezing all layers for deep fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # Re-initialize optimizer with a smaller LR for the whole network
            optimizer = optim.AdamW(model.parameters(), lr=LR / 10, weight_decay=1e-4)
            # You might want to reset the scheduler as well
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-epoch+1, eta_min=1e-6)


        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, ema_model, epoch)
        # Validate using the more stable EMA model weights
        val_loss, val_acc = validate(ema_model.ema_model, val_loader, criterion)
        
        scheduler.step()

        print(f"✅ Epoch {epoch}/{EPOCHS} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(SAVE_DIR, f"optimized_best_acc_{best_acc:.2f}.pth")
            # Save the EMA model state
            torch.save(ema_model.ema_model.state_dict(), save_path)
            print(f"⭐ New best model saved: {save_path}")

    print(f"\n--- Training Complete ---")
    print(f"🎯 Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()