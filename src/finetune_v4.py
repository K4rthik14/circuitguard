import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# --- CHANGE: Import 'create_model' from timm ---
from timm import create_model
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# =============================
# CONFIG
# =============================
# --- Corrected, robust path setup ---
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.path.join(project_root, "outputs", "labeled_rois_jpeg")
# --- IMPORTANT: Update this to your best saved model's filename ---
MODEL_LOAD_PATH = os.path.join(project_root, "best_model_acc_95.71.pth")
OUTPUT_DIR = os.path.join(project_root, "models_finetune_v4")

EPOCHS = 30
BATCH_SIZE = 16
LR = 2e-5 # A smaller learning rate is better for the final fine-tuning
SEED = 42

# (The rest of the script is the same until the create_model function)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def prepare_datasets(data_dir):
    train_tfms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tfms)
    val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tfms)

    print(f"📦 Dataset split -> Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    return train_dataset, val_dataset, train_dataset.classes

# =============================
# MODEL SETUP (Corrected)
# =============================
def get_model(num_classes, model_path):
    # --- CHANGE: Use timm.create_model to match the saved architecture ---
    model = create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt)
    print("✅ Pretrained weights loaded.")

    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

# (The rest of the script remains the same)
# =============================
# TRAIN LOOP
# =============================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
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
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    train_dataset, val_dataset, classes = prepare_datasets(DATA_DIR)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- CHANGE: Renamed the function call ---
    model = get_model(num_classes=len(classes), model_path=MODEL_LOAD_PATH).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🧠 Epoch [{epoch}/{EPOCHS}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"✅ Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {scheduler.get_last_lr()[0]:.2e}")

        if epoch == 5:
            print("🔓 Unfreezing all layers for deeper fine-tuning...")
            for param in model.parameters():
                param.requires_grad = True
            # Re-initialize optimizer to include all parameters
            optimizer = optim.AdamW(model.parameters(), lr=LR / 10, weight_decay=1e-4)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(OUTPUT_DIR, f"finetune_v4_best_acc_{best_acc:.2f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Saved new best model: {save_path}")

    print(f"\n🎯 Fine-tuning complete. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()