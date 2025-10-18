import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import timm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from timm.utils import ModelEmaV2
import warnings
warnings.filterwarnings("ignore")

# ============================
# ⚙️ CONFIGURATION
# ============================
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
data_dir = os.path.join(project_root, 'outputs', 'labeled_rois_jpeg')
save_dir = os.path.join(project_root, 'models')
os.makedirs(save_dir, exist_ok=True)

num_epochs = 40
batch_size = 32
learning_rate = 3e-4
early_stop_patience = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ============================
# 🧩 DATA PREPARATION
# ============================
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"📦 Classes: {dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ============================
# ⚖️ CLASS WEIGHTING (for imbalance)
# ============================
targets = [s[1] for s in dataset.samples]
class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=targets)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"⚖️ Class Weights: {class_weights.cpu().numpy()}")

# ============================
# 🧠 MODEL SETUP
# ============================
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
ema_model = ModelEmaV2(model, decay=0.9999)  # exponential moving average of weights

model = model.to(device)
scaler = GradScaler()

# ============================
# 🏋️ TRAINING LOOP
# ============================
best_acc = 0.0
epochs_no_improve = 0
train_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema_model.update(model)
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)

    # --- Validation ---
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_acc = 100 * correct / total
    val_accuracies.append(epoch_acc)
    scheduler.step(epoch_acc)

    print(f"📈 Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}%")

    # --- Save best model ---
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        epochs_no_improve = 0
        best_path = os.path.join(save_dir, f'best_model_acc{best_acc:.2f}.pth')
        torch.save(model.state_dict(), best_path)
        print(f"💾 Saved best model: {best_path}")
    else:
        epochs_no_improve += 1

    # --- Early Stopping ---
    if epochs_no_improve >= early_stop_patience:
        print(f"⏹️ Early stopping at epoch {epoch+1}")
        break

# ============================
# 📊 EVALUATION
# ============================
print("\nEvaluating best model...")
model.load_state_dict(torch.load(best_path))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=dataset.classes))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
plt.show()

# Loss & Accuracy Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green', marker='o')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'training_curves.png'))
plt.show()

print(f"\n✅ Best Validation Accuracy: {best_acc:.2f}%")
print(f"📂 Saved models in: {save_dir}")
