import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from timm import create_model
from timm.utils import ModelEmaV2
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def main():
    # --- Paths ---
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    data_dir = os.path.join(project_root, 'outputs', 'labeled_rois_jpeg')

    # --- Config ---
    batch_size = 32
    num_epochs = 40
    lr = 3e-4
    img_size = 128
    val_split = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data transforms ---
    data_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print("📦 Loading dataset...")
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    num_classes = len(dataset.classes)
    print(f"📦 Classes: {dataset.classes}")

    # --- Split dataset ---
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Compute class weights ---
    targets = [y for _, y in dataset.imgs]
    class_counts = np.bincount(targets)
    class_weights = len(targets) / (num_classes * class_counts)
    print(f"⚖️ Class Weights: {class_weights}")
    weights_tensor = torch.FloatTensor(class_weights).to(device)

    # --- Model ---
    model = create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
    model.to(device)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler("cuda")

    # --- EMA (Exponential Moving Average of weights) ---
    ema_model = ModelEmaV2(model, decay=0.9999, device=device)
    print(f"🚀 Using device: {device}")

    # --- Training ---
    best_acc = 0
    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema_model.update(model)  # keep EMA weights synced

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)

        # --- Validation ---
        ema_model.module.eval()  # <-- Fixed
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = ema_model.module(images)  # <-- Fixed
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        epoch_acc = 100 * correct / total
        val_accuracies.append(epoch_acc)
        scheduler.step()

        print(f"✅ Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}%")

        # --- Save Best Model ---
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            model_path = os.path.join(project_root, f'best_model_acc{best_acc:.2f}.pth')
            torch.save(ema_model.module.state_dict(), model_path)
            print(f"💾 Saved best model: {model_path}")

    print(f"\n🎯 Training complete! Best Accuracy: {best_acc:.2f}%")

    # --- Graphs ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='blue', marker='o')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Validation Accuracy", color='green', marker='o')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, "training_curve.png"))
    plt.show()

    # --- Confusion Matrix ---
    print("\n📊 Generating confusion matrix...")
    all_preds, all_labels = [], []
    ema_model.module.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = ema_model.module(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(project_root, "confusion_matrix.png"))
    plt.show()


if __name__ == "__main__":
    main()
