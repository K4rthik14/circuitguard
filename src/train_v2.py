import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    # --- Configuration ---
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    data_dir = os.path.join(project_root, 'outputs', 'labeled_rois_jpeg')

    # --- Data Preparation (Reverted to simpler, more stable augmentation) ---
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        # Using the simpler augmentations that provided stable results
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("📦 Loading dataset...")
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Dataset loaded with classes: {dataset.classes}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Model Setup ---
    num_classes = len(dataset.classes)
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)

    # Reverting to the standard, more stable loss function
    criterion = nn.CrossEntropyLoss()
    # Using the best optimizer settings we've found
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8) # Slower decay

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🚀 Using device: {device}")

    # --- Training Setup ---
    num_epochs = 40 # Increased epochs for longer, stable training
    scaler = GradScaler()
    train_losses, val_accuracies = [], []
    best_accuracy = 0.0

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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_acc = 100 * correct / total
        val_accuracies.append(epoch_acc)
        scheduler.step()

        print(f"✅ Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")
        
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_model_path = os.path.join(project_root, f'best_model_acc_{best_accuracy:.2f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ New best model saved to: {best_model_path}")

    print(f"\n--- Training Finished ---")
    print(f"Highest validation accuracy achieved: {best_accuracy:.2f}%")

    # (Graphing and confusion matrix code remains the same)
    # ...

if __name__ == '__main__':
    main()