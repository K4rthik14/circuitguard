import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

def main():
    # --- Configuration ---
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    data_dir = os.path.join(project_root, 'outputs', 'labeled_rois_jpeg')
    model_save_path = os.path.join(project_root, 'pcb_defect_classifier.pth')

    # --- Data Preparation ---
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
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

    # On Windows, set num_workers = 0 to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"✅ Dataset loaded with classes: {dataset.classes}")
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --- Model Setup ---
    num_classes = len(dataset.classes)
    model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🚀 Using device: {device}")

    # --- Training Setup ---
    num_epochs = 10
    scaler = GradScaler()
    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)

            # --- Batch-wise Live Accuracy ---
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            batch_acc = 100 * correct / total

            if (batch_idx + 1) % 10 == 0:
                print(f"[Epoch {epoch+1}/{num_epochs}] Batch {batch_idx+1}/{len(train_loader)} "
                      f"- Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%")

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

    # --- Save Model ---
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 Model saved to: {model_save_path}")

    # --- Graphs ---
    plt.figure(figsize=(10, 5))
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
    graph_path = os.path.join(project_root, 'training_performance.png')
    plt.savefig(graph_path)
    print(f"📈 Saved training graph to: {graph_path}")
    plt.show()


if __name__ == '__main__':
    main()
