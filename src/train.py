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

    # --- Data Preparation (Using stable augmentation) ---
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        # Simpler augmentations that proved more stable
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

    criterion = nn.CrossEntropyLoss()
    # Using the learning rate that performed better
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🚀 Using device: {device}")

    # --- Training Setup ---
    # Training for longer to allow for convergence
    num_epochs = 35
    scaler = GradScaler()
    train_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
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

    # --- Save Model with a unique name ---
    final_accuracy = val_accuracies[-1]
    model_save_path = os.path.join(project_root, f'pcb_classifier_epochs{num_epochs}_acc{final_accuracy:.2f}.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"\n💾 Model saved to: {model_save_path}")

    # --- Generate and Save Graphs ---
    # Plot 1: Training Loss and Validation Accuracy
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
    graph_path = os.path.join(project_root, 'training_performance.png')
    plt.savefig(graph_path)
    print(f"📈 Saved training performance graph to: {graph_path}")
    plt.show()

    # Plot 2: Confusion Matrix
    print("\nGenerating confusion matrix...")
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45)
    cm_path = os.path.join(project_root, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"📈 Saved confusion matrix to: {cm_path}")
    plt.show()

if __name__ == '__main__':
    main()