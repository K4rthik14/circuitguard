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
import torch.nn.functional as F

# --- Focal Loss Implementation ---
# A smarter loss function to handle hard-to-classify examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    # --- Configuration ---
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    data_dir = os.path.join(project_root, 'outputs', 'labeled_rois_jpeg')

    # --- Data Preparation (with advanced augmentation) ---
    data_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        # Using TrivialAugment for better and more varied augmentation
        transforms.TrivialAugmentWide(),
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

    # Using Focal Loss instead of CrossEntropyLoss
    criterion = FocalLoss()
    # Optimizer with weight decay to prevent overfitting
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🚀 Using device: {device}")

    # --- Training Setup ---
    # Increased epochs for more thorough training
    num_epochs = 50
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
        
        # Save the model only if its performance has improved
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_model_path = os.path.join(project_root, f'best_model_acc_{best_accuracy:.2f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ New best model saved to: {best_model_path}")

    print(f"\n--- Training Finished ---")
    print(f"Highest validation accuracy achieved: {best_accuracy:.2f}%")

    # --- Generate and Save Graphs ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss'); plt.title("Training Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='g'); plt.title("Validation Accuracy"); plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'final_training_performance.png'))
    plt.show()

    # --- Generate Confusion Matrix ---
    print("\nGenerating final confusion matrix...")
    all_preds, all_labels = [], []
    # Load the best performing model for the final evaluation
    model.load_state_dict(torch.load(best_model_path))
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
    plt.title("Final Confusion Matrix"); plt.xticks(rotation=45)
    plt.savefig(os.path.join(project_root, 'final_confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    main()