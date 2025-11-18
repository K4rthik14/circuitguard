import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import timm


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loaders(data_dir: str, batch_size: int = 32) -> tuple:
    img_size = 128

    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=train_tfms)
    val_dataset = datasets.ImageFolder(val_path, transform=val_tfms)
    test_dataset = datasets.ImageFolder(test_path, transform=val_tfms) if os.path.isdir(test_path) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(num_classes: int) -> nn.Module:
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    return model


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)


def train_and_validate(project_root: str) -> None:
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(project_root, "outputs", "labeled_rois_jpeg")

    #Create a new timestamped folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(project_root, "models", f"run_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Models for this run will be saved in: {save_dir}")

    train_loader, val_loader, test_loader, classes = get_data_loaders(data_dir, batch_size=32)
    num_classes = len(classes)
#Building the model here
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

#num_epochs
    num_epochs = 50
    early_stop_patience = 8
    best_val_acc = 0.0
    best_path = "" # Pathupdated dynamically
    no_improve_epochs = 0

    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        epoch_train_loss = running_loss / max(1, num_samples)
        train_losses.append(epoch_train_loss)

        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                val_total += labels.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        epoch_val_loss = val_running_loss / max(1, val_total)
        epoch_val_acc = 100.0 * val_correct / max(1, val_total)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        scheduler.step()

        print(f"Epoch {epoch:02d}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

            # Save a new file for each best model
            best_path = os.path.join(save_dir, f"epoch_{epoch:02d}_acc_{best_val_acc:.2f}.pth")
            torch.save(model.state_dict(), best_path)
            print(f"⭐ New best model saved to: {best_path}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    if best_path:
        print(f"Best model was saved to {best_path}")

    # Reload best for evaluation/plots
    if not os.path.exists(best_path):
        print("\nNo best model was saved. Skipping plots and confusion matrix.")
        return

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    # Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy", color="g")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()

    plots_path = os.path.join(save_dir, "final_training_performance.png")
    plt.tight_layout()
    plt.savefig(plots_path)
    print(f"Saved training curves to {plots_path}")

    # Confusion matrix on validation set (or test set if available)
    eval_loader = None
    eval_split_name = "val"
    if test_loader is not None:
        eval_loader = test_loader
        eval_split_name = "test"
    else:
        eval_loader = val_loader

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"Confusion Matrix ({eval_split_name})")
    cm_path = os.path.join(save_dir, f"confusion_matrix_{eval_split_name}.png")
    plt.savefig(cm_path)
    print(f"Saved confusion matrix to {cm_path}")


def main() -> None:
    project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    start = time.time()
    train_and_validate(project_root)
    end = time.time()
    print(f"Total time: {(end - start)/60:.1f} min")


if __name__ == "__main__":
    main()