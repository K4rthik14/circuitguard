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
    # --- IMPORTANT: Update this to the name of your best saved model ---
    model_load_path = os.path.join(project_root, 'pcb_classifier_epochs35_acc94.61.pth') # Example filename

    # --- Data Preparation ---
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Using a slightly larger image size for fine-tuning
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("📦 Loading dataset for fine-tuning...")
    dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Reduce batch size to handle larger images
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Dataset loaded. Classes: {dataset.classes}")

    # --- Model Setup ---
    num_classes = len(dataset.classes)
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes) # Start with an un-trained model structure
    
    # --- Load the weights from your best model ---
    model.load_state_dict(torch.load(model_load_path))
    print(f"✅ Weights loaded from {os.path.basename(model_load_path)}")

    # --- Use a very small learning rate for fine-tuning ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5) # lr=0.00001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"🚀 Starting fine-tuning on device: {device}")

    # --- Training Setup ---
    num_epochs = 30 # Fine-tuning for 30 more epochs
    scaler = GradScaler()
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
        print(f"✅ Fine-Tuning Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}%")
        
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            best_model_path = os.path.join(project_root, f'fine_tuned_model_acc_{best_accuracy:.2f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"⭐ New best model saved to: {os.path.basename(best_model_path)}")

    print(f"\n--- Fine-Tuning Finished ---")
    print(f"Highest validation accuracy achieved: {best_accuracy:.2f}%")

    # (You can also add the graphing and confusion matrix code here if you wish)

if __name__ == '__main__':
    main()