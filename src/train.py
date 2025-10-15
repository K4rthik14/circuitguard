import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import torch.nn as nn
import torch.optim as optim
import os

# --- Configuration ---
# This setup assumes the script is in the 'src' folder
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
# Make sure this path points to your folder with the labeled JPEGs
data_dir = os.path.join(project_root, 'outputs', 'labeled_rois_jpeg') 

# --- 1. Data Loading and Preparation ---

# Define the transformations for the images
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128 [cite: 84]
    # Add data augmentation to improve model robustness
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    # Convert images to PyTorch tensors and normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading dataset...")
# Load the dataset using PyTorch's ImageFolder
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)

# Split the dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Create DataLoaders to feed data to the model in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Dataset loaded. Classes found: {full_dataset.classes}")
print(f"Total images: {len(full_dataset)}")

# --- 2. Model, Loss, and Optimizer Setup ---

# Set up the model as required
num_classes = len(full_dataset.classes)
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes) # [cite: 83]

# Define Loss Function and Optimizer
criterion = nn.CrossEntropyLoss() # [cite: 85]
optimizer = optim.Adam(model.parameters(), lr=0.001) # [cite: 85]

# Check if a GPU is available and move the model to it
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# --- 3. The Training Loop ---
num_epochs = 10  # Start with 10 and you can increase later

for epoch in range(num_epochs):
    # --- Training Phase ---
    model.train() # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        # Move images and labels to the GPU if available
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # Reset gradients from the previous step
        outputs = model(images) # Get model predictions
        loss = criterion(outputs, labels) # Calculate how wrong the predictions were
        loss.backward() # Calculate gradients
        optimizer.step() # Update the model's weights

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_dataset)

    # --- Validation Phase ---
    model.eval() # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # No need to calculate gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_acc = 100 * correct / total
    
    # Print a summary for the epoch
    print(f'--- Epoch {epoch+1}/{num_epochs} ---')
    print(f'Training Loss: {epoch_loss:.4f} | Validation Accuracy: {epoch_acc:.2f}%')

print("\n--- Training Finished ---")

# --- 4. Save the Trained Model ---
model_save_path = os.path.join(project_root, 'pcb_defect_classifier.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Trained model saved to: {model_save_path}")