import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import timm
import cv2
import pandas as pd

#Configuration
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# Path to the folder containing your best model run
MODEL_RUN_FOLDER = os.path.join(PROJECT_ROOT, "models") 
# Name of your best model file
MODEL_NAME = "final_model.pth"
# Path to your dataset directory
DATA_DIR = os.path.join(PROJECT_ROOT, "outputs", "labeled_rois_jpeg")
# Folder to save the final evaluation results
SAVE_DIR = os.path.join(MODEL_RUN_FOLDER, "final_evaluation_on_test_set")

def load_model(num_classes: int, model_path: str, device: torch.device) -> timm.models.EfficientNet:
    """Loads the trained model from a .pth file."""
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f" Model loaded from: {model_path}")
    return model

def get_test_loader(data_dir: str, batch_size: int = 32) -> tuple:
    """Creates a DataLoader for the test set."""
    img_size = 128
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    test_path = os.path.join(data_dir, "test")
    if not os.path.isdir(test_path):
        print(f"Error: Test directory not found at '{test_path}'")
        print("Please create the 'test' directory and move some validation images into it.")
        return None, None

    test_dataset = datasets.ImageFolder(test_path, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f" Test dataset loaded. Found {len(test_dataset)} images in {len(test_dataset.classes)} classes.")
    return test_loader, test_dataset.classes

def evaluate_model(model: timm.models.EfficientNet, test_loader: DataLoader, device: torch.device) -> tuple:
    """Runs the model on the test set and returns predictions and labels."""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    return np.array(all_labels), np.array(all_preds)

def save_annotated_images(test_loader: DataLoader, predictions: np.ndarray, classes: list, save_dir: str, num_images: int = 15):
    """Saves a few test images with their predicted labels drawn on them."""
    annotated_dir = os.path.join(save_dir, "annotated_test_images")
    os.makedirs(annotated_dir, exist_ok=True)
    
    # Get the file paths from the dataset
    image_paths = test_loader.dataset.samples
    
    # Get a random sample of images to annotate
    indices = random.sample(range(len(image_paths)), min(num_images, len(image_paths)))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for i in indices:
        image_path, true_label_idx = image_paths[i]
        predicted_label_idx = predictions[i]

        true_label = classes[true_label_idx]
        predicted_label = classes[predicted_label_idx]

        # Read image with OpenCV
        image = cv2.imread(image_path)
        
        # Set text and color
        text = f"Predicted: {predicted_label}"
        color = (0, 255, 0) # Green for correct
        if predicted_label != true_label:
            text += f" (True: {true_label})"
            color = (0, 0, 255) # Red for incorrect

        # Draw the text on the image
        cv2.putText(image, text, (10, 20), font, 0.6, color, 2)
        
        # Save the annotated image
        save_path = os.path.join(annotated_dir, f"annotated_{i}_{os.path.basename(image_path)}")
        cv2.imwrite(save_path, image)
        
    print(f" Saved {len(indices)} annotated images to: {annotated_dir}")
def plot_per_class_accuracy(report_dict: dict, classes: list, save_path: str):
    """Generates and saves a bar chart of per-class accuracy (recall)."""
    accuracies = [report_dict[cls]['recall'] * 100 for cls in classes] # Recall is accuracy per class
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(classes, accuracies, color='skyblue')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy (Recall) on Test Set')
    ax.set_ylim(0, 105)
    plt.xticks(rotation=45, ha='right')
    # Add accuracy values on top of bars
    for i, v in enumerate(accuracies):
        ax.text(i, v + 1, f"{v:.2f}%", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Per-class accuracy bar chart saved to: {save_path}")
    plt.close(fig) # Close the figure to free memory
def main():
    """Main function to run the evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # 1. Load the test data
    test_loader, classes = get_test_loader(DATA_DIR)
    if test_loader is None:
        return
        
    # 2. Load the best model
    model_path = os.path.join(MODEL_RUN_FOLDER, MODEL_NAME)
    if not os.path.exists(model_path):
        print(f" Error: Model file not found at '{model_path}'")
        return
    model = load_model(len(classes), model_path, device)
    
    #Run inference to get predictions
    print(" Running inference on the test set...")
    true_labels, predictions = evaluate_model(model, test_loader, device)
    
    # In the main() function, after printing the report:

    # 4. Final evaluation report with metrics
    print("\n--- Final Evaluation Report ---")
    # --- CHANGE: Get report as dict for plotting ---
    report_dict = classification_report(true_labels, predictions, target_names=classes, digits=4, output_dict=True)
    report_text = classification_report(true_labels, predictions, target_names=classes, digits=4) # For text file
    print(report_text)

    # Save the report to a text file
    report_path = os.path.join(SAVE_DIR, "final_evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("--- Final Evaluation Report (on Test Set) ---\n\n")
        f.write(report_text)
    print(f"Report saved to: {report_path}")

    #Plot per-class accuracy ---
    acc_chart_path = os.path.join(SAVE_DIR, "per_class_accuracy_test_set.png")
    # Exclude accuracy/macro/weighted avg keys before plotting
    class_metrics = {k: v for k, v in report_dict.items() if k in classes}
    plot_per_class_accuracy(class_metrics, classes, acc_chart_path)

    
    
    #Final confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Final Confusion Matrix (on Test Set)")
    cm_path = os.path.join(SAVE_DIR, "final_confusion_matrix_test_set.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to: {cm_path}")
    
    # Annotated output test images
    save_annotated_images(test_loader, predictions, classes, SAVE_DIR)
    
    print("\n--- Module 4 Complete ---")


if __name__ == "__main__":
    main()