# src/finetune_v3.py
import os
import math
import copy
import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from torchvision import datasets, transforms

from timm import create_model
from timm.utils import ModelEmaV2
from timm.data import Mixup  # optional, use if available

# ---------------------------
# Utility functions
# ---------------------------
def unfreeze_last_n_blocks(model, n):
    """
    For timm EfficientNet (features is a Sequential of blocks), unfreeze last n children of model.features.
    """
    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False

    # Attempt typical timm EfficientNet structure
    if hasattr(model, "features"):
        children = list(model.features.children())
        if n <= 0:
            return
        to_unfreeze = children[-n:]
        for block in to_unfreeze:
            for p in block.parameters():
                p.requires_grad = True

    # Ensure classifier head is trainable
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True


def get_num_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def tta_predict(model_fn, images, tta_transforms, device):
    """
    images: PIL / tensor batch (already normalized tensor batch)
    tta_transforms: list of functions that take a batch tensor and return an augmented batch tensor
    model_fn: function that given images returns logits (on device)
    Returns averaged softmax probs.
    """
    model_fn = model_fn
    probs = None
    with torch.no_grad():
        for tfunc in tta_transforms:
            aug_images = tfunc(images)
            logits = model_fn(aug_images)
            p = torch.softmax(logits, dim=1)
            probs = p if probs is None else probs + p
    probs = probs / len(tta_transforms)
    return probs


# ---------------------------
# Main fine-tune script
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    parser.add_argument("--data-dir", default=None)  # if None, derived from project_root/outputs/labeled_rois_jpeg
    parser.add_argument("--model-load-path", default=None, help="Path to base checkpoint to fine-tune")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=12)  # smaller if using bigger input size
    parser.add_argument("--img-size", type=int, default=380)  # B4 native is 380
    parser.add_argument("--lr", type=float, default=5e-5)  # low LR for fine-tune
    parser.add_argument("--unfreeze-blocks", type=int, default=3, help="Number of last blocks to unfreeze in features")
    parser.add_argument("--mixup", action="store_true", help="Enable MixUp/CutMix (timm Mixup)")
    parser.add_argument("--tta", action="store_true", help="Enable TTA during validation")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    data_dir = args.data_dir or os.path.join(project_root, "outputs", "labeled_rois_jpeg")
    save_dir = args.save_dir or os.path.join(project_root, "models_finetune_v3")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("📦 Loading dataset for fine-tuning...")
    print(f"Project root: {project_root}")
    print(f"Data dir: {data_dir}")

    # ---------------------------
    # Transforms
    # ---------------------------
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.25)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_base_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # TTA transforms - define as lambdas that take a tensor batch and return augmented tensor batch
    def tta_identity(x): return x
    def tta_hflip(x): return torch.flip(x, dims=[3])
    def tta_vflip(x): return torch.flip(x, dims=[2])
    def tta_hvflip(x): return torch.flip(x, dims=[2, 3])

    tta_fns = [tta_identity, tta_hflip, tta_vflip, tta_hvflip] if args.tta else [tta_identity]

    # ---------------------------
    # Dataset & DataLoaders
    # ---------------------------
    dataset = datasets.ImageFolder(data_dir, transform=train_transforms)
    classes = dataset.classes
    num_classes = len(classes)
    print("Dataset loaded. Classes:", classes)

    # stable split
    val_ratio = 0.2
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42))

    # override val_dataset transform with deterministic val transforms
    val_dataset.dataset = copy.copy(dataset)  # shallow copy to avoid altering original
    val_dataset.dataset.transform = val_base_transforms

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ---------------------------
    # Class weights (balanced)
    # ---------------------------
    targets = [y for _, y in dataset.imgs]
    counts = Counter(targets)
    class_counts = np.array([counts[i] for i in range(num_classes)])
    class_weights = len(targets) / (num_classes * class_counts + 1e-12)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class counts:", class_counts)
    print("Class weights:", class_weights.cpu().numpy())

    # ---------------------------
    # Model
    # ---------------------------
    model = create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)  # instantiate first
    # load checkpoint if provided
    if args.model_load_path and os.path.exists(args.model_load_path):
        print(f"Loading weights from {args.model_load_path} ...")
        ckpt = torch.load(args.model_load_path, map_location="cpu")
        # try to handle both state_dict or full checkpoint
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        # some check for prefix
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            # try strip prefixes like 'module.' or 'model.'
            new_sd = {}
            for k, v in state_dict.items():
                new_k = k
                if k.startswith("module."):
                    new_k = k[len("module."):]
                if k.startswith("model."):
                    new_k = k[len("model."):]
                new_sd[new_k] = v
            model.load_state_dict(new_sd)
        print("✅ Weights loaded.")
    else:
        # fallback to pretrained weights from timm (internet required)
        model = create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        print("⚠️ No load path provided; using timm pretrained then reinitialized classifier.")

    model.to(device)

    # Gradual unfreeze: unfreeze last N blocks + classifier
    unfreeze_last_n_blocks(model, args.unfreeze_blocks)
    print("Trainable params after unfreeze:", get_num_trainable_params(model))

    # ---------------------------
    # Loss, optimizer, scheduler, EMA, amp
    # ---------------------------
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    # Cosine annealing with warmup helper
    total_epochs = args.epochs
    warmup_epochs = max(2, int(0.05 * total_epochs))  # e.g., 2 or 5% of total
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs))

    ema = ModelEmaV2(model, decay=0.9999, device=device)
    try:
        scaler = GradScaler(device_type="cuda" if device.type == "cuda" else "cpu")
    except TypeError:
        scaler = GradScaler()


    # optional mixup
    mixup_fn = None
    if args.mixup:
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=num_classes)
        print("MixUp enabled")

    # ---------------------------
    # Training loop
    # ---------------------------
    best_acc = 0.0
    best_path = None
    train_losses = []
    val_accs = []
    epochs_no_improve = 0
    early_stop_patience = 8

    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        # linear warmup of lr
        if epoch <= warmup_epochs:
            warmup_lr = args.lr * epoch / float(max(1, warmup_epochs))
            for g in optimizer.param_groups:
                g["lr"] = warmup_lr

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_loss = running_loss / (total_samples + 1e-12)
        train_losses.append(epoch_loss)

        # step lr scheduler (cosine) after warmup phase using one-step call
        if epoch > warmup_epochs:
            scheduler.step()

        # ---------------------------
        # Validation with TTA (averaging predictions)
        # ---------------------------
        ema.module.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Prepare TTA batch: tta_fns operate on batch tensors
                probs = None
                for tfunc in tta_fns:
                    aug = tfunc(images)
                    logits = ema.module(aug)
                    p = torch.softmax(logits, dim=1)
                    probs = p if probs is None else probs + p
                probs = probs / len(tta_fns)

                preds = torch.argmax(probs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_acc = 100.0 * correct / (total + 1e-12)
        val_accs.append(epoch_acc)

        # Logging
        print(f"✅ Fine-Tune Epoch [{epoch}/{total_epochs}] - Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save best EMA weights
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            epochs_no_improve = 0
            best_path = os.path.join(save_dir, f"finetune_v3_best_acc_{best_acc:.2f}.pth")
            torch.save(ema.module.state_dict(), best_path)
            print(f"⭐ New best model saved to: {best_path}")
        else:
            epochs_no_improve += 1

        # Early stop condition
        if epochs_no_improve >= early_stop_patience:
            print(f"⏹️ Early stopping: no improvement for {early_stop_patience} epochs.")
            break

    print("\n--- Fine-tuning finished ---")
    print(f"Highest validation accuracy achieved: {best_acc:.2f}%")
    if best_path:
        print("Loading best model for final evaluation:", best_path)
        model.load_state_dict(torch.load(best_path, map_location=device))

    # ---------------------------
    # Final evaluation: confusion matrix + classification report
    # ---------------------------
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            probs = None
            for tfunc in tta_fns:
                aug = tfunc(images)
                logits = model(aug)
                p = torch.softmax(logits, dim=1)
                probs = p if probs is None else probs + p
            probs = probs / len(tta_fns)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "finetune_v3_confusion_matrix.png")
    plt.savefig(cm_path)
    print("Saved confusion matrix to:", cm_path)

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Train Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="Val Acc", marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Val Accuracy")
    plt.legend()
    plt.tight_layout()
    curves_path = os.path.join(save_dir, "finetune_v3_curves.png")
    plt.savefig(curves_path)
    print("Saved training curves to:", curves_path)

if __name__ == "__main__":
    main()
