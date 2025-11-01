import os
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


DATASET_DIR = Path("Dataset/RealWaste")
SPLITS_JSON = Path("Dataset/realwaste_splits.json")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class SimpleCNNNoBN(nn.Module):
    """A small CNN without any normalization layers.
    Conv -> ReLU -> MaxPool blocks with Dropout for regularization.
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Dropout(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Dropout(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
            nn.Dropout(dropout),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14
            nn.Dropout(dropout),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def get_dataloaders(img_size: int = 224, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    # Augmentation for train; light transforms for val/test. No per-channel normalization.
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Use ImageFolder and rely on split lists from realwaste_splits.json
    with open(SPLITS_JSON, "r", encoding="utf-8") as f:
        splits = json.load(f)

    class SubsetFromList(datasets.ImageFolder):
        def __init__(self, root, files, transform=None):
            super().__init__(root, transform=transform)
            # Map absolute file list to indices in samples
            file_set = set(os.path.normpath(p) for p in files)
            new_samples = []
            for path, target in self.samples:
                if os.path.normpath(path) in file_set:
                    new_samples.append((path, target))
            self.samples = new_samples
            self.imgs = new_samples

    # Expand relative paths to absolute paths to match ImageFolder samples
    def expand_paths(file_list):
        return [str((DATASET_DIR / p).resolve()) for p in file_list]

    train_files = expand_paths(splits["train"])
    val_files = expand_paths(splits["val"])
    test_files = expand_paths(splits["test"])

    train_ds = SubsetFromList(DATASET_DIR, train_files, transform=train_tf)
    val_ds = SubsetFromList(DATASET_DIR, val_files, transform=eval_tf)
    test_ds = SubsetFromList(DATASET_DIR, test_files, transform=eval_tf)

    class_names = train_ds.classes

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader, class_names


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_labels, all_preds = [], []
    running_loss, total = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        total += labels.size(0)
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return running_loss / total, acc, cm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(batch_size=args.batch_size)

    model = SimpleCNNNoBN(num_classes=len(class_names), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0.0
    best_path = ARTIFACTS_DIR / "simple_cnn_nobn.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {tr_loss:.4f} acc {tr_acc:.3f} | val_loss {val_loss:.4f} acc {val_acc:.3f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": class_names}, best_path)

    # Load best and test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"]) 
    test_loss, test_acc, cm = evaluate(model, test_loader, device)
    print(f"Test acc: {test_acc:.3f}")
    # Save confusion matrix and summary
    np.save(ARTIFACTS_DIR / "simple_cnn_nobn_confusion.npy", cm)
    with open(ARTIFACTS_DIR / "simple_cnn_nobn_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "epochs": args.epochs,
            "val_best_acc": float(best_val_acc),
            "test_acc": float(test_acc),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
        }, f, indent=2)


if __name__ == "__main__":
    main()
