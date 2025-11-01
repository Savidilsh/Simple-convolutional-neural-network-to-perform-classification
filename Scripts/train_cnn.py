import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "Dataset").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT / "Dataset/RealWaste"
SPLIT_MANIFEST = PROJECT_ROOT / "Dataset/realwaste_splits.json"
STATS_CACHE = PROJECT_ROOT / "Dataset/realwaste_stats.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WasteDataset(Dataset):
    def __init__(self, root, samples, class_to_idx, transform=None):
        self.root = root
        self.samples = list(samples)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        label_name = Path(rel_path).parts[0]
        target = self.class_to_idx[label_name]
        path = self.root / rel_path
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, target


class WasteCNN(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.dropout_conv = nn.Dropout2d(0.25)
        
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x, return_features=False):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        features = x
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        x = self.fc3(x)
        
        return (x, features) if return_features else x


def load_data():
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    with open(STATS_CACHE) as f:
        stats = json.load(f)
    return manifest, stats['mean'], stats['std']


def get_transforms(mean, std):
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(30),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
        T.RandomErasing(p=0.3),
    ])
    
    eval_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    
    return train_transform, eval_transform


def create_loaders(manifest, mean, std, batch_size=24):
    class_names = manifest['class_names']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    train_t, eval_t = get_transforms(mean, std)
    
    train_ds = WasteDataset(DATA_ROOT, manifest['splits']['train'], class_to_idx, train_t)
    val_ds = WasteDataset(DATA_ROOT, manifest['splits']['val'], class_to_idx, eval_t)
    test_ds = WasteDataset(DATA_ROOT, manifest['splits']['test'], class_to_idx, eval_t)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_names


def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, num_epochs=40):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_acc = 0
    best_model = None
    patience = 0
    max_patience = 12
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"Training on {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 70)
    
    for epoch in range(num_epochs):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        elapsed = time.time() - start
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"train: {train_loss:.4f}/{train_acc:.3f} | "
              f"val: {val_loss:.4f}/{val_acc:.3f} | "
              f"{elapsed:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            
        if patience >= max_patience:
            print(f"\nEarly stop at epoch {epoch+1}. Best val_acc: {best_val_acc:.4f}")
            break
    
    if best_model:
        model.load_state_dict(best_model)
    
    return history


def evaluate(model, loader, class_names):
    model.eval()
    targets_all = []
    preds_all = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            targets_all.extend(targets.numpy())
            preds_all.extend(preds)
    
    targets_np = np.array(targets_all)
    preds_np = np.array(preds_all)
    
    cm = confusion_matrix(targets_np, preds_np)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_np, preds_np, average='macro', zero_division=0
    )
    accuracy = (preds_np == targets_np).mean()
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm,
    }


def plot_results(history, cm, class_names):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                xticklabels=class_names, yticklabels=class_names)
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    axes[2].set_title('Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / 'cnn_results.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved results to {ARTIFACT_DIR / 'cnn_results.png'}")


def visualize_features(model, loader, class_names, mean, std, num_samples=4):
    model.eval()
    out_dir = ARTIFACT_DIR / "features"
    out_dir.mkdir(exist_ok=True)
    
    inv_norm = T.Normalize([-m/s for m, s in zip(mean, std)], [1/s for s in std])
    
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if count >= num_samples:
                break
            inputs = inputs.to(DEVICE)
            for i in range(min(inputs.size(0), num_samples - count)):
                img = inputs[i:i+1]
                label = targets[i].item()
                
                _, features = model(img, return_features=True)
                features = features[0].cpu()
                
                img_np = inv_norm(inputs[i]).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                
                fig = plt.figure(figsize=(12, 6))
                ax = plt.subplot(2, 5, 1)
                ax.imshow(img_np)
                ax.set_title(f"{class_names[label]}", fontsize=9)
                ax.axis('off')
                
                for j in range(min(9, features.shape[0])):
                    ax = plt.subplot(2, 5, j + 2)
                    ax.imshow(features[j].numpy(), cmap='viridis')
                    ax.axis('off')
                
                plt.tight_layout()
                plt.savefig(out_dir / f"sample_{count}.png", dpi=100, bbox_inches='tight')
                plt.close()
                count += 1
            if count >= num_samples:
                break
    
    print(f"Saved {count} feature visualizations to {out_dir}")


def main():
    set_seed(42)
    
    manifest, mean, std = load_data()
    train_loader, val_loader, test_loader, class_names = create_loaders(manifest, mean, std)
    
    model = WasteCNN(len(class_names)).to(DEVICE)
    
    print("=" * 70)
    print("Custom CNN Training")
    print("=" * 70)
    
    history = train_model(model, train_loader, val_loader, num_epochs=40)
    
    print("\n" + "=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)
    
    results = evaluate(model, test_loader, class_names)
    
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1']:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'test_accuracy': results['accuracy'],
    }, ARTIFACT_DIR / 'cnn_model.pt')
    
    with open(ARTIFACT_DIR / 'cnn_metrics.json', 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
        }, f, indent=2)
    
    plot_results(history, results['confusion_matrix'], class_names)
    visualize_features(model, test_loader, class_names, mean, std)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
