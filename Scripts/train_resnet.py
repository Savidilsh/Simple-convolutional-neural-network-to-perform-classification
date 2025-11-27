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
from torchvision import datasets, models

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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        loss = output[0, class_idx]
        loss.backward(retain_graph=True)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def load_data():
    with open(SPLIT_MANIFEST) as f:
        manifest = json.load(f)
    with open(STATS_CACHE) as f:
        stats = json.load(f)
    return manifest, stats['mean'], stats['std']


def get_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(20),
        T.ColorJitter(0.3, 0.3, 0.3, 0.1),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    eval_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    
    return train_transform, eval_transform, mean, std


def create_loaders(manifest, batch_size=32):
    class_names = manifest['class_names']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    train_t, eval_t, mean, std = get_transforms()
    
    train_ds = WasteDataset(DATA_ROOT, manifest['splits']['train'], class_to_idx, train_t)
    val_ds = WasteDataset(DATA_ROOT, manifest['splits']['val'], class_to_idx, eval_t)
    test_ds = WasteDataset(DATA_ROOT, manifest['splits']['test'], class_to_idx, eval_t)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_names, mean, std


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


def train_model(model, train_loader, val_loader, num_epochs=15):
    criterion = nn.CrossEntropyLoss()
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    print("Stage 1: Training classifier head (4 epochs)")
    for epoch in range(4):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/4 | train: {train_loss:.4f}/{train_acc:.3f} | val: {val_loss:.4f}/{val_acc:.3f}")
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
    
    best_val_acc = 0
    best_model = None
    patience = 0
    
    print("\nStage 2: Fine-tuning all layers")
    for epoch in range(num_epochs):
        start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
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
            
        if patience >= 6:
            print(f"\nEarly stop. Best val_acc: {best_val_acc:.4f}")
            break
    
    if best_model:
        model.load_state_dict(best_model)


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


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('ResNet-18 Confusion Matrix')
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / 'resnet_confusion.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved confusion matrix to {ARTIFACT_DIR / 'resnet_confusion.png'}")


def generate_gradcam(model, loader, class_names, mean, std, num_samples=4):
    out_dir = ARTIFACT_DIR / "gradcam"
    out_dir.mkdir(exist_ok=True)
    
    target_layer = model.layer4[-1].conv2
    gcam = GradCAM(model, target_layer)
    
    inv_norm = T.Normalize([-m/s for m, s in zip(mean, std)], [1/s for s in std])
    
    count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            if count >= num_samples:
                break
            inputs_gpu = inputs.to(DEVICE)
            for i in range(min(inputs.size(0), num_samples - count)):
                img = inputs_gpu[i:i+1]
                label = targets[i].item()
                
                heatmap = gcam(img)
                
                img_np = inv_norm(inputs[i]).permute(1, 2, 0).clamp(0, 1).numpy()
                
                cmap = plt.get_cmap('jet')
                colored = cmap(heatmap)[:, :, :3]
                overlay = 0.5 * img_np + 0.5 * colored
                overlay = np.clip(overlay, 0, 1)
                
                plt.imsave(out_dir / f"gradcam_{count}_{class_names[label].replace('/', '_')}.png", overlay)
                count += 1
            if count >= num_samples:
                break
    
    gcam.remove_hooks()
    print(f"Saved {count} Grad-CAM visualizations to {out_dir}")


def main():
    set_seed(42)
    
    manifest, _, _ = load_data()
    train_loader, val_loader, test_loader, class_names, mean, std = create_loaders(manifest)
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(DEVICE)
    
    print("=" * 70)
    print("ResNet-18 Transfer Learning")
    print("=" * 70)
    print(f"Training on {DEVICE}\n")
    
    train_model(model, train_loader, val_loader)
    
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
    }, ARTIFACT_DIR / 'resnet_model.pt')
    
    with open(ARTIFACT_DIR / 'resnet_metrics.json', 'w') as f:
        json.dump({
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'confusion_matrix': results['confusion_matrix'].tolist(),
        }, f, indent=2)
    
    plot_confusion_matrix(results['confusion_matrix'], class_names)
    generate_gradcam(model, test_loader, class_names, mean, std)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
