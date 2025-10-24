"""Utility to print confusion matrices for the fine-tuned SimpleWasteCNN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / "Dataset").exists() and (PROJECT_ROOT.parent / "Dataset").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_ROOT = PROJECT_ROOT / "Dataset/RealWaste"
SPLIT_MANIFEST = PROJECT_ROOT / "Dataset/realwaste_splits.json"
STATS_CACHE = PROJECT_ROOT / "Dataset/realwaste_stats.json"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
WEIGHTS_PATH = ARTIFACT_DIR / "simple_cnn_adam.pt"


def load_split_manifest(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_dataset_stats(path: Path) -> Tuple[List[float], List[float]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload["mean"], payload["std"]


class WasteDataset(Dataset):
    def __init__(
        self,
        root: Path,
        samples: Iterable[str],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.samples = list(samples)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = datasets.folder.default_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rel_path = self.samples[idx]
        label_name = Path(rel_path).parts[0]
        label = self.class_to_idx[label_name]
        image_path = self.root / rel_path
        image = self.loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label


class SimpleWasteCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 9,
        conv_channels: Tuple[int, int] = (32, 64),
        kernel_sizes: Tuple[int, int] = (3, 3),
        fc_units: int = 256,
        dropout: float = 0.3,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ) -> None:
        super().__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels, conv_channels[0], kernel_size=kernel_sizes[0], padding=1)
        self.conv2 = nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=1)
        self.bn1 = nn.BatchNorm2d(conv_channels[0])
        self.bn2 = nn.BatchNorm2d(conv_channels[1])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)

        feature_dim = conv_channels[1] * (224 // 4) * (224 // 4)
        self.fc1 = nn.Linear(feature_dim, fc_units)
        self.bn_fc = nn.BatchNorm1d(fc_units)
        self.fc_out = nn.Linear(fc_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.activation(self.bn1(self.conv1(x))))
        x = self.pool(self.activation(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(self.activation(self.bn_fc(self.fc1(x))))
        x = self.fc_out(x)
        return x


def build_transform(mean: List[float], std: List[float]) -> T.Compose:
    return T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def make_loader(
    split_samples: Iterable[str],
    transform: Callable,
    class_to_idx: Dict[str, int],
) -> DataLoader:
    dataset = WasteDataset(DATA_ROOT, split_samples, class_to_idx, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    return np.concatenate(all_targets), np.concatenate(all_preds)


def main() -> None:
    manifest = load_split_manifest(SPLIT_MANIFEST)
    class_names: List[str] = manifest["class_names"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    mean, std = load_dataset_stats(STATS_CACHE)
    transform = build_transform(mean, std)

    weights = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model = SimpleWasteCNN(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(weights["model_state_dict"])

    rows = []
    for split in ["val", "test"]:
        loader = make_loader(manifest["splits"][split], transform, class_to_idx)
        targets, preds = evaluate_model(model, loader)
        cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
        rows.append((split, cm))

    for split, cm in rows:
        print(f"{split.upper()} confusion matrix (rows=true, cols=predicted):")
        print(cm)
        print("Class order:", ", ".join(class_names))
        print()


if __name__ == "__main__":
    main()
