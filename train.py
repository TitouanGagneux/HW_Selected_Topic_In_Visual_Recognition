"""Train a DETR object detector on a COCO-style dataset.

This script is a cleaned, PEP 8-compliant version of the provided notebook.
It keeps the same default paths and main outputs:
- best checkpoint
- final model weights
- training/validation loss plots
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.amp as amp
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DETR on a COCO-style dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/content/nycu-hw2-data",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default=None,
        help="Path to train.json. Defaults to <data-dir>/train.json.",
    )
    parser.add_argument(
        "--valid-json",
        type=str,
        default=None,
        help="Path to valid.json. Defaults to <data-dir>/valid.json.",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        default=None,
        help="Path to training images. Defaults to <data-dir>/train.",
    )
    parser.add_argument(
        "--valid-dir",
        type=str,
        default=None,
        help="Path to validation images. Defaults to <data-dir>/valid.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive",
        help="Directory where checkpoints, weights, and plots are saved.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/detr-resnet-50",
        help="Hugging Face model/config identifier.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for non-backbone parameters.",
    )
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-5,
        help="Learning rate for backbone parameters.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="StepLR step size.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="StepLR gamma.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.1,
        help="Gradient clipping max norm.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use, e.g. "cuda" or "cpu". Defaults to auto.',
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """Resolve dataset and output paths."""
    data_dir = Path(args.data_dir)
    paths = {
        "data_dir": data_dir,
        "train_json": Path(args.train_json) if args.train_json else data_dir / "train.json",
        "valid_json": Path(args.valid_json) if args.valid_json else data_dir / "valid.json",
        "train_dir": Path(args.train_dir) if args.train_dir else data_dir / "train",
        "valid_dir": Path(args.valid_dir) if args.valid_dir else data_dir / "valid",
        "output_dir": Path(args.output_dir),
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def is_finite_bbox(values: List[float]) -> bool:
    """Return True if all bbox values are finite."""
    return all(not math.isnan(v) and not math.isinf(v) for v in values)


class CocoDataset(Dataset):
    """COCO-style dataset with bbox sanitization."""

    def __init__(self, img_dir: Path, coco: Dict[str, Any]) -> None:
        self.img_dir = img_dir
        self.coco = coco

        valid_image_ids = set()
        for img_info in coco["images"]:
            path = img_dir / img_info["file_name"]
            if path.exists():
                valid_image_ids.add(img_info["id"])

        filtered_annotations = []
        for ann in coco["annotations"]:
            if ann["image_id"] not in valid_image_ids:
                continue

            x, y, w, h = ann["bbox"]
            if (
                w > 1
                and h > 1
                and x >= 0
                and y >= 0
                and is_finite_bbox([x, y, w, h])
            ):
                filtered_annotations.append(ann)

        self.img_to_anns: Dict[Any, List[Dict[str, Any]]] = {}
        for ann in filtered_annotations:
            self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.images = [
            img
            for img in coco["images"]
            if img["id"] in valid_image_ids and img["id"] in self.img_to_anns
        ]

        print(f"Dataset: {len(self.images)} images with valid annotations")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        img_info = self.images[idx]
        path = self.img_dir / img_info["file_name"]
        image = Image.open(path).convert("RGB")
        img_w, img_h = image.size

        anns = self.img_to_anns.get(img_info["id"], [])
        safe_anns = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x = max(0.0, min(x, img_w - 1))
            y = max(0.0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if w > 1 and h > 1:
                new_ann = dict(ann)
                new_ann["bbox"] = [x, y, w, h]
                safe_anns.append(new_ann)

        if not safe_anns:
            return self.__getitem__((idx + 1) % len(self.images))

        target = {"image_id": img_info["id"], "annotations": safe_anns}
        return image, target


def make_collate_fn(
    processor: DetrImageProcessor,
):
    """Create a collate function using the DETR image processor."""

    def collate_fn_with_processor(
        batch: List[Tuple[Image.Image, Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]

        valid_pairs = [
            (img, tgt) for img, tgt in zip(images, targets) if tgt["annotations"]
        ]
        if not valid_pairs:
            return None

        images, targets = zip(*valid_pairs)

        try:
            encoding = processor(
                images=list(images),
                annotations=list(targets),
                return_tensors="pt",
            )
        except Exception as exc:  # pragma: no cover - defensive runtime safeguard
            print(f"[collate_fn] processor error: {exc} — batch skipped")
            return None

        safe_labels = []
        for label in encoding["labels"]:
            safe_label = {}
            for key, value in label.items():
                if value.is_floating_point():
                    value = value.float()
                    value = torch.nan_to_num(
                        value,
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                safe_label[key] = value
            safe_labels.append(safe_label)

        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": safe_labels,
        }

    return collate_fn_with_processor


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_dataloaders(
    train_coco: Dict[str, Any],
    valid_coco: Dict[str, Any],
    paths: Dict[str, Path],
    processor: DetrImageProcessor,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """Build training and validation dataloaders."""
    train_dataset = CocoDataset(paths["train_dir"], train_coco)
    valid_dataset = CocoDataset(paths["valid_dir"], valid_coco)

    collate_fn = make_collate_fn(processor)

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    return train_loader, valid_loader


def build_model(
    model_name: str,
    num_dataset_classes: int,
    device: torch.device,
) -> DetrForObjectDetection:
    """Build a DETR model from config, without pretrained weights."""
    config = DetrConfig.from_pretrained(model_name)
    config.num_labels = num_dataset_classes + 1
    model = DetrForObjectDetection(config)
    model.to(device)
    return model


def has_non_finite_boxes(labels: List[Dict[str, torch.Tensor]]) -> bool:
    """Return True if any target contains non-finite boxes."""
    for target in labels:
        if not torch.isfinite(target["boxes"]).all():
            return True
    return False


def save_loss_plots(
    train_losses: List[float],
    valid_losses: List[float],
    output_dir: Path,
) -> None:
    """Save train/validation loss plots."""
    plt.figure()
    plt.plot(train_losses)
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(valid_losses)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_dir / "valid_loss.png")
    plt.close()


def train_one_epoch(
    model: DetrForObjectDetection,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: amp.GradScaler,
    device: torch.device,
    grad_clip_norm: float,
) -> Tuple[float, int]:
    """Run one training epoch and return total loss and valid step count."""
    model.train()
    total_train_loss = 0.0
    valid_steps = 0

    use_amp = device.type == "cuda"

    for batch in tqdm(loader, desc="Training", leave=False):
        if batch is None:
            continue

        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [
            {key: value.to(device) for key, value in target.items()}
            for target in batch["labels"]
        ]

        if has_non_finite_boxes(labels):
            print("Warning: non-finite boxes detected after collate, batch skipped")
            continue

        optimizer.zero_grad()

        with amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            loss = outputs.loss

        if not torch.isfinite(loss):
            print(f"Warning: non-finite loss ({loss.item()}), batch skipped")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += float(loss.item())
        valid_steps += 1

    return total_train_loss, valid_steps


def validate_one_epoch(
    model: DetrForObjectDetection,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, int]:
    """Run one validation epoch and return total loss and valid step count."""
    model.eval()
    total_valid_loss = 0.0
    valid_steps = 0

    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [
                {key: value.to(device) for key, value in target.items()}
                for target in batch["labels"]
            ]

            if has_non_finite_boxes(labels):
                continue

            with amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )
                loss = outputs.loss

            if not torch.isfinite(loss):
                continue

            total_valid_loss += float(loss.item())
            valid_steps += 1

    return total_valid_loss, valid_steps


def main() -> None:
    """Entry point."""
    args = parse_args()
    paths = resolve_paths(args)

    train_coco = load_json(paths["train_json"])
    valid_coco = load_json(paths["valid_json"])

    processor = DetrImageProcessor.from_pretrained(
        args.model_name,
        do_random_flip=True,
    )

    train_loader, valid_loader = build_dataloaders(
        train_coco=train_coco,
        valid_coco=valid_coco,
        paths=paths,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    num_dataset_classes = len(train_coco["categories"])
    model = build_model(args.model_name, num_dataset_classes, device)

    param_dicts = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if "backbone" not in name and param.requires_grad
            ]
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if "backbone" in name and param.requires_grad
            ],
            "lr": args.backbone_lr,
        },
    ]
    optimizer = AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    scaler = amp.GradScaler(enabled=device.type == "cuda")
    train_losses: List[float] = []
    valid_losses: List[float] = []

    best_valid_loss = float("inf")
    best_epoch = -1

    best_model_path = paths["output_dir"] / "best_detr_model.pth"
    final_model_path = paths["output_dir"] / "final_detr_model.pth"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_total_loss, train_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip_norm=args.grad_clip_norm,
        )

        if train_steps == 0:
            print(f"Warning: epoch {epoch + 1} had no valid training batch")
            train_losses.append(float("nan"))
        else:
            avg_train_loss = train_total_loss / train_steps
            train_losses.append(avg_train_loss)
            scheduler.step()

        valid_total_loss, valid_steps = validate_one_epoch(
            model=model,
            loader=valid_loader,
            device=device,
        )

        avg_valid_loss = valid_total_loss / max(valid_steps, 1)
        valid_losses.append(avg_valid_loss)

        current_lr = scheduler.get_last_lr()[0]
        avg_train_loss = train_losses[-1]
        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
            f"Valid Loss: {avg_valid_loss:.4f} | LR: {current_lr:.1e}"
        )

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_valid_loss": best_valid_loss,
                },
                best_model_path,
            )
            print(
                f"Best model saved at epoch {best_epoch} "
                f"(valid loss: {best_valid_loss:.4f})"
            )

    torch.save(model.state_dict(), final_model_path)
    print(f"Final model state saved to {final_model_path}")

    save_loss_plots(train_losses, valid_losses, paths["output_dir"])

    history = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
    }
    with (paths["output_dir"] / "training_history.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(history, file, indent=2)


if __name__ == "__main__":
    main()
