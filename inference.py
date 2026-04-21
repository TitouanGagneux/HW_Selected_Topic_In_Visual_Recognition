"""Run DETR inference on a test folder and export COCO-style predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import DetrConfig, DetrForObjectDetection, DetrImageProcessor


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DETR inference and export a COCO-style JSON file."
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
        "--test-dir",
        type=str,
        default=None,
        help="Path to test image folder. Defaults to <data-dir>/test.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/content/drive/MyDrive/best_detr_model.pth",
        help="Path to the best model checkpoint or raw state_dict.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="/content/drive/MyDrive/pred.json",
        help="Path to the output prediction JSON file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/detr-resnet-50",
        help="Hugging Face model/config identifier.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.01,
        help="Detection score threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device to use, e.g. "cuda" or "cpu". Defaults to auto.',
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_model(
    model_name: str,
    num_dataset_classes: int,
    device: torch.device,
) -> DetrForObjectDetection:
    """Rebuild the model with the correct class count."""
    config = DetrConfig.from_pretrained(model_name)
    config.num_labels = num_dataset_classes + 1
    model = DetrForObjectDetection(config)
    model.to(device)
    return model


def load_checkpoint(
    model: DetrForObjectDetection,
    model_path: Path,
    device: torch.device,
) -> None:
    """Load a checkpoint into the model."""
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", "?")
        best_valid_loss = checkpoint.get("best_valid_loss", "?")
        print(
            "Checkpoint loaded — "
            f"epoch {epoch}, valid loss: {best_valid_loss}"
        )
    else:
        state_dict = checkpoint
        print("Raw state_dict checkpoint loaded")

    model.load_state_dict(state_dict, strict=True)


def infer_image_id(filename_stem: str) -> Union[int, str]:
    """Infer the image_id from the filename stem."""
    return int(filename_stem) if filename_stem.isdigit() else filename_stem


def main() -> None:
    """Entry point."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    train_json_path = Path(args.train_json) if args.train_json else data_dir / "train.json"
    test_dir = Path(args.test_dir) if args.test_dir else data_dir / "test"
    model_path = Path(args.model_path)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    train_coco = load_json(train_json_path)
    num_dataset_classes = len(train_coco["categories"])

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model = build_model(args.model_name, num_dataset_classes, device)
    load_checkpoint(model, model_path, device)
    model.eval()

    processor = DetrImageProcessor.from_pretrained(args.model_name)

    test_images = sorted(
        file_name
        for file_name in os.listdir(test_dir)
        if file_name.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    predictions: List[Dict[str, Any]] = []

    with torch.no_grad():
        for file_name in tqdm(test_images, desc="Inference"):
            stem = Path(file_name).stem
            image_id = infer_image_id(stem)

            image = Image.open(test_dir / file_name).convert("RGB")
            orig_w, orig_h = image.size

            inputs = processor(images=image, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)

            target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
            results = processor.post_process_object_detection(
                outputs,
                threshold=args.score_threshold,
                target_sizes=target_sizes,
            )[0]

            scores = results["scores"].cpu().tolist()
            labels = results["labels"].cpu().tolist()
            boxes = results["boxes"].cpu().tolist()

            for score, label, box in zip(scores, labels, boxes):
                x0, y0, x1, y1 = box
                width = x1 - x0
                height = y1 - y0

                predictions.append(
                    {
                        "image_id": image_id,
                        "bbox": [x0, y0, width, height],
                        "score": score,
                        "category_id": label,
                    }
                )

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(predictions, file)

    print(f"Done — {len(predictions)} predictions saved to {output_json}")


if __name__ == "__main__":
    main()
