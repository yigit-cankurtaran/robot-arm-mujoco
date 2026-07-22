from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from visual_policy import (
    TinyUNet,
    VisualDataset,
    choose_device,
    colorize_semantic,
    confusion_matrix,
    metrics_from_confusion,
    segmentation_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RGB part/bin perception.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/visual_policy"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="optional compatible checkpoint for fine-tuning",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def run_epoch(
    model: TinyUNet,
    loader: DataLoader,
    device: torch.device,
    optimizer: AdamW | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    matrix = torch.zeros((3, 3), dtype=torch.int64)
    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = segmentation_loss(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_items += batch_size
        matrix += confusion_matrix(logits.detach().cpu(), targets.detach().cpu())
    metrics = metrics_from_confusion(matrix)
    metrics["loss"] = total_loss / max(total_items, 1)
    return metrics


@torch.inference_mode()
def write_preview(
    model: TinyUNet,
    dataset: VisualDataset,
    device: torch.device,
    path: Path,
) -> None:
    panels = []
    for index in np.linspace(0, len(dataset) - 1, min(6, len(dataset)), dtype=int):
        item = dataset[int(index)]
        image = item["image"].numpy().transpose(1, 2, 0)
        target = item["target"].numpy()
        logits = model(item["image"].unsqueeze(0).to(device))
        prediction = logits.argmax(1)[0].cpu().numpy()
        rgb = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        target_overlay = cv2.addWeighted(rgb, 0.65, colorize_semantic(target), 0.35, 0)
        prediction_overlay = cv2.addWeighted(
            rgb, 0.65, colorize_semantic(prediction), 0.35, 0
        )
        panels.append(np.concatenate([rgb, target_overlay, prediction_overlay], axis=1))
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(panels, axis=0), cv2.COLOR_RGB2BGR))


def write_curves(history: list[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="validation")
    axes[0].set_title("Segmentation loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["val_part_iou"] for row in history], label="part IoU")
    axes[1].plot(epochs, [row["val_bin_iou"] for row in history], label="bin IoU")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Held-out IoU")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    train_dataset = VisualDataset(
        args.train_data, image_size=args.image_size, augment=True
    )
    val_dataset = VisualDataset(args.val_data, image_size=args.image_size, augment=False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    model = TinyUNet(args.base_channels).to(device)
    if args.initial_checkpoint is not None:
        initial_payload = torch.load(
            args.initial_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(initial_payload["model_state"])
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    config = {
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "train_data": str(args.train_data),
        "val_data": str(args.val_data),
        "seed": args.seed,
        "initial_checkpoint": (
            str(args.initial_checkpoint) if args.initial_checkpoint is not None else None
        ),
    }
    history: list[dict[str, float]] = []
    best_iou = -1.0
    started = time.perf_counter()
    print(json.dumps({"device": str(device), "config": config}, indent=2))
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, None)
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row))
        payload = {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(payload, args.output / "last.pt")
        if val_metrics["mean_foreground_iou"] > best_iou:
            best_iou = val_metrics["mean_foreground_iou"]
            torch.save(payload, args.output / "best.pt")
            write_preview(model, val_dataset, device, args.output / "best_preview.png")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        with (args.output / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        write_curves(history, args.output / "training_curves.png")

    summary = {
        "best_mean_foreground_iou": best_iou,
        "epochs": args.epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output / "best.pt"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
