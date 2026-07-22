from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


CLASS_BACKGROUND = 0
CLASS_PART = 1
CLASS_BIN = 2
CLASS_NAMES = ("background", "part", "bin")


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TinyUNet(nn.Module):
    """Small semantic segmenter for background, loose parts, and bins."""

    def __init__(self, base_channels: int = 16, classes: int = 3):
        super().__init__()
        self.down1 = ConvBlock(3, base_channels)
        self.down2 = ConvBlock(base_channels, base_channels * 2)
        self.bridge = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.decode2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.decode1 = ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.down1(x)
        second = self.down2(F.max_pool2d(first, 2))
        bridge = self.bridge(F.max_pool2d(second, 2))
        decoded2 = self.decode2(torch.cat([self.up2(bridge), second], dim=1))
        decoded1 = self.decode1(torch.cat([self.up1(decoded2), first], dim=1))
        return self.head(decoded1)


class VisualDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        image_size: int = 128,
        augment: bool = False,
    ):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format_version", 0) < 2:
            raise ValueError("visual training requires dataset format version 2+")
        self.records = self.manifest["records"]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        with np.load(self.root / record["file"]) as sample:
            rgb = sample["rgb"].copy()
            part_union = sample["part_masks"].any(axis=0)
            bin_union = sample["bin_masks"].any(axis=0)

        target = np.full(rgb.shape[:2], CLASS_BACKGROUND, dtype=np.uint8)
        target[bin_union] = CLASS_BIN
        target[part_union] = CLASS_PART
        rgb = cv2.resize(
            rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        target = cv2.resize(
            target,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST,
        )
        if self.augment:
            rgb = augment_rgb(rgb)
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()
        image = image / 255.0
        return {
            "image": image,
            "target": torch.from_numpy(target.astype(np.int64)),
            "file": record["file"],
        }


def augment_rgb(rgb: np.ndarray) -> np.ndarray:
    """Sensor-style augmentation that preserves relational object colors."""
    rng = np.random.default_rng()
    image = rgb.astype(np.float32) / 255.0
    image = (image - 0.5) * float(rng.uniform(0.75, 1.30)) + 0.5
    image *= float(rng.uniform(0.72, 1.28))
    image = np.power(np.clip(image, 0.0, 1.0), float(rng.uniform(0.75, 1.35)))
    channel_gain = rng.uniform(0.88, 1.12, size=(1, 1, 3))
    image *= channel_gain
    if rng.random() < 0.35:
        kernel = int(rng.choice([3, 5]))
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    if rng.random() < 0.65:
        image += rng.normal(0.0, rng.uniform(0.003, 0.025), image.shape)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor([0.18, 3.0, 1.2], device=logits.device)
    cross_entropy = F.cross_entropy(logits, target, weight=weights)
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, len(CLASS_NAMES)).permute(0, 3, 1, 2).float()
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum((0, 2, 3))
    denominator = (probabilities[:, 1:] + one_hot[:, 1:]).sum((0, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return cross_entropy + dice_loss


def confusion_matrix(
    logits: torch.Tensor, target: torch.Tensor, classes: int = 3
) -> torch.Tensor:
    prediction = logits.argmax(dim=1)
    indices = target.flatten() * classes + prediction.flatten()
    return torch.bincount(indices, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, float]:
    matrix = matrix.double()
    true_positive = matrix.diag()
    union = matrix.sum(0) + matrix.sum(1) - true_positive
    iou = true_positive / union.clamp_min(1.0)
    accuracy = true_positive.sum() / matrix.sum().clamp_min(1.0)
    return {
        "pixel_accuracy": float(accuracy),
        "part_iou": float(iou[CLASS_PART]),
        "bin_iou": float(iou[CLASS_BIN]),
        "mean_foreground_iou": float(iou[1:].mean()),
    }


@dataclass(frozen=True)
class CameraCalibration:
    width: int = 240
    height: int = 240
    camera_x: float = 0.56
    camera_y: float = 0.0
    camera_z: float = 1.75
    fovy_degrees: float = 52.0
    workspace_plane_z: float = 0.52

    def pixel_to_world(self, centroid_xy: tuple[float, float]) -> np.ndarray:
        u, v = centroid_xy
        focal_pixels = 0.5 * self.height / math.tan(
            math.radians(self.fovy_degrees) / 2.0
        )
        distance = self.camera_z - self.workspace_plane_z
        return np.array(
            [
                self.camera_x + (u - (self.width - 1) / 2.0) * distance / focal_pixels,
                self.camera_y - (v - (self.height - 1) / 2.0) * distance / focal_pixels,
                self.workspace_plane_z,
            ],
            dtype=np.float32,
        )


@dataclass
class VisualInstance:
    kind: str
    mask: np.ndarray
    centroid_xy: tuple[float, float]
    world_position: np.ndarray
    mean_rgb: np.ndarray
    confidence: float


@dataclass
class VisualPick:
    part: VisualInstance
    target_bin: VisualInstance
    color_distance: float
    color_margin: float


@dataclass
class VisualTaskEstimate:
    parts: list[VisualInstance]
    bins: list[VisualInstance]
    picks: list[VisualPick]
    semantic_mask: np.ndarray


class RGBVisualPolicy:
    """RGB-only detector plus label-free, relational color matcher."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "auto",
        calibration: CameraCalibration | None = None,
    ):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.image_size = int(payload["config"]["image_size"])
        self.model = TinyUNet(base_channels=int(payload["config"]["base_channels"]))
        self.model.load_state_dict(payload["model_state"])
        self.device = choose_device(device)
        self.model.to(self.device).eval()
        self.calibration = calibration or CameraCalibration()

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> VisualTaskEstimate:
        resized = cv2.resize(
            rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )
        tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float()
        logits = self.model((tensor / 255.0).unsqueeze(0).to(self.device))
        probabilities = logits.softmax(dim=1)[0]
        probabilities = F.interpolate(
            probabilities.unsqueeze(0),
            size=rgb.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0]
        semantic = probabilities.argmax(dim=0).cpu().numpy().astype(np.uint8)
        confidence_map = probabilities.max(dim=0).values.cpu().numpy()
        parts = extract_instances(
            rgb,
            semantic == CLASS_PART,
            confidence_map,
            "part",
            self.calibration,
            minimum_area=18,
        )
        bins = extract_instances(
            rgb,
            semantic == CLASS_BIN,
            confidence_map,
            "bin",
            self.calibration,
            minimum_area=250,
        )
        # After a drop, the object remains visible inside its destination bin.
        # Exclude those regions so closed-loop replanning only returns unsorted
        # feed objects instead of attempting to pick already sorted pieces.
        if bins:
            bin_regions = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for instance in bins:
                contours, _ = cv2.findContours(
                    instance.mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in contours:
                    x, y, width, height = cv2.boundingRect(contour)
                    bin_regions[y : y + height, x : x + width] = 1
            parts = [
                instance
                for instance in parts
                if not bin_regions[
                    int(round(instance.centroid_xy[1])),
                    int(round(instance.centroid_xy[0])),
                ]
            ]
        picks: list[VisualPick] = []
        if len(bins) == 2:
            for part in parts:
                distances = np.array(
                    [
                        relational_color_distance(part.mean_rgb, instance.mean_rgb)
                        for instance in bins
                    ]
                )
                target_index = int(np.argmin(distances))
                ordered = np.sort(distances)
                picks.append(
                    VisualPick(
                        part,
                        bins[target_index],
                        float(distances[target_index]),
                        float(ordered[1] - ordered[0]),
                    )
                )
        return VisualTaskEstimate(parts, bins, picks, semantic)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_instances(
    rgb: np.ndarray,
    binary_mask: np.ndarray,
    confidence_map: np.ndarray,
    kind: str,
    calibration: CameraCalibration,
    *,
    minimum_area: int,
) -> list[VisualInstance]:
    mask_u8 = binary_mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Closing is useful for the five-piece bin silhouette, but can merge two
    # neighboring loose parts into one instance in denser scenes.
    if kind == "bin":
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, 8)
    instances: list[VisualInstance] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        component = labels == label
        color_mask = cv2.erode(component.astype(np.uint8), kernel, iterations=1).astype(bool)
        if not color_mask.any():
            color_mask = component
        pixels = rgb[color_mask]
        mean_rgb = np.median(pixels, axis=0).astype(np.float32)
        centroid = (float(centroids[label, 0]), float(centroids[label, 1]))
        instances.append(
            VisualInstance(
                kind=kind,
                mask=component,
                centroid_xy=centroid,
                world_position=calibration.pixel_to_world(centroid),
                mean_rgb=mean_rgb,
                confidence=float(confidence_map[component].mean()),
            )
        )
    return sorted(instances, key=lambda item: item.centroid_xy[0])


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    pixel = np.clip(rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


def relational_color_distance(first_rgb: np.ndarray, second_rgb: np.ndarray) -> float:
    """Compare chromaticity while discounting illumination-driven brightness."""
    first = np.maximum(first_rgb.astype(np.float32), 1.0)
    second = np.maximum(second_rgb.astype(np.float32), 1.0)
    first /= float(first.sum())
    second /= float(second.sum())
    return float(np.linalg.norm(first - second))


def colorize_semantic(mask: np.ndarray) -> np.ndarray:
    palette = np.array([[0, 0, 0], [60, 220, 80], [70, 130, 255]], dtype=np.uint8)
    return palette[mask]


def draw_task_estimate(rgb: np.ndarray, estimate: VisualTaskEstimate) -> np.ndarray:
    """Return an RGB diagnostic overlay of instances and proposed matches."""
    canvas = rgb.copy()
    for index, instance in enumerate(estimate.bins):
        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (70, 130, 255), 2)
        point = tuple(int(value) for value in instance.centroid_xy)
        cv2.putText(
            canvas, f"B{index}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 130, 255), 2
        )
    for index, instance in enumerate(estimate.parts):
        contours, _ = cv2.findContours(
            instance.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, (60, 220, 80), 2)
        point = tuple(int(value) for value in instance.centroid_xy)
        cv2.putText(
            canvas, f"P{index}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 220, 80), 2
        )
    for pick in estimate.picks:
        start = tuple(int(value) for value in pick.part.centroid_xy)
        end = tuple(int(value) for value in pick.target_bin.centroid_xy)
        cv2.arrowedLine(canvas, start, end, (240, 60, 50), 1, tipLength=0.08)
    return canvas
