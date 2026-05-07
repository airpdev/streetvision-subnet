"""Dataset module for StreetVision roadwork classification.

Primary source is `natix-network-org/roadwork` on Hugging Face — the same dataset
the validator samples "real" challenges from. Auxiliary roadwork sources can be
mixed in via the `--extra_datasets` flag.

The dataset is binary:
    label = 1  -> roadwork present (validator's positive class)
    label = 0  -> no roadwork

We expose two PyTorch Datasets:
    - RoadworkHFDataset: random-access wrapper over an HF Dataset split
    - StreamingRoadworkDataset: optional streaming version for huge unions

We also expose a helper `build_train_val_splits()` that does a stratified
90/10 split on label and returns ready-to-train Datasets with augmentations.
"""
from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBU = True
except ImportError:
    _HAS_ALBU = False

from datasets import Dataset as HFDataset
from datasets import concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


PRIMARY_DATASET = "natix-network-org/roadwork"

# Auxiliary sources you can mix in. All are public on HF or convertible to it.
# Each entry is a callable that returns an HF Dataset with at least
# {"image": PIL.Image, "label": 0/1}. Add more as you collect them.
# The lambdas are optional: only loaded when --extra_datasets passes their key.
AUX_DATASET_LOADERS: Dict[str, Callable[[], HFDataset]] = {
    # Example: a pre-curated HF mirror (replace with a real one you have access to)
    # "rdd2022": lambda: load_dataset("user/rdd2022-binary-roadwork", split="train"),
    # "bdd100k_construction": lambda: load_dataset("user/bdd100k-roadwork", split="train"),
}


def _to_pil_rgb(value: Any) -> Optional[Image.Image]:
    """Coerce common image fields to a RGB PIL Image."""
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        for k in ("bytes", "image", "image_bytes"):
            if k in value and value[k] is not None:
                return _to_pil_rgb(value[k])
        return None
    if isinstance(value, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(value)).convert("RGB")
        except Exception as e:
            logger.debug("Failed to decode bytes image: %s", e)
            return None
    return None


def _extract_label(example: Dict[str, Any]) -> Optional[int]:
    """Pull a binary label from common label fields. Falls back to scene_description."""
    if "label" in example and example["label"] is not None:
        try:
            return int(example["label"])
        except Exception:
            pass
    if "labels" in example and example["labels"] is not None:
        try:
            return int(example["labels"])
        except Exception:
            pass
    sd = example.get("scene_description")
    if isinstance(sd, str) and sd.strip():
        return 1
    if sd is None or sd == "":
        return 0
    return None


def load_primary_dataset(
    repo_id: str = PRIMARY_DATASET,
    split: str = "train",
    cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load the canonical roadwork dataset from HuggingFace."""
    logger.info("Loading primary dataset %s split=%s", repo_id, split)
    ds = load_dataset(repo_id, split=split, cache_dir=cache_dir)
    return ds


def load_aux_datasets(keys: Sequence[str]) -> List[HFDataset]:
    out: List[HFDataset] = []
    for k in keys:
        loader = AUX_DATASET_LOADERS.get(k)
        if loader is None:
            logger.warning("Unknown auxiliary dataset key: %s (skipping)", k)
            continue
        try:
            ds = loader()
            logger.info("Loaded aux dataset %s with %d rows", k, len(ds))
            out.append(ds)
        except Exception as e:
            logger.warning("Failed to load aux dataset %s: %s", k, e)
    return out


def build_default_train_transform(image_size: int = 384) -> Callable[[Image.Image], torch.Tensor]:
    """Strong augmentation pipeline: better generalization to validator's synthetic+API images."""
    if not _HAS_ALBU:
        raise RuntimeError(
            "albumentations is required. Install with `pip install albumentations`."
        )

    train_tf = A.Compose([
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(min_height=int(image_size * 1.15), min_width=int(image_size * 1.15),
                      border_mode=0, value=(0, 0, 0)),
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=5),
        ], p=0.25),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        ], p=0.5),
        A.ImageCompression(quality_range=(60, 95), p=0.4),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.25),
        A.CoarseDropout(num_holes_range=(1, 6),
                        hole_height_range=(8, int(image_size * 0.15)),
                        hole_width_range=(8, int(image_size * 0.15)),
                        fill=0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def fn(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB"))
        return train_tf(image=arr)["image"]

    return fn


def build_default_eval_transform(image_size: int = 384) -> Callable[[Image.Image], torch.Tensor]:
    if not _HAS_ALBU:
        raise RuntimeError("albumentations is required.")
    eval_tf = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, value=(0, 0, 0)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    def fn(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.convert("RGB"))
        return eval_tf(image=arr)["image"]

    return fn


@dataclass
class RoadworkHFDataset(Dataset):
    """Random-access dataset wrapping an HF Dataset.

    Yields dicts compatible with HuggingFace `Trainer` collator:
        {"pixel_values": Tensor[3, H, W], "labels": int}
    """
    hf_dataset: HFDataset
    transform: Callable[[Image.Image], torch.Tensor]
    drop_unlabeled: bool = True

    def __post_init__(self) -> None:
        if self.drop_unlabeled:
            keep_idx: List[int] = []
            for i in range(len(self.hf_dataset)):
                lbl = _extract_label(self.hf_dataset[i])
                if lbl in (0, 1):
                    keep_idx.append(i)
            if len(keep_idx) != len(self.hf_dataset):
                logger.info("Dropped %d unlabeled rows (kept %d)",
                            len(self.hf_dataset) - len(keep_idx), len(keep_idx))
                self.hf_dataset = self.hf_dataset.select(keep_idx)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.hf_dataset[int(idx)]
        img = _to_pil_rgb(ex.get("image"))
        if img is None and "image_url" in ex:
            from urllib.request import urlopen
            try:
                with urlopen(ex["image_url"], timeout=10) as r:
                    img = Image.open(io.BytesIO(r.read())).convert("RGB")
            except Exception as e:
                logger.warning("Failed to fetch image_url %s: %s", ex.get("image_url"), e)
        if img is None:
            raise RuntimeError(f"No image found at row {idx}: keys={list(ex.keys())}")

        label = _extract_label(ex)
        if label is None:
            raise RuntimeError(f"No label found at row {idx}: keys={list(ex.keys())}")

        return {
            "pixel_values": self.transform(img),
            "labels": int(label),
        }


def stratified_train_val_split(
    ds: HFDataset,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[HFDataset, HFDataset]:
    """Pure-Python stratified split on the binary label.

    Avoids the surprises of HF's `train_test_split(stratify_by_column=...)`
    when the label column has odd dtypes.
    """
    rng = random.Random(seed)
    pos_idx: List[int] = []
    neg_idx: List[int] = []
    for i in range(len(ds)):
        lbl = _extract_label(ds[i])
        if lbl == 1:
            pos_idx.append(i)
        elif lbl == 0:
            neg_idx.append(i)

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_val = max(1, int(len(pos_idx) * val_fraction))
    n_neg_val = max(1, int(len(neg_idx) * val_fraction))

    val_idx = pos_idx[:n_pos_val] + neg_idx[:n_neg_val]
    train_idx = pos_idx[n_pos_val:] + neg_idx[n_neg_val:]
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    train_ds = ds.select(train_idx)
    val_ds = ds.select(val_idx)

    logger.info(
        "Stratified split: train=%d (pos=%d, neg=%d) | val=%d (pos=%d, neg=%d)",
        len(train_ds), len(pos_idx) - n_pos_val, len(neg_idx) - n_neg_val,
        len(val_ds), n_pos_val, n_neg_val,
    )
    return train_ds, val_ds


def build_class_weighted_sampler(ds: RoadworkHFDataset) -> torch.utils.data.WeightedRandomSampler:
    """Balanced sampler so the minority class isn't drowned out."""
    labels: List[int] = []
    for i in range(len(ds.hf_dataset)):
        lbl = _extract_label(ds.hf_dataset[i])
        labels.append(int(lbl))
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    weights_per_class = 1.0 / counts
    sample_weights = np.array([weights_per_class[l] for l in labels], dtype=np.float64)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_train_val_datasets(
    image_size: int = 384,
    val_fraction: float = 0.1,
    seed: int = 42,
    extra_dataset_keys: Iterable[str] = (),
    cache_dir: Optional[str] = None,
) -> Tuple[RoadworkHFDataset, RoadworkHFDataset]:
    primary = load_primary_dataset(cache_dir=cache_dir)
    extras = load_aux_datasets(list(extra_dataset_keys))

    full = primary
    if extras:
        try:
            full = concatenate_datasets([primary] + extras)
            logger.info("Combined dataset size: %d", len(full))
        except Exception as e:
            logger.warning("concatenate_datasets failed (%s); using primary only", e)

    train_split, val_split = stratified_train_val_split(full, val_fraction=val_fraction, seed=seed)

    train_tf = build_default_train_transform(image_size=image_size)
    eval_tf = build_default_eval_transform(image_size=image_size)
    train_ds = RoadworkHFDataset(hf_dataset=train_split, transform=train_tf)
    val_ds = RoadworkHFDataset(hf_dataset=val_split, transform=eval_tf)
    return train_ds, val_ds
