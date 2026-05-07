"""Standalone evaluation + decision threshold sweep over an existing checkpoint.

Run after training to get a calibration.json (the one auto-produced by train.py
will be overwritten if you re-run). Useful when you want to re-calibrate against
a different validation slice.

Usage:
    python training/eval_threshold.py \
        --checkpoint ./checkpoints/convnextv2-roadwork/final \
        --image_size 384 \
        --batch_size 64 \
        --val_fraction 0.1 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from data import build_train_val_datasets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
log = logging.getLogger("eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a saved AutoModelForImageClassification dir.")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", type=str, default=None)
    return p.parse_args()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading model from %s on %s", args.checkpoint, device)
    model = AutoModelForImageClassification.from_pretrained(args.checkpoint).to(device).eval()

    _, val_ds = build_train_val_datasets(
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        }

    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False, collate_fn=collate)

    all_probs, all_labels = [], []
    for i, batch in enumerate(loader):
        pv = batch["pixel_values"].to(device, non_blocking=True)
        out = model(pixel_values=pv).logits.float().cpu().numpy()
        probs = softmax_np(out)[:, 1]
        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())
        if i % 10 == 0:
            log.info("Eval batch %d/%d", i, len(loader))

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    log.info("AUC: %.4f", roc_auc_score(labels, probs))

    best = {"threshold": 0.5, "mcc": -1.0, "accuracy": 0.0, "f1": 0.0, "proxy": -1.0}
    for thr in np.linspace(0.05, 0.95, 181):
        preds = (probs >= thr).astype(np.int64)
        if len(np.unique(preds)) > 1 and len(np.unique(labels)) > 1:
            mcc = matthews_corrcoef(labels, preds)
        else:
            mcc = 0.0
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        proxy = 0.5 * mcc + 0.5 * acc
        if proxy > best["proxy"]:
            best = {
                "threshold": float(thr),
                "mcc": float(mcc),
                "accuracy": float(acc),
                "f1": float(f1),
                "proxy": float(proxy),
            }

    log.info("Best: %s", json.dumps(best, indent=2))
    out_path = Path(args.checkpoint) / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
