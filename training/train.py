"""Fine-tune a vision backbone for binary roadwork classification.

Optimized for the StreetVision validator's reward (rolling MCC@100 + Accuracy@10).
Default backbone is ConvNeXt-V2-Large @ 384px (loadable as
`AutoModelForImageClassification`, which means the trained checkpoint is a
drop-in replacement for the existing `ViTImageDetector` — just point your
`ViT_roadwork.yaml` `hf_repo` at the saved directory).

Usage (single GPU, sanity test):
    python training/train.py --epochs 3 --batch_size 16

Multi-GPU (2x A100 40GB):
    accelerate launch --num_processes 2 \
        training/train.py \
        --output_dir ./checkpoints/convnextv2-roadwork \
        --backbone facebook/convnextv2-large-22k-384 \
        --image_size 384 \
        --batch_size 32 \
        --grad_accum 1 \
        --epochs 12 \
        --lr_backbone 1e-5 \
        --lr_head 1e-3 \
        --weight_decay 0.05 \
        --warmup_ratio 0.06 \
        --label_smoothing 0.05 \
        --bf16

Resume / fine-tune from a previous checkpoint:
    accelerate launch ... training/train.py \
        --resume_from_checkpoint ./checkpoints/convnextv2-roadwork/last
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

# Local
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from data import (  # noqa: E402
    build_class_weighted_sampler,
    build_train_val_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train")


# ---------- args ----------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    p.add_argument("--cache_dir", type=str, default=None,
                   help="HuggingFace datasets cache dir.")
    p.add_argument("--image_size", type=int, default=384)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--extra_datasets", type=str, nargs="*", default=[],
                   help="Optional aux dataset keys defined in data.AUX_DATASET_LOADERS")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)

    # model
    p.add_argument("--backbone", type=str,
                   default="facebook/convnextv2-large-22k-384",
                   help="HF model id loadable via AutoModelForImageClassification.")
    p.add_argument("--num_labels", type=int, default=2)

    # training
    p.add_argument("--output_dir", type=str, default="./checkpoints/roadwork")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=32, help="Per-device batch size.")
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use_class_balanced_sampler", action="store_true",
                   help="If set, use a WeightedRandomSampler instead of label smoothing-only.")
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)

    # tracking
    p.add_argument("--report_to", type=str, default="none",
                   choices=["none", "wandb", "tensorboard"])
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


# ---------- model ----------

def build_model(backbone: str, num_labels: int = 2) -> nn.Module:
    """Load backbone with `ignore_mismatched_sizes=True` so the head is rebuilt fresh."""
    logger.info("Loading backbone: %s", backbone)
    model = AutoModelForImageClassification.from_pretrained(
        backbone,
        num_labels=num_labels,
        id2label={0: "None", 1: "Roadwork"},
        label2id={"None": 0, "Roadwork": 1},
        ignore_mismatched_sizes=True,
    )
    return model


def split_param_groups(model: nn.Module, lr_backbone: float, lr_head: float,
                       weight_decay: float) -> List[Dict[str, Any]]:
    """Two LR groups: smaller LR on backbone, larger on the new classification head."""
    head_keywords = ("classifier", "head", "score", "fc")
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight",
                         "norm.weight", "layernorm.weight")

    groups = {
        "backbone_decay": {"params": [], "lr": lr_backbone, "weight_decay": weight_decay},
        "backbone_nodecay": {"params": [], "lr": lr_backbone, "weight_decay": 0.0},
        "head_decay": {"params": [], "lr": lr_head, "weight_decay": weight_decay},
        "head_nodecay": {"params": [], "lr": lr_head, "weight_decay": 0.0},
    }

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_head = any(k in name for k in head_keywords)
        is_no_decay = any(k in name for k in no_decay_keywords)
        if is_head and is_no_decay:
            groups["head_nodecay"]["params"].append(p)
        elif is_head:
            groups["head_decay"]["params"].append(p)
        elif is_no_decay:
            groups["backbone_nodecay"]["params"].append(p)
        else:
            groups["backbone_decay"]["params"].append(p)

    out = [g for g in groups.values() if g["params"]]
    for g in out:
        logger.info("Param group lr=%g wd=%g size=%d",
                    g["lr"], g["weight_decay"], len(g["params"]))
    return out


# ---------- metrics ----------

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=-1, keepdims=True)


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Validator-aligned metrics. Primary reward signal is MCC; we also report Acc/F1/AUC."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = softmax_np(np.asarray(logits, dtype=np.float64))[:, 1]
    preds = (probs >= 0.5).astype(np.int64)
    labels = np.asarray(labels, dtype=np.int64)

    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(labels, preds))
    out["precision"] = float(precision_score(labels, preds, zero_division=0))
    out["recall"] = float(recall_score(labels, preds, zero_division=0))
    out["f1"] = float(f1_score(labels, preds, zero_division=0))
    if len(np.unique(labels)) > 1 and len(np.unique(preds)) > 1:
        out["mcc"] = float(matthews_corrcoef(labels, preds))
    else:
        out["mcc"] = 0.0
    if len(np.unique(labels)) > 1:
        try:
            out["auc"] = float(roc_auc_score(labels, probs))
        except Exception:
            out["auc"] = 0.0
    else:
        out["auc"] = 0.0

    # Validator's reward proxy: 0.5 * MCC + 0.5 * Accuracy
    out["validator_proxy"] = 0.5 * out["mcc"] + 0.5 * out["accuracy"]
    return out


# ---------- collator ----------

def data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return {"pixel_values": pixel_values, "labels": labels}


# ---------- custom Trainer ----------

class WeightedSamplerTrainer(Trainer):
    """Trainer that swaps in a class-balanced sampler when requested."""

    def __init__(self, *args, train_sampler: Optional[torch.utils.data.Sampler] = None,
                 param_groups: Optional[List[Dict[str, Any]]] = None, **kwargs):
        self._custom_sampler = train_sampler
        self._custom_param_groups = param_groups
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:  # type: ignore[override]
        if self._custom_sampler is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=self._custom_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is not None or self._custom_param_groups is None:
            return super().create_optimizer()
        self.optimizer = torch.optim.AdamW(
            self._custom_param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        return self.optimizer


# ---------- main ----------

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.bf16 and args.fp16:
        raise SystemExit("Pick one of --bf16 / --fp16, not both.")
    # On A100, prefer bf16.
    if not args.bf16 and not args.fp16 and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            logger.info("A100/H100 detected -> auto-enabling bf16.")
            args.bf16 = True

    logger.info("Args: %s", json.dumps(vars(args), indent=2))

    # ---------------- data ----------------
    train_ds, val_ds = build_train_val_datasets(
        image_size=args.image_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        extra_dataset_keys=args.extra_datasets,
        cache_dir=args.cache_dir,
    )
    logger.info("Train size=%d  Val size=%d", len(train_ds), len(val_ds))

    train_sampler = build_class_weighted_sampler(train_ds) if args.use_class_balanced_sampler else None

    # ---------------- model ----------------
    model = build_model(args.backbone, num_labels=args.num_labels)
    param_groups = split_param_groups(
        model,
        lr_backbone=args.lr_backbone,
        lr_head=args.lr_head,
        weight_decay=args.weight_decay,
    )

    # Save the backbone's image processor next to checkpoints for inference parity.
    try:
        proc = AutoImageProcessor.from_pretrained(args.backbone, use_fast=True)
        proc.save_pretrained(args.output_dir)
    except Exception as e:
        logger.warning("Could not save image processor: %s", e)

    # ---------------- training args ----------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr_head,  # superseded by custom param groups
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        label_smoothing_factor=args.label_smoothing,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,

        logging_steps=args.logging_steps,
        report_to=args.report_to,
        run_name=args.run_name,

        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    # ---------------- trainer ----------------
    trainer = WeightedSamplerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_sampler=train_sampler,
        param_groups=param_groups,
    )

    if args.resume_from_checkpoint:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        train_result = trainer.train()

    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # ---------------- final eval + threshold sweep ----------------
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    if trainer.is_world_process_zero():
        logger.info("Sweeping decision threshold on validation for max MCC")
        try:
            preds_out = trainer.predict(val_ds)
            logits = preds_out.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = softmax_np(np.asarray(logits, dtype=np.float64))[:, 1]
            labels = np.asarray(preds_out.label_ids, dtype=np.int64)
            best_thr, best_mcc, best_acc = 0.5, -1.0, 0.0
            for thr in np.linspace(0.1, 0.9, 81):
                p = (probs >= thr).astype(np.int64)
                if len(np.unique(p)) > 1 and len(np.unique(labels)) > 1:
                    m = matthews_corrcoef(labels, p)
                else:
                    m = 0.0
                a = accuracy_score(labels, p)
                proxy = 0.5 * m + 0.5 * a
                if proxy > 0.5 * best_mcc + 0.5 * best_acc:
                    best_thr, best_mcc, best_acc = float(thr), float(m), float(a)
            logger.info("Best threshold=%.3f  MCC=%.4f  Acc=%.4f", best_thr, best_mcc, best_acc)
            with open(Path(args.output_dir) / "calibration.json", "w") as f:
                json.dump({"threshold": best_thr, "mcc": best_mcc, "accuracy": best_acc}, f, indent=2)
        except Exception as e:
            logger.warning("Threshold sweep failed: %s", e)

        # Save final HF-format model (the directory is loadable by AutoModelForImageClassification)
        final_dir = Path(args.output_dir) / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        try:
            proc = AutoImageProcessor.from_pretrained(args.backbone, use_fast=True)
            proc.save_pretrained(str(final_dir))
        except Exception:
            pass
        logger.info("Saved final model to %s", final_dir)


if __name__ == "__main__":
    main()
