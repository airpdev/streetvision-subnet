# Training pipeline — StreetVision roadwork classifier

A drop-in training pipeline for fine-tuning a strong vision backbone for the
NATIX StreetVision subnet. Optimized for the validator's reward (rolling
**MCC@100 + Accuracy@10**, threshold 0.5).

## TL;DR

```bash
# 0. (one-time) install training deps in a SEPARATE venv from your miner
python3.11 -m venv venv-train
source venv-train/bin/activate
pip install -U pip
pip install -r training/requirements.txt

# 1. (one-time) login to HF so you can download the dataset
huggingface-cli login   # paste read token

# 2. train on 2x A100 40GB (DDP via accelerate)
bash training/launch_ddp.sh

# 3. (optional) re-calibrate decision threshold on a new split
python training/eval_threshold.py \
    --checkpoint ./checkpoints/convnextv2-roadwork/final \
    --image_size 384

# 4. push the trained model to HF (so the miner can pull it)
huggingface-cli upload <your-username>/roadwork-convnextv2 \
    ./checkpoints/convnextv2-roadwork/final
```

Then point your miner config at the published model:

```yaml
# natix/miner/detectors/configs/ViT_roadwork.yaml
hf_repo: '<your-username>/roadwork-convnextv2'
config_name: 'config.yaml'
weights: 'model.safetensors'
```

The default `ViTImageDetector` already loads any HF
`AutoModelForImageClassification` model — no detector code changes needed.

---

## What this pipeline does

* Loads the **canonical evaluation dataset** `natix-network-org/roadwork`
  (the same one the validator uses for "real" challenges).
* Optionally concatenates additional roadwork-related sources (see
  `data.AUX_DATASET_LOADERS`).
* Stratified 90/10 train/val split on the binary label.
* Strong albumentations pipeline (random crop, blur, jitter, JPEG compression,
  CoarseDropout) — designed to generalize to the validator's
  synthetic + API images.
* HuggingFace `Trainer` with:
  * Cosine LR schedule + warmup
  * **Discriminative LR**: smaller LR on backbone, larger on the new head
  * Label smoothing
  * Mixed precision (bf16 by default on A100)
  * Optional class-balanced sampler
  * `metric_for_best_model="mcc"` (matches validator's primary signal)
* Final automatic **decision-threshold sweep** over [0.1, 0.9] in steps of 0.01,
  maximizing `0.5 * MCC + 0.5 * Accuracy` (the validator's reward proxy).
* Saves a final HuggingFace-format checkpoint at `./<output_dir>/final/`.

## Files

| File | Purpose |
|------|---------|
| `requirements.txt` | Training-time pip dependencies. |
| `data.py` | Dataset loader + albumentations transforms + stratified split. |
| `train.py` | Main training script. Use with `accelerate launch`. |
| `launch_ddp.sh` | Wrapper that calls `accelerate launch` for 2x A100. |
| `eval_threshold.py` | Standalone evaluator + decision-threshold sweep. |
| `README.md` | This file. |

---

## Hardware sizing (2x A100 40GB)

Recommended defaults at `image_size=384`:

| Backbone (HF id) | Per-device batch | Mem / GPU | Throughput | Notes |
|---|---|---|---|---|
| `facebook/convnextv2-large-22k-384` | 32 | ~32 GB | ~280 img/s | **Default. Best accuracy/latency tradeoff.** |
| `facebook/convnextv2-huge-22k-384`  | 12 | ~36 GB | ~140 img/s | Top accuracy, ~2x compute. |
| `facebook/dinov2-large` (`224`)     | 64 | ~28 GB | ~600 img/s | Use `--image_size 224`. Strongest frozen features. |
| `microsoft/swinv2-large-patch4-window12to24-192to384-22kto1k-ft` | 24 | ~36 GB | ~180 img/s | Solid alternative. |
| `google/siglip-so400m-patch14-384` | 24 | ~38 GB | ~190 img/s | Excellent for combining with zero-shot. |

For SigLIP / DINOv2 (which expose features without a head), the trainer adds
a fresh classification head automatically thanks to
`ignore_mismatched_sizes=True`.

To swap backbone, pass `--backbone <id>` (and matching `--image_size`):

```bash
BACKBONE=facebook/dinov2-large IMAGE_SIZE=224 \
    bash training/launch_ddp.sh
```

---

## Hyperparameters (defaults explained)

* `--lr_backbone 1e-5`, `--lr_head 1e-3` — discriminative LR. The head is
  randomly initialized so it needs a much higher LR than the pretrained
  backbone.
* `--weight_decay 0.05`, **disabled on biases / LayerNorm** (standard).
* `--warmup_ratio 0.06`, cosine decay afterwards.
* `--label_smoothing 0.05` — improves calibration, slightly raises MCC.
* `--epochs 12` — usually plateaus by ~8 on the 8.5k primary set; 12 leaves
  margin if you mix in extra data.
* `--use_class_balanced_sampler` — recommended; toggles a
  `WeightedRandomSampler` over the labels.

---

## Adding your own datasets

Open `training/data.py` and edit `AUX_DATASET_LOADERS`:

```python
AUX_DATASET_LOADERS: Dict[str, Callable[[], HFDataset]] = {
    "my_extra": lambda: load_dataset("you/your-roadwork-set", split="train"),
}
```

Each loader must return rows with at least an `image` (PIL or bytes) and a
`label` column with values 0/1. Then run:

```bash
bash training/launch_ddp.sh --extra_datasets my_extra
```

---

## Multi-stage strategy (recommended for winning)

This trainer is designed to be the **classifier component** of a 3-way
ensemble at inference time:

1. **This fine-tuned backbone** (~120 ms on A100, ~600 ms on CPU).
2. **Qwen2.5-VL-7B (INT4)** — local VLM voter (~1.5 s on A100).
3. **SigLIP zero-shot** with engineered prompts (~50 ms, no training).

Save each model's prob, blend with a small calibrated logistic regression on
your validation set, threshold-tune. The 9 s axon timeout fits this comfortably
on A100 hardware.

---

## Troubleshooting

* **OOM**: drop `--batch_size` and/or `--image_size`, or set
  `--grad_accum 2` to keep effective batch size.
* **Dataset 401**: run `huggingface-cli login` with a Read token.
* **Slow data loading**: bump `--num_workers` (default 8 is fine for 2 GPUs).
* **Trainer never saves "best"**: confirm metrics are computed — eval should
  print `mcc`, `accuracy`, `validator_proxy` each `--eval_steps`.
* **Loss diverges early**: lower `--lr_head` to `5e-4` and increase
  `--warmup_ratio` to `0.1`.
