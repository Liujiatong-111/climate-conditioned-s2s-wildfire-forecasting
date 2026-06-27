# TeleViT Multi-Branch Wildfire Forecasting

This is a cleaned public release of the TeleViT multi-branch transformer code for global wildfire forecasting. It keeps the training, inference, model, dataset, and analysis source code, but excludes generated results, checkpoints, logs, compiled paper files, and local experiment artifacts.

## Structure

```text
public_release/
├── configs/
│   └── config.example.yaml      # edit data paths before running
├── src/
│   ├── train.py                 # training entry point
│   ├── inference.py             # sliding-window global inference
│   ├── model_multibranch_vit.py # model architecture
│   ├── dataset.py               # SeasFire patch dataset
│   ├── focal_loss.py
│   ├── logger.py
│   ├── utils.py
│   ├── optimize_threshold.py
│   ├── visualize.py
│   └── visualize_mask.py
├── scripts/
│   ├── analysis/                # paper/diagnostic analysis scripts only
│   └── diagnostics/             # mask checking utilities
├── docs/                        # experiment notes retained as documentation
├── requirements.txt
└── .gitignore
```

## What Is Not Included

The following directories from the working tree were intentionally not copied:

- `checkpoints/`
- `inference/`
- `inference_results/`
- `logs/`
- `paper/`
- `fire-paper/`
- `visualizations/`
- analysis output folders containing figures, CSVs, PDFs, or NumPy arrays

## Setup

```bash
conda create -n televit python=3.10 -y
conda activate televit
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if the default wheel is not suitable.

## Data

Edit `configs/config.example.yaml`:

```yaml
data:
  zarr_path: /path/to/SeasFireCube_v3.zarr
  target_zarr_path: /path/to/SeasFireCube_v3.zarr
```

The expected variables are listed in the config file. Generated data, checkpoints, and inference outputs should stay outside version control.

## Train

Run from `public_release/`:

```bash
python src/train.py --config configs/config.example.yaml
```

By default, training writes to:

- `outputs/`
- `checkpoints/`
- `logs/`

## Inference

```bash
python src/inference.py \
  --config configs/config.example.yaml \
  --checkpoint checkpoints/best_model.pth \
  --output-dir inference_results
```

## Analysis Scripts

Analysis scripts under `scripts/analysis/` default to `data/SeasFireCube_v3.zarr`. You can override the dataset location with:

```bash
export SEASFIRE_ZARR=/path/to/SeasFireCube_v3.zarr
python scripts/analysis/analysis_timescale_acf.py
```

