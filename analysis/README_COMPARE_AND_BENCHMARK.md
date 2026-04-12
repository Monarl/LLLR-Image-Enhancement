# Multi-Model Comparison and Latency Benchmark

This folder now contains two scripts:

- `analysis/compare_restoration_panels.py`
- `analysis/benchmark_multi_models_latency.py`
- `analysis/infer_single_image_multi_models.py`

And two example config files:

- `analysis/configs/compare_restoration_example.json`
- `analysis/configs/latency_benchmark_example.json`
- `analysis/configs/single_image_infer_example.json`

## 0) First: run one-image inference for all models

If you do not already have model output images, run:

```bash
python analysis/infer_single_image_multi_models.py \
  --config analysis/configs/single_image_infer_example.json
```

This will create outputs under:

- `analysis/outputs/single_image_outputs/<model_name>/`

Then the comparison panel script can consume these folders directly.

## 1) Generate visual comparison panels (with zoomed crops)

This script can optionally run test commands first, then create per-image comparison panels and a CSV of PSNR/SSIM.

### Run

From `MambaIR` root:

```bash
python analysis/compare_restoration_panels.py \
  --config analysis/configs/compare_restoration_example.json \
  --run-models
```

If outputs already exist and you only want panel rendering:

```bash
python analysis/compare_restoration_panels.py \
  --config analysis/configs/compare_restoration_example.json
```

### Output

- Panels: `analysis/outputs/compare_panels/*_comparison.png`
- Metrics: `analysis/outputs/compare_panels/metrics_per_image.csv`

### Notes

- `cases` lets you lock image names and crop rectangles (`xywh`) to match paper-style figures.
- For each model, `output_dir` must contain restored images.
- Name matching supports exact stem and prefix style (example: `00018-3.0_test_...png`).

## 2) Benchmark inference time (and optional FLOPs)

This script benchmarks all enabled models in one config and exports a single CSV.

### Run

```bash
python analysis/benchmark_multi_models_latency.py \
  --config analysis/configs/latency_benchmark_example.json
```

With optional FLOPs (requires `thop`):

```bash
python analysis/benchmark_multi_models_latency.py \
  --config analysis/configs/latency_benchmark_example.json \
  --flops
```

### Output

- CSV: `analysis/outputs/latency/latency_report.csv`

The CSV includes:

- params (M)
- mean/std inference time (ms)
- FPS
- optional GFLOPs
- checkpoint loading status

## 3) How to add more models in future

Edit only the config JSON.

Each model entry needs:

- `name`, `label`, `enabled`
- either `builder` or `ctor`
- `module_root` (folder added to `sys.path`)
- `kwargs` to construct model
- `input_mode`: `rgb` or `rgb_gray`
- optional `checkpoint`, `param_key`, `strict`

### Builder-style example

```json
{
  "name": "my_model",
  "label": "MyModel",
  "enabled": true,
  "module_root": ".",
  "builder": {"module": "analysis.model_zoo.my_model", "name": "buildMyModel"},
  "kwargs": {"upscale": 2},
  "input_mode": "rgb"
}
```

### Constructor-style example

```json
{
  "name": "my_model_ctor",
  "label": "MyModelCtor",
  "enabled": true,
  "module_root": "../ExternalRepo",
  "ctor": {"module": "models.arch", "name": "MyNet"},
  "kwargs": {"scale": 2},
  "checkpoint": "../ExternalRepo/weights/mynet_x2.pth",
  "param_key": "auto",
  "strict": false,
  "input_mode": "rgb_gray"
}
```

## 4) Do you need pretrained models?

Yes, if you want fair visual/latency comparison of actual restored quality.

- Without checkpoints, benchmarking random-weight networks is not meaningful for restoration quality.
- For timing-only architecture comparison, you can run without checkpoints, but this is less representative for deployment reports.

## 5) About missing baselines in this workspace

The provided config includes placeholders for `PAN`, `MIRNet`, `Restormer`, and `SRFormer` with `enabled: false`.

To benchmark them, you need:

1. The actual implementation module path.
2. The correct constructor or builder function.
3. (Recommended) corresponding pretrained checkpoint.

After these are available, set `enabled: true` and update `module`/`name`/`checkpoint` in config.
