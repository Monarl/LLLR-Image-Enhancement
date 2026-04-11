# Baseline Retraining on RELLISUR (Option B)

This folder contains retraining templates for:

- PAN
- SwinIR
- HAT
- SRFormer

All use `model_type: SRModel` and BasicSR-registered wrappers in:

- `basicsr/archs/baseline_compare_arch.py`

## Train commands

From `MambaIR` root:

```bash
python basicsr/train.py -opt options/train/baselines/train_PAN_RELLISUR_x2.yml
python basicsr/train.py -opt options/train/baselines/train_SwinIR_RELLISUR_x2.yml
python basicsr/train.py -opt options/train/baselines/train_HAT_RELLISUR_x2.yml
python basicsr/train.py -opt options/train/baselines/train_SRFormer_RELLISUR_x2.yml
```

## Test commands

```bash
python basicsr/test.py -opt options/test/baselines/test_PAN_RELLISUR_x2.yml
python basicsr/test.py -opt options/test/baselines/test_SwinIR_RELLISUR_x2.yml
python basicsr/test.py -opt options/test/baselines/test_HAT_RELLISUR_x2.yml
python basicsr/test.py -opt options/test/baselines/test_SRFormer_RELLISUR_x2.yml
```

## Important notes

- Replace dataset paths if your local RELLISUR paths differ.
- Replace checkpoint paths in `options/test/baselines/*.yml` with your trained `.pth` files.
- `swinIR.py` depends on `timm` (`pip install timm`).
- If memory is tight:
  - reduce `batch_size_per_gpu`
  - reduce `gt_size`
  - reduce model width/depth in `network_g`.
