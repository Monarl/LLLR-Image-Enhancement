#!/bin/bash

echo "Checking GPU and CUDA versions..."
/venv/main/bin/python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')"
nvcc --version

echo "Installing causal-conv1d..."
cd /workspace
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.3
CAUSAL_CONV1D_FORCE_BUILD=TRUE /venv/main/bin/pip install --no-build-isolation .

echo "Installing mamba-ssm..."
cd /workspace
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.6
MAMBA_FORCE_BUILD=TRUE /venv/main/bin/pip install --no-build-isolation .

echo "Installing dependencies..."
/venv/main/bin/pip install -q einops timm==0.9.16 transformers==4.38.1 safetensors \
    tensorboard opencv-python pyyaml matplotlib==3.7.5 \
    scipy==1.11.4 tqdm huggingface-hub==0.21.1 regex addict lmdb gdown rawpy ipdb yacs scikit-image fvcore lpips

echo "Cloning LLLR-Image-Enhancement repository..."
cd /workspace
git clone https://github.com/Monarl/LLLR-Image-Enhancement.git
# git clone https://github.com/Monarl/UltraIS-main.git
cd LLLR-Image-Enhancement
git checkout Mini-ASSP

echo "Downloading and extracting the dataset..."
/venv/main/bin/python -m gdown 1MPyuVEY4oQpRLligJvQ4mRLfXtEAn6Ff -O /tmp/rellisur.zip
unzip -o /tmp/rellisur.zip -d /workspace/LLLR-Image-Enhancement/datasets/
rm /tmp/rellisur.zip

echo "Setup complete! Starting MambaIRv2 training..."
cd /workspace/LLLR-Image-Enhancement
# /venv/main/bin/python basicsr/train.py -opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_igm.yml
# /venv/main/bin/python basicsr/train.py -opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_isdm.yml
/venv/main/bin/python basicsr/train.py -opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_igm_input_image.yml
# /venv/main/bin/python basicsr/train.py -opt options/train/baselines/train_SwinIR_RELLISUR_x2.yml

# /venv/main/bin/python basicsr/test.py -opt options/test/mambairv2_llie_sr/test_MambaIRv2_LLIESR_x2_ablate_igm.yml

# /venv/main/bin/python metric_full_eval_out2csv.py \
#   --pred_dir results/test_MambaIRv2_LLIESR_x2_A1_NoIGM/visualization/RELLISUR_Test \
#   --gt_dir datasets/RELLISUR-Dataset/Test/NLHR/X2 \
#   --csv_path results/test_MambaIRv2_LLIESR_x2_A1_NoIGM/metrics_full.csv \
#   --crop_border 2

: <<'ABLATION_COPY_COMMANDS'
----------------------- New x2 Ablation Commands -----------------------

# 1) No illumination loss
/venv/main/bin/python basicsr/train.py -opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_no_illumination_loss.yml
/venv/main/bin/python basicsr/test.py -opt options/test/mambairv2_llie_sr/test_MambaIRv2_LLIESR_x2_ablate_no_illumination_loss.yml
/venv/main/bin/python metric_full_eval_out2csv.py \
  --pred_dir results/test_MambaIRv2_LLIESR_x2_A5_NoIlluminationLoss/visualization/RELLISUR_Test \
  --gt_dir datasets/RELLISUR-Dataset/Test/NLHR/X2 \
  --csv_path results/test_MambaIRv2_LLIESR_x2_A5_NoIlluminationLoss/metrics_full.csv \
  --crop_border 2 \
  --test_y_channel
/venv/main/bin/python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_no_illumination_loss.yml \
  --checkpoint experiments/pretrained_models/net_g_latest_no_illumination_loss.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray

# 2) No perceptual loss
/venv/main/bin/python basicsr/train.py -opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_no_perceptual_loss.yml
/venv/main/bin/python basicsr/test.py -opt options/test/mambairv2_llie_sr/test_MambaIRv2_LLIESR_x2_ablate_no_perceptual_loss.yml
/venv/main/bin/python metric_full_eval_out2csv.py \
  --pred_dir results/test_MambaIRv2_LLIESR_x2_A6_NoPerceptualLoss/visualization/RELLISUR_Test \
  --gt_dir datasets/RELLISUR-Dataset/Test/NLHR/X2 \
  --csv_path results/test_MambaIRv2_LLIESR_x2_A6_NoPerceptualLoss/metrics_full.csv \
  --crop_border 2 \
  --test_y_channel
/venv/main/bin/python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_no_perceptual_loss.yml \
  --checkpoint experiments/pretrained_models/net_g_latest_no_perceptual_loss.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray

# 3) IGM input image
/venv/main/bin/python basicsr/train.py -opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_igm_input_image.yml
/venv/main/bin/python basicsr/test.py -opt options/test/mambairv2_llie_sr/test_MambaIRv2_LLIESR_x2_ablate_igm_input_image.yml
/venv/main/bin/python metric_full_eval_out2csv.py \
  --pred_dir results/test_MambaIRv2_LLIESR_x2_A4_IGMInputImage/visualization/RELLISUR_Test \
  --gt_dir datasets/RELLISUR-Dataset/Test/NLHR/X2 \
  --csv_path results/test_MambaIRv2_LLIESR_x2_A4_IGMInputImage/metrics_full.csv \
  --crop_border 2 \
  --test_y_channel
/venv/main/bin/python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_igm_input_image.yml \
  --checkpoint experiments/pretrained_models/net_g_latest_igm_input_image_x2.pth --height 128 --width 128 --warmup 10 --repetitions 300

ABLATION_COPY_COMMANDS

# /venv/main/bin/python analysis/benchmark_mambairv2_llie_sr.py --checkpoint experiments/pretrained_models/net_g_latest_no_retinex_x4.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
# /venv/main/bin/python analysis/benchmark_mambairv2_llie_sr.py --checkpoint experiments/pretrained_models/net_g_60000_no_retinex.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
# /venv/main/bin/python analysis/benchmark_mambairv2_llie_sr.py --checkpoint experiments/pretrained_models/net_g_45000_mobilenet.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
# python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_igm.yml \
#   --checkpoint experiments/pretrained_models/net_g_latest_no_igm.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
# python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_isdm.yml \
#   --checkpoint experiments/pretrained_models/net_g_latest_no_isdm.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_mini_aspp.yml \
  --checkpoint experiments/pretrained_models/net_g_latest_no_mini.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
# python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_no_illumination_loss.yml \
#   --checkpoint experiments/pretrained_models/net_g_latest_no_illumination_loss.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
# python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_no_perceptual_loss.yml \
#   --checkpoint experiments/pretrained_models/net_g_latest_no_perceptual_loss.pth --height 128 --width 128 --warmup 10 --repetitions 300 --pass-gray
# python analysis/benchmark_mambairv2_llie_sr.py --opt options/train/mambairv2_llie_sr/train_MambaIRv2_LLIESR_x2_ablate_igm_input_image.yml \
#   --checkpoint experiments/pretrained_models/net_g_latest_igm_input_image_x2.pth --height 128 --width 128 --warmup 10 --repetitions 300
 
# cd /workspace/UltraIS-main
# sh train.sh Super_Resolution/Options/CSDLLSR_v9_7_5_3_scale2.yml

# python analysis/infer_single_image_multi_models.py --config analysis/configs/single_image_infer_example.json 
# python analysis/compare_restoration_panels.py --config analysis/configs/compare_restoration_example.json
# python analysis/benchmark_multi_models_latency.py --config analysis/configs/latency_benchmark_rellisur_full.json --flops
