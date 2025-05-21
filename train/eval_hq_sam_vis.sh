#!/bin/bash

# 设置环境变量以启用MPS回退
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 创建输出目录
mkdir -p work_dirs/hq_sam_b_vis

# 评估HQ-SAM模型并生成可视化结果
python train.py \
  --device mps \
  --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --output work_dirs/hq_sam_b_vis \
  --eval \
  --visualize \
  --restore-model work_dirs/hq_sam_b/epoch_11.pth

echo "可视化评估完成！结果保存在 work_dirs/hq_sam_b_vis 目录中" 