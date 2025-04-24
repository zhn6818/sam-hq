#!/bin/bash

# 设置环境变量以启用MPS回退
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 创建输出目录
mkdir -p work_dirs/hq_sam_b

# 训练HQ-SAM模型
python train.py \
  --device mps \
  --checkpoint ./pretrained_checkpoint/sam_vit_b_01ec64.pth \
  --model-type vit_b \
  --output work_dirs/hq_sam_b

echo "训练完成！模型保存在 work_dirs/hq_sam_b 目录中" 