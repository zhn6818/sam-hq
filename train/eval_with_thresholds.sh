#!/bin/bash

# 设置环境变量以启用MPS回退（如果使用Mac的MPS设备）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 创建输出目录
OUTPUT_DIR="work_dirs/hq_sam_threshold_eval"
mkdir -p $OUTPUT_DIR

# 检查命令行参数
MODEL_TYPE=${1:-"vit_b"}
CHECKPOINT_PATH=${2:-"./pretrained_checkpoint/sam_vit_b_01ec64.pth"}
RESTORE_MODEL=${3:-"work_dirs/hq_sam_b/epoch_11.pth"}
DEVICE=${4:-"cuda"}

echo "评估HQ-SAM模型，使用多个阈值..."
echo "模型类型: $MODEL_TYPE"
echo "SAM模型路径: $CHECKPOINT_PATH"
echo "HQ-Decoder路径: $RESTORE_MODEL"
echo "使用设备: $DEVICE"
echo "结果将保存到: $OUTPUT_DIR"

# 使用多阈值评估
python train.py \
  --device $DEVICE \
  --checkpoint $CHECKPOINT_PATH \
  --model-type $MODEL_TYPE \
  --output $OUTPUT_DIR \
  --eval \
  --visualize \
  --restore-model $RESTORE_MODEL \
  --threshold_values 0.0 0.1 0.3 0.5 0.7 0.9

echo "多阈值评估完成！结果保存在 $OUTPUT_DIR 目录中"

# 单阈值评估示例（阈值为0.5）
echo "使用单一阈值0.5进行评估..."
SINGLE_THRESH_DIR="$OUTPUT_DIR/single_threshold_0.5"
mkdir -p $SINGLE_THRESH_DIR

python train.py \
  --device $DEVICE \
  --checkpoint $CHECKPOINT_PATH \
  --model-type $MODEL_TYPE \
  --output $SINGLE_THRESH_DIR \
  --eval \
  --visualize \
  --restore-model $RESTORE_MODEL \
  --threshold_values 0.5

echo "单阈值评估完成！结果保存在 $SINGLE_THRESH_DIR 目录中" 