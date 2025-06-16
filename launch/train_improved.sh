#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  # 使用第一张GPU
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建输出目录
mkdir -p output/improved

# 训练参数
BATCH_SIZE=32
NUM_EPOCHS=100
LEARNING_RATE=0.001
WARMUP_STEPS=1000
SAVE_FREQ=5
EARLY_STOPPING_PATIENCE=10

# 运行训练脚本
python train/train_improved.py \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --save_freq $SAVE_FREQ \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --model_name "improved" \
    --output_dir "output/improved" \
    --log_dir "logs/improved" \
    2>&1 | tee logs/improved/training.log 