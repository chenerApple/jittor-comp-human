import jittor as jt
import argparse
import os
from pathlib import Path
import logging
from typing import Dict, Any

from models.improved_skeleton import create_model
from train.improved_trainer import ImprovedSkeletonTrainer
from dataset import create_dataloader

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练改进的骨骼预测模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='data',
                      help='数据目录路径')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='improved',
                      help='模型名称')
    parser.add_argument('--feat_dim', type=int, default=256,
                      help='特征维度')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                      help='预热步数')
    parser.add_argument('--save_freq', type=int, default=5,
                      help='保存检查点频率')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='早停耐心值')
    
    # 其他参数
    parser.add_argument('--output_dir', type=str, default='output/improved',
                      help='输出目录')
    parser.add_argument('--log_dir', type=str, default='logs/improved',
                      help='日志目录')
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    
    return parser.parse_args()

def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    jt.set_global_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.log_dir)
    logger.info("开始训练改进的骨骼预测模型")
    logger.info(f"参数配置: {args}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        split='train',
        shuffle=True
    )
    val_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        split='val',
        shuffle=False
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(
        model_name=args.model_name,
        feat_dim=args.feat_dim
    )
    
    # 创建优化器
    optimizer = jt.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = jt.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    # 创建训练器
    trainer = ImprovedSkeletonTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=1.0,
        warmup_steps=args.warmup_steps,
        save_dir=args.output_dir
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_freq=args.save_freq,
        early_stopping_patience=args.early_stopping_patience
    )
    
    logger.info("训练完成！")

if __name__ == '__main__':
    main() 