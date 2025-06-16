import jittor as jt
from jittor import nn
import numpy as np
from typing import Dict, List, Optional
import time
import logging
from pathlib import Path

from models.improved_skeleton import MultiTaskSkeletonModel, SkeletonLoss

class WarmupScheduler:
    """学习率预热调度器"""
    def __init__(self, optimizer, warmup_steps: int, d_model: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.current_step = 0
        
    def step(self):
        """更新学习率"""
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            # 线性预热
            lr = self.d_model ** (-0.5) * min(
                self.current_step ** (-0.5),
                self.current_step * self.warmup_steps ** (-1.5)
            )
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

class DynamicWeightAveraging:
    """动态权重平均器"""
    def __init__(self, num_tasks: int, alpha: float = 0.5):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.task_weights = jt.ones(num_tasks) / num_tasks
        self.task_losses = jt.zeros(num_tasks)
        
    def update(self, losses: List[jt.Var]):
        """更新任务权重"""
        # 计算每个任务的相对损失
        for i, loss in enumerate(losses):
            self.task_losses[i] = self.alpha * self.task_losses[i] + (1 - self.alpha) * loss.item()
        
        # 更新权重
        weights = jt.exp(-self.task_losses)
        self.task_weights = weights / jt.sum(weights)
        
        return self.task_weights

class ImprovedSkeletonTrainer:
    """改进的骨骼预测训练器"""
    def __init__(self, 
                 model: MultiTaskSkeletonModel,
                 optimizer: nn.Optimizer,
                 scheduler: Optional[nn.LRScheduler] = None,
                 grad_clip: float = 1.0,
                 warmup_steps: int = 1000,
                 save_dir: str = "output/improved"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 损失函数
        self.criterion = SkeletonLoss()
        
        # 学习率预热
        self.warmup_scheduler = WarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            d_model=512
        )
        
        # 动态权重平均
        self.weight_averaging = DynamicWeightAveraging(num_tasks=4)  # joint, skin, vertex, bone_length
        
        # 设置日志
        self.setup_logging()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.save_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {
            'total_loss': 0.0,
            'joint_loss': 0.0,
            'bone_length_loss': 0.0,
            'symmetry_loss': 0.0,
            'vertex_loss': 0.0
        }
        
        for batch_idx, batch in enumerate(train_loader):
            # 1. 前向传播
            outputs = self.model(batch['vertices'])
            
            # 2. 计算损失
            losses = self.criterion(
                outputs['joints'],
                batch['joints'],
                outputs['vertices'],
                batch['vertices']
            )
            
            # 3. 动态权重调整
            task_losses = [
                losses['joint_loss'],
                losses['bone_length_loss'],
                losses['symmetry_loss'],
                losses['vertex_loss']
            ]
            weights = self.weight_averaging.update(task_losses)
            
            # 4. 计算加权总损失
            total_loss = (weights[0] * losses['joint_loss'] +
                         weights[1] * losses['bone_length_loss'] +
                         weights[2] * losses['symmetry_loss'] +
                         weights[3] * losses['vertex_loss'])
            
            # 5. 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 6. 梯度裁剪
            jt.clip_grad_norm(self.model.parameters(), self.grad_clip)
            
            # 7. 优化器步进
            self.optimizer.step()
            self.warmup_scheduler.step()
            
            # 8. 更新统计信息
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            
            # 9. 打印进度
            if (batch_idx + 1) % 10 == 0:
                self.logger.info(
                    f'Epoch: {self.current_epoch + 1} '
                    f'Batch: {batch_idx + 1}/{len(train_loader)} '
                    f'Loss: {total_loss.item():.4f}'
                )
        
        # 计算平均损失
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
        
        return epoch_losses
    
    def validate(self, val_loader) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'joint_loss': 0.0,
            'bone_length_loss': 0.0,
            'symmetry_loss': 0.0,
            'vertex_loss': 0.0
        }
        
        with jt.no_grad():
            for batch in val_loader:
                # 1. 前向传播
                outputs = self.model(batch['vertices'])
                
                # 2. 计算损失
                losses = self.criterion(
                    outputs['joints'],
                    batch['joints'],
                    outputs['vertices'],
                    batch['vertices']
                )
                
                # 3. 更新统计信息
                for k, v in losses.items():
                    val_losses[k] += v.item()
        
        # 计算平均损失
        for k in val_losses:
            val_losses[k] /= len(val_loader)
        
        return val_losses
    
    def train(self, 
              train_loader,
              val_loader,
              num_epochs: int,
              save_freq: int = 5,
              early_stopping_patience: int = 10):
        """训练模型"""
        self.logger.info("开始训练...")
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses = self.validate(val_loader)
            
            # 打印epoch统计信息
            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs} - '
                f'Train Loss: {train_losses["total_loss"]:.4f} - '
                f'Val Loss: {val_losses["total_loss"]:.4f}'
            )
            
            # 保存最佳模型
            if val_losses['total_loss'] < self.best_loss:
                self.best_loss = val_losses['total_loss']
                self.save_checkpoint('best_model.pkl')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pkl')
            
            # 早停
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        training_time = time.time() - start_time
        self.logger.info(f'训练完成！总用时: {training_time:.2f}秒')
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'weight_averaging': self.weight_averaging.task_weights
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        jt.save(checkpoint, self.save_dir / filename)
        self.logger.info(f'保存检查点到 {filename}')
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = jt.load(self.save_dir / filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        self.weight_averaging.task_weights = checkpoint['weight_averaging']
        self.logger.info(f'从 {filename} 加载检查点') 
