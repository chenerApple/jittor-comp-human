import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from typing import Dict

# Import the PCT model components
from PCT.networks.cls.pct import Point_Transformer, Point_Transformer2, Point_Transformer_Last, SA_Layer, Local_op, sample_and_group

class HumanPriorModule(nn.Module):
    """人体先验知识模块，用于约束骨骼预测结果"""
    def __init__(self):
        super().__init__()
        # 定义人体骨骼连接关系 (基于SMPL模型的22个关节)
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3),  # 脊柱
            (3, 4), (4, 5), (5, 6),  # 左臂
            (3, 7), (7, 8), (8, 9),  # 右臂
            (0, 10), (10, 11), (11, 12),  # 左腿
            (0, 13), (13, 14), (14, 15),  # 右腿
            (0, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21)  # 头部
        ]
        
        # 定义骨骼长度约束
        self.MIN_BONE_LENGTH = 0.05
        self.MAX_BONE_LENGTH = 0.5
        self.TARGET_BONE_LENGTH = 0.2
        
        # 定义对称关节对
        self.symmetric_pairs = [
            (4, 7), (5, 8), (6, 9),    # 手臂
            (10, 13), (11, 14), (12, 15)  # 腿部
        ]
    
    def apply_bone_constraints(self, joints: jt.Var) -> jt.Var:
        """应用骨骼约束"""
        # 确保输入是正确的形状 [B, 66] -> [B, 22, 3]
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        
        # 1. 应用骨骼长度约束
        for parent, child in self.bone_connections:
            if parent < joints.shape[1] and child < joints.shape[1]:
                bone_length = jt.norm(joints[:, child] - joints[:, parent], dim=1)
                # 使用向量化操作处理所有批次样本
                mask = (bone_length < self.MIN_BONE_LENGTH) | (bone_length > self.MAX_BONE_LENGTH)
                if jt.any(mask):
                    direction = (joints[:, child] - joints[:, parent]) / (bone_length.unsqueeze(1) + 1e-6)
                    target_pos = joints[:, parent] + direction * self.TARGET_BONE_LENGTH
                    joints[:, child] = jt.where(mask.unsqueeze(1), target_pos, joints[:, child])
        
        # 2. 应用对称性约束
        for left, right in self.symmetric_pairs:
            if left < joints.shape[1] and right < joints.shape[1]:
                # 确保左右对称关节的y坐标相同，x坐标相反
                joints[:, right, 0] = -joints[:, left, 0]
                joints[:, right, 1] = joints[:, left, 1]
                joints[:, right, 2] = joints[:, left, 2]
        
        # 返回展平的形状 [B, 66]
        return joints.reshape(joints.shape[0], -1)

class SkeletonLoss(nn.Module):
    """改进的骨骼预测损失函数，包含人体先验约束"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.human_prior = HumanPriorModule()
    
    def _compute_bone_lengths(self, joints: jt.Var) -> jt.Var:
        """计算骨骼长度"""
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        
        bone_lengths = []
        for parent, child in self.human_prior.bone_connections:
            if parent < joints.shape[1] and child < joints.shape[1]:
                length = jt.norm(joints[:, child] - joints[:, parent], dim=1)
                bone_lengths.append(length)
        
        if bone_lengths:
            return jt.stack(bone_lengths, dim=1)  # [B, num_bones]
        else:
            return jt.zeros((joints.shape[0], 1))
    
    def _compute_symmetry_loss(self, joints: jt.Var) -> jt.Var:
        """计算对称性损失"""
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        
        symmetry_loss = jt.zeros(1)
        for left, right in self.human_prior.symmetric_pairs:
            if left < joints.shape[1] and right < joints.shape[1]:
                # 计算左右对称关节的差异
                diff = jt.abs(joints[:, left, 1:] - joints[:, right, 1:])  # y和z坐标应该相同
                symmetry_loss += jt.mean(diff)
                # x坐标应该相反
                symmetry_loss += jt.mean(jt.abs(joints[:, left, 0] + joints[:, right, 0]))
        
        if len(self.human_prior.symmetric_pairs) > 0:
            return symmetry_loss / len(self.human_prior.symmetric_pairs)
        else:
            return symmetry_loss
    
    def execute(self, pred_joints: jt.Var, target_joints: jt.Var) -> Dict[str, jt.Var]:
        """计算多个损失项"""
        # 1. 基础关节位置损失
        joint_loss = self.l1_loss(pred_joints, target_joints)
        
        # 2. 骨骼长度约束损失
        pred_bone_lengths = self._compute_bone_lengths(pred_joints)
        target_bone_lengths = self._compute_bone_lengths(target_joints)
        bone_length_loss = self.mse_loss(pred_bone_lengths, target_bone_lengths)
        
        # 3. 对称性损失
        symmetry_loss = self._compute_symmetry_loss(pred_joints)
        
        # 4. 组合所有损失
        total_loss = joint_loss + 0.1 * bone_length_loss + 0.05 * symmetry_loss
        
        return {
            'total_loss': total_loss,
            'joint_loss': joint_loss,
            'bone_length_loss': bone_length_loss,
            'symmetry_loss': symmetry_loss
        }

class SimpleSkeletonModel(nn.Module):
    
    def __init__(self, feat_dim: int, output_channels: int):
        super().__init__()
        self.feat_dim           = feat_dim
        self.output_channels    = output_channels
        
        self.transformer = Point_Transformer(output_channels=feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_channels),
        )
        
        # 添加人体先验模块
        self.human_prior = HumanPriorModule()
    
    def execute(self, vertices: jt.Var):
        x = self.transformer(vertices)
        joints = self.mlp(x)
        
        # 应用人体先验约束
        joints = self.human_prior.apply_bone_constraints(joints)
        
        return joints

# Factory function to create models
def create_model(model_name='pct', output_channels=66, **kwargs):
    if model_name == "pct":
        return SimpleSkeletonModel(feat_dim=256, output_channels=output_channels)
    raise NotImplementedError()
