import jittor as jt
from jittor import nn
from jittor import init
from jittor.contrib import concat
import numpy as np
import sys
from typing import Dict, List, Tuple

# 导入改进的PointTransformerV3
from PCT.networks.cls.pct import Point_Transformer

class HumanPriorModule(nn.Module):
    """人体先验知识模块，用于约束骨骼预测结果"""
    def __init__(self):
        super().__init__()
        # 定义人体骨骼连接关系
        self.bone_connections = [
            (0, 1), (1, 2), (2, 3),  # 脊柱
            (3, 4), (4, 5), (5, 6),  # 左臂
            (3, 7), (7, 8), (8, 9),  # 右臂
            (0, 10), (10, 11), (11, 12),  # 左腿
            (0, 13), (13, 14), (14, 15)   # 右腿
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
        # 1. 应用骨骼长度约束
        for parent, child in self.bone_connections:
            bone_length = jt.norm(joints[child] - joints[parent])
            if bone_length < self.MIN_BONE_LENGTH or bone_length > self.MAX_BONE_LENGTH:
                direction = (joints[child] - joints[parent]) / (bone_length + 1e-6)
                joints[child] = joints[parent] + direction * self.TARGET_BONE_LENGTH
        
        # 2. 应用对称性约束
        for left, right in self.symmetric_pairs:
            # 确保左右对称关节的y坐标相同，x坐标相反
            joints[right, 0] = -joints[left, 0]
            joints[right, 1] = joints[left, 1]
            joints[right, 2] = joints[left, 2]
        
        return joints

class SkeletonLoss(nn.Module):
    """改进的骨骼预测损失函数"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.human_prior = HumanPriorModule()
    
    def _compute_bone_lengths(self, joints: jt.Var) -> jt.Var:
        """计算骨骼长度"""
        bone_lengths = []
        for parent, child in self.human_prior.bone_connections:
            length = jt.norm(joints[child] - joints[parent])
            bone_lengths.append(length)
        return jt.stack(bone_lengths)
    
    def _compute_symmetry_loss(self, joints: jt.Var) -> jt.Var:
        """计算对称性损失"""
        symmetry_loss = 0.0
        for left, right in self.human_prior.symmetric_pairs:
            # 计算左右对称关节的差异
            diff = jt.abs(joints[left, 1:] - joints[right, 1:])  # y和z坐标应该相同
            symmetry_loss += jt.mean(diff)
            # x坐标应该相反
            symmetry_loss += jt.mean(jt.abs(joints[left, 0] + joints[right, 0]))
        return symmetry_loss / len(self.human_prior.symmetric_pairs)
    
    def execute(self, pred_joints: jt.Var, target_joints: jt.Var, 
                pred_vertices: jt.Var, target_vertices: jt.Var) -> Dict[str, jt.Var]:
        """计算多个损失项"""
        # 1. 基础关节位置损失
        joint_loss = self.l1_loss(pred_joints, target_joints)
        
        # 2. 骨骼长度约束损失
        pred_bone_lengths = self._compute_bone_lengths(pred_joints)
        target_bone_lengths = self._compute_bone_lengths(target_joints)
        bone_length_loss = self.mse_loss(pred_bone_lengths, target_bone_lengths)
        
        # 3. 对称性损失
        symmetry_loss = self._compute_symmetry_loss(pred_joints)
        
        # 4. 顶点位置损失
        vertex_loss = self.l1_loss(pred_vertices, target_vertices)
        
        # 5. 组合所有损失
        total_loss = (joint_loss + 
                     0.1 * bone_length_loss + 
                     0.05 * symmetry_loss + 
                     0.2 * vertex_loss)
        
        return {
            'total_loss': total_loss,
            'joint_loss': joint_loss,
            'bone_length_loss': bone_length_loss,
            'symmetry_loss': symmetry_loss,
            'vertex_loss': vertex_loss
        }

class MultiTaskSkeletonModel(nn.Module):
    """多任务骨骼预测模型"""
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        # 共享特征提取器
        self.shared_encoder = Point_Transformer(
            output_channels=feat_dim  # 修改参数以匹配Point_Transformer的接口
        )
        
        # 特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 骨骼位置预测分支
        self.joint_branch = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 66)  # 22个关节，每个3个坐标
        )
        
        # 蒙皮权重预测分支
        self.skin_branch = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 22)  # 22个关节的权重
        )
        
        # 顶点位置预测分支
        self.vertex_branch = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3)  # 3D坐标
        )
        
        # 人体先验模块
        self.human_prior = HumanPriorModule()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
    
    def execute(self, vertices: jt.Var) -> Dict[str, jt.Var]:
        """前向传播"""
        # 1. 提取共享特征
        shared_features = self.shared_encoder(vertices)
        enhanced_features = self.feature_enhancement(shared_features)
        
        # 2. 多任务预测
        joint_pred = self.joint_branch(enhanced_features)
        skin_pred = self.skin_branch(enhanced_features)
        vertex_pred = self.vertex_branch(enhanced_features)
        
        # 3. 应用人体先验约束
        joint_pred = self.human_prior.apply_bone_constraints(joint_pred)
        
        return {
            'joints': joint_pred,
            'skin_weights': skin_pred,
            'vertices': vertex_pred
        }

# 工厂函数
def create_model(model_name: str = 'improved', **kwargs) -> nn.Module:
    """创建模型的工厂函数"""
    if model_name == "improved":
        return MultiTaskSkeletonModel(**kwargs)
    raise NotImplementedError(f"Model {model_name} not implemented") 
