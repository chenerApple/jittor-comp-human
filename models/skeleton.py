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
        
        # 定义骨骼类型分类
        self.bone_types = {
            'spine': [(0, 1), (1, 2), (2, 3)],           # 脊柱骨骼
            'arm': [(3, 4), (4, 5), (5, 6), (3, 7), (7, 8), (8, 9)],  # 手臂骨骼
            'leg': [(0, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15)],  # 腿部骨骼
            'head': [(0, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21)]  # 头部骨骼
        }
        
        # 为不同类型的骨骼定义精细的长度约束
        self.bone_length_constraints = {
            'spine': {
                'min_length': 0.08,    # 脊柱段最小长度 8cm
                'max_length': 0.25,    # 脊柱段最大长度 25cm
                'target_length': 0.15, # 目标长度 15cm
                'weight': 1.0          # 权重
            },
            'arm': {
                'min_length': 0.12,    # 手臂骨骼最小长度 12cm
                'max_length': 0.35,    # 手臂骨骼最大长度 35cm
                'target_length': 0.25, # 目标长度 25cm
                'weight': 1.0
            },
            'leg': {
                'min_length': 0.15,    # 腿部骨骼最小长度 15cm
                'max_length': 0.45,    # 腿部骨骼最大长度 45cm
                'target_length': 0.35, # 目标长度 35cm
                'weight': 1.0
            },
            'head': {
                'min_length': 0.05,    # 头部骨骼最小长度 5cm
                'max_length': 0.15,    # 头部骨骼最大长度 15cm
                'target_length': 0.08, # 目标长度 8cm
                'weight': 0.8          # 头部约束稍松一些
            }
        }
        
        # 定义对称关节对
        self.symmetric_pairs = [
            (4, 7), (5, 8), (6, 9),    # 手臂
            (10, 13), (11, 14), (12, 15)  # 腿部
        ]
    
    def _get_bone_type(self, parent: int, child: int) -> str:
        """根据关节对确定骨骼类型"""
        bone_pair = (parent, child)
        for bone_type, connections in self.bone_types.items():
            if bone_pair in connections:
                return bone_type
        return 'spine'  # 默认类型
    
    def apply_bone_constraints(self, joints: jt.Var) -> jt.Var:
        """应用精细的骨骼约束"""
        # 确保输入是正确的形状 [B, 66] -> [B, 22, 3]
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        
        # 1. 应用精细的骨骼长度约束
        for parent, child in self.bone_connections:
            if parent < joints.shape[1] and child < joints.shape[1]:
                # 获取骨骼类型和对应的约束
                bone_type = self._get_bone_type(parent, child)
                constraints = self.bone_length_constraints[bone_type]
                
                # 计算当前骨骼长度
                bone_length = jt.norm(joints[:, child] - joints[:, parent], dim=1)
                
                # 检查是否超出该类型骨骼的合理范围
                min_len = constraints['min_length']
                max_len = constraints['max_length']
                target_len = constraints['target_length']
                weight = constraints['weight']
                
                # 使用更温和的约束：只对极端情况进行调整
                extreme_mask = (bone_length < min_len) | (bone_length > max_len)
                
                if jt.any(extreme_mask):
                    # 计算骨骼方向向量
                    direction = (joints[:, child] - joints[:, parent]) / (bone_length.unsqueeze(1) + 1e-6)
                    
                    # 计算目标位置
                    target_pos = joints[:, parent] + direction * target_len
                    
                    # 使用软约束：不是直接替换，而是向目标位置移动一部分
                    alpha = 0.3  # 调整强度，0.3表示只调整30%的距离
                    adjusted_pos = joints[:, child] * (1 - alpha) + target_pos * alpha
                    
                    # 只对极端情况进行调整
                    joints[:, child] = jt.where(extreme_mask.unsqueeze(1), adjusted_pos, joints[:, child])
        
        # 2. 应用对称性约束
        for left, right in self.symmetric_pairs:
            if left < joints.shape[1] and right < joints.shape[1]:
                # 确保左右对称关节的y坐标相同，x坐标相反
                joints[:, right, 0] = -joints[:, left, 0]
                joints[:, right, 1] = joints[:, left, 1]
                joints[:, right, 2] = joints[:, left, 2]
        
        # 返回展平的形状 [B, 66]
        return joints.reshape(joints.shape[0], -1)
    
    def debug_bone_lengths(self, joints: jt.Var) -> Dict[str, jt.Var]:
        """调试骨骼长度，返回各类型骨骼的统计信息"""
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        
        debug_info = {}
        
        for bone_type, connections in self.bone_types.items():
            bone_lengths = []
            for parent, child in connections:
                if parent < joints.shape[1] and child < joints.shape[1]:
                    length = jt.norm(joints[:, child] - joints[:, parent], dim=1)
                    bone_lengths.append(length)
            
            if bone_lengths:
                lengths = jt.stack(bone_lengths, dim=1)
                constraints = self.bone_length_constraints[bone_type]
                
                debug_info[bone_type] = {
                    'mean_length': jt.mean(lengths),
                    'min_length': jt.min(lengths),
                    'max_length': jt.max(lengths),
                    'constraint_min': constraints['min_length'],
                    'constraint_max': constraints['max_length'],
                    'target_length': constraints['target_length'],
                    'violations': jt.sum((lengths < constraints['min_length']) | (lengths > constraints['max_length']))
                }
        
        return debug_info

class SkeletonLoss(nn.Module):
    """改进的骨骼预测损失函数，包含人体先验约束"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.human_prior = HumanPriorModule()
    
    def _compute_bone_lengths(self, joints: jt.Var) -> jt.Var:
        """计算骨骼长度 - 按类型分组"""
        if joints.ndim == 2:
            joints = joints.reshape(joints.shape[0], -1, 3)
        
        bone_lengths_by_type = {}
        
        # 按骨骼类型计算长度
        for bone_type, connections in self.human_prior.bone_types.items():
            bone_lengths = []
            for parent, child in connections:
                if parent < joints.shape[1] and child < joints.shape[1]:
                    length = jt.norm(joints[:, child] - joints[:, parent], dim=1)
                    bone_lengths.append(length)
            
            if bone_lengths:
                bone_lengths_by_type[bone_type] = jt.stack(bone_lengths, dim=1)
            else:
                bone_lengths_by_type[bone_type] = jt.zeros((joints.shape[0], 1))
        
        return bone_lengths_by_type
    
    def _compute_bone_length_loss(self, pred_joints: jt.Var, target_joints: jt.Var) -> jt.Var:
        """计算加权骨骼长度损失"""
        pred_bone_lengths = self._compute_bone_lengths(pred_joints)
        target_bone_lengths = self._compute_bone_lengths(target_joints)
        
        total_loss = jt.zeros(1)
        total_weight = 0
        
        for bone_type in pred_bone_lengths.keys():
            pred_lengths = pred_bone_lengths[bone_type]
            target_lengths = target_bone_lengths[bone_type]
            
            # 获取该类型骨骼的权重
            weight = self.human_prior.bone_length_constraints[bone_type]['weight']
            
            # 计算该类型骨骼的长度损失
            type_loss = self.mse_loss(pred_lengths, target_lengths)
            
            total_loss += weight * type_loss
            total_weight += weight
        
        # 返回加权平均损失
        return total_loss / (total_weight + 1e-6)
    
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
        bone_length_loss = self._compute_bone_length_loss(pred_joints, target_joints)
        
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
