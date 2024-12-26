# agents/actor.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """改进的多头注意力机制，支持动态输入维度"""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 分离QKV投影以更好地控制维度
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 添加LayerNorm提高稳定性
        self.norm = nn.LayerNorm(dim)
        
    def _reshape_for_attention(self, x, batch_size):
        # 重塑输入张量以进行多头注意力计算
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # [batch, heads, seq, dim]
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 应用LayerNorm
        x = self.norm(x)
        
        # 独立投影Q、K、V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑维度用于注意力计算
        q = self._reshape_for_attention(q, batch_size)
        k = self._reshape_for_attention(k, batch_size)
        v = self._reshape_for_attention(v, batch_size)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, seq_len, self.dim)
        
        return self.out_proj(out)

class Actor(nn.Module):
    """改进的Actor网络，支持异构智能体"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, agent_type=None):
        super(Actor, self).__init__()
        self.agent_type = agent_type
        
        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 多头注意力层
        self.attention = MultiHeadAttention(hidden_dim)
        
        # 策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        # 根据智能体类型调整网络参数
        if agent_type:
            self._adjust_for_type(agent_type)
        
        # 使用改进的初始化
        self.apply(self._init_weights)
    
    def _adjust_for_type(self, agent_type):
        """根据智能体类型调整网络参数"""
        if agent_type == 'scout':
            nn.init.kaiming_normal_(self.feature_net[0].weight, nonlinearity='relu')
        elif agent_type == 'bomber':
            self.policy_net[-2].weight.data.mul_(0.1)
    
    def _init_weights(self, m):
        """改进的权重初始化"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        # 特征提取
        features = self.feature_net(state)
        
        # 应用注意力机制
        if len(features.shape) == 2:
            features = features.unsqueeze(1)
        attended = self.attention(features).squeeze(1)
        
        # 生成动作
        return self.policy_net(attended)