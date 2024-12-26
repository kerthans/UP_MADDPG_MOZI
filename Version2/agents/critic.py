# agents/critic.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.actor import MultiHeadAttention

class Critic(nn.Module):
    """改进的Critic网络，优化特征处理和注意力机制"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_agents=1):
        super(Critic, self).__init__()
        self.state_dim = state_dim // num_agents
        self.action_dim = action_dim // num_agents
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 多头注意力层
        self.attention = MultiHeadAttention(hidden_dim)
        
        # 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Q值预测层
        self.q_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, state, action):
        try:
            batch_size = state.size(0)
            
            # 重塑输入
            states = state.view(batch_size, self.num_agents, -1)
            actions = action.view(batch_size, self.num_agents, -1)
            
            # 处理每个智能体的特征
            encoded_states = []
            encoded_actions = []
            
            for i in range(self.num_agents):
                # 编码状态和动作
                state_i = self.state_encoder(states[:, i])
                action_i = self.action_encoder(actions[:, i])
                
                encoded_states.append(state_i)
                encoded_actions.append(action_i)
            
            # 堆叠特征
            encoded_states = torch.stack(encoded_states, dim=1)  # [batch, num_agents, hidden]
            encoded_actions = torch.stack(encoded_actions, dim=1)  # [batch, num_agents, hidden]
            
            # 合并状态和动作特征
            joint_features = torch.cat([encoded_states, encoded_actions], dim=-1)  # [batch, num_agents, 2*hidden]
            
            # 融合特征
            fused_features = self.fusion_layer(joint_features)  # [batch, num_agents, hidden]
            
            # 应用注意力机制
            attended_features = self.attention(fused_features)  # [batch, num_agents, hidden]
            
            # 全局特征聚合
            global_features = torch.mean(attended_features, dim=1)  # [batch, hidden]
            
            # 预测Q值
            q_value = self.q_predictor(global_features)
            return q_value
            
        except Exception as e:
            print(f"Error in Critic forward: {e}")
            print(f"State shape: {state.shape}")
            print(f"Action shape: {action.shape}")
            raise