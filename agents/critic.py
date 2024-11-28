# agents/critic.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # 首先处理状态
        self.state_fc1 = nn.Linear(state_dim, hidden_dim)
        self.state_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 处理动作
        self.action_fc1 = nn.Linear(action_dim, hidden_dim)
        
        # 合并处理
        self.joint_fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.joint_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.joint_fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Layer Normalization代替Batch Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state, action):
        # 处理状态
        s = F.relu(self.ln1(self.state_fc1(state)))
        s = F.relu(self.ln2(self.state_fc2(s)))
        
        # 处理动作
        a = F.relu(self.action_fc1(action))
        
        # 合并状态和动作
        x = torch.cat([s, a], dim=1)
        
        # 联合处理
        x = F.relu(self.ln3(self.joint_fc1(x)))
        x = F.relu(self.joint_fc2(x))
        x = self.joint_fc3(x)
        
        return x