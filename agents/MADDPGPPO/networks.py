# networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class PPOActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super(PPOActor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # 添加Layer Normalization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # 分离动作均值和标准差输出
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        
        # 初始化参数
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        
        self.log_std_min = -20
        self.log_std_max = 2
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = self.ln1(x)  # LayerNorm 后应用激活
        x = F.relu(self.fc2(x))
        x = self.ln2(x)
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        
        if deterministic:
            action = torch.tanh(mean)
            # 对于确定性动作，使用较小的固定log_prob
            log_prob = torch.zeros(mean.size(0), 1, device=mean.device)
            return action, log_prob
            
        std = log_std.exp()
        
        # 使用重参数化技巧采样
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        
        # 应用tanh压缩
        action = torch.tanh(x_t)
        
        # 计算log概率
        log_prob = normal.log_prob(x_t)
        # 应用tanh变换的修正
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        # 沿动作维度求和
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, n_agents: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        input_dim = obs_dim * n_agents + act_dim * n_agents
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # 初始化参数
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc4.weight, gain=1.0)
        
    def forward(self, obs_all_agents: torch.Tensor, act_all_agents: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_all_agents, act_all_agents], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化参数
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def compute_gae(rewards: torch.Tensor, 
                values: torch.Tensor, 
                next_values: torch.Tensor, 
                dones: torch.Tensor, 
                gamma: float = 0.99, 
                lambda_: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算广义优势估计(GAE)"""
    # 确保输入具有至少2个维度
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
    if dones.dim() == 1:
        dones = dones.unsqueeze(1)
    if values.dim() == 1:
        values = values.unsqueeze(1)
    if next_values.dim() == 1:
        next_values = next_values.unsqueeze(1)
    
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(rewards.size(0), device=rewards.device)
    
    for t in reversed(range(rewards.size(1))):
        mask = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_values[:, t] * mask - values[:, t]
        gae = delta + gamma * lambda_ * mask * gae
        advantages[:, t] = gae
    
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # 返回1D张量以保持向后兼容
    return advantages.squeeze(1), returns.squeeze(1)


