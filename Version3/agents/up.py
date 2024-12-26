# up.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
import math

# ============================
# 1. 优先经验回放优化
# ============================

# 在up.py最开头，import语句之后添加
Experience = namedtuple("Experience", field_names=["obs", "acts", "rews", "next_obs", "dones", "priority"])

class PrioritizedReplayBuffer:
    def __init__(self, capacity=1000000, alpha=0.6, n_step=1, gamma=0.95, 
                 n_agents=1, n_red=1, n_blue=1, obs_dim=49, act_dim=1):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["obs", "acts", "rews", "next_obs", "dones", "priority"])
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha  # 可调整的优先级参数
        self.n_step = n_step  # 可调整的n-step参数
        self.gamma = gamma
        self.n_agents = n_agents
        self.n_red = n_red
        self.n_blue = n_blue
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_step_buffer = deque(maxlen=n_step)
        
        # 生成agent_ids列表，顺序为红方智能体后蓝方智能体
        self.agent_ids = [f'red_{i}' for i in range(n_red)] + [f'blue_{i}' for i in range(n_blue)]
        self.agent_id_to_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}

    def add(self, obs, acts, rews, next_obs, dones):
        """添加经验到缓冲区"""
        # 预处理数据确保维度一致
        obs_flat = self._process_data(obs, self.obs_dim)
        acts_flat = self._process_data(acts, self.act_dim)
        rews_flat = self._process_data(rews, 1)  # 假设每个智能体的奖励为标量
        next_obs_flat = self._process_data(next_obs, self.obs_dim)
        dones_flat = self._process_data(dones, 1)  # 假设每个智能体的done为标量

        transition = (obs_flat, acts_flat, rews_flat, next_obs_flat, dones_flat)
        self.n_step_buffer.append(transition)
        
        if len(self.n_step_buffer) < self.n_step:
            return
            
        obs_n, acts_n, rews_n, next_obs_n, dones_n = self._get_n_step_info()
        priority = max(self.priorities, default=1.0)
        e = self.experience(obs_n, acts_n, rews_n, next_obs_n, dones_n, priority)
        self.buffer.append(e)
        self.priorities.append(priority)

    def _process_data(self, data, dim):
        """将输入数据标准化为列表格式，按agent_ids顺序处理"""
        if isinstance(data, dict):
            result = []
            for agent_id in self.agent_ids:
                if agent_id in data:
                    value = data[agent_id]
                    if isinstance(value, (list, np.ndarray)):
                        arr = np.array(value).flatten()
                        if len(arr) != dim:
                            if len(arr) > dim:
                                arr = arr[:dim]
                            else:
                                arr = np.pad(arr, (0, dim - len(arr)), 'constant')
                        result.append(arr.tolist())
                    else:
                        # 如果数据是标量，扩展为指定维度
                        result.append([float(value)] + [0.0]*(dim -1))
                else:
                    # 如果该智能体不存在，填充零向量
                    result.append([0.0]*dim)
            return result
        elif isinstance(data, (list, np.ndarray)):
            arr = np.array(data).flatten()
            expected_length = self.n_agents * dim
            if len(arr) < expected_length:
                arr = np.pad(arr, (0, expected_length - len(arr)), 'constant')
            else:
                arr = arr[:expected_length]
            return arr.reshape(self.n_agents, dim).tolist()
        else:
            # 如果数据是单个标量，扩展为n_agents个相同的标量向量
            return [[float(data)]*dim for _ in range(self.n_agents)]
        
    def _get_n_step_info(self):
        """计算n步奖励和下一个状态"""
        obs, acts, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_obs, dones = self.n_step_buffer[-1]
        
        rewards = np.zeros((self.n_agents,))  # 初始化为0向量
        done = False
        
        # 正确处理多智能体奖励
        for idx, (_, _, r, _, d) in enumerate(self.n_step_buffer):
            if done:
                break
            step_rewards = np.array(r).flatten()
            if len(step_rewards) < self.n_agents:
                step_rewards = np.pad(step_rewards, (0, self.n_agents - len(step_rewards)), 'constant')
            rewards += step_rewards * (self.gamma ** idx)
            done = np.any(d) if isinstance(d, (list, np.ndarray)) else d
        
        return obs, acts, rewards.tolist(), next_obs, dones

    def sample(self, batch_size, beta=0.4):
        """采样经验"""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")

        # 计算采样概率
        priorities = np.array([float(p) for p in self.priorities])
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样和重要性采样权重计算
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # 转换为tensor并确保维度正确
        obs = torch.FloatTensor([e.obs for e in experiences])          # [batch_size, n_agents, obs_dim]
        acts = torch.FloatTensor([e.acts for e in experiences])        # [batch_size, n_agents, act_dim]
        rews = torch.FloatTensor([e.rews for e in experiences])        # [batch_size, n_agents]
        next_obs = torch.FloatTensor([e.next_obs for e in experiences])# [batch_size, n_agents, obs_dim]
        dones = torch.FloatTensor([e.dones for e in experiences])      # [batch_size, n_agents]
        weights = torch.FloatTensor(weights).unsqueeze(1)              # [batch_size, 1]

        return obs, acts, rews, next_obs, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# ============================
# 2. 网络架构增强
# ============================

# 增强版Actor网络，加入残差连接、Dropout和LayerNorm
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, dropout=0.2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim, act_dim)
        
        # 残差连接
        self.residual = nn.Linear(obs_dim, hidden_dim)
        
    def forward(self, x):
        residual = F.relu(self.residual(x))
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x += residual  # 残差连接
        return torch.tanh(self.fc4(x))  # 输出范围[-1,1]

# 增强版Critic网络，加入残差连接、Dropout和LayerNorm
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents, hidden_dim=256, dropout=0.3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim * n_agents + act_dim * n_agents, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(hidden_dim, 1)
        
        # 残差连接
        self.residual = nn.Linear(obs_dim * n_agents + act_dim * n_agents, hidden_dim)
        
    def forward(self, obs_all_agents, act_all_agents):
        x = torch.cat([obs_all_agents, act_all_agents], dim=-1)  # [batch_size, n_agents*(obs_dim + act_dim)]
        residual = F.relu(self.residual(x))
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        x += residual  # 残差连接
        return self.fc4(x)

# ============================
# 3. 混合噪声用于探索
# ============================

class MixedNoise:
    def __init__(self, size, mu=0.0, theta=0.25, sigma=0.15, sigma_gaussian=0.07, alpha=0.4):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.sigma_gaussian = sigma_gaussian
        self.alpha = alpha
        # 预计算OU噪声的理论方差
        self.ou_theoretical_var = self.sigma**2 / (2 * self.theta)
        # 初始化移动平均估计器，用于稳定方差
        self.var_ema_alpha = 0.01
        self.var_estimate = self.ou_theoretical_var
        self.reset()

    def reset(self):
        """重置OU状态和方差估计"""
        self.state = np.ones(self.size) * self.mu
        self.var_estimate = self.ou_theoretical_var

    def sample(self):
        """生成混合噪声，同时确保均值和方差的准确性"""
        # 1. 生成基础OU噪声
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.size)
        self.state += dx
        
        # 2. 中心化OU噪声并计算当前方差
        ou_noise = self.state - self.mu
        current_var = np.var(ou_noise) if self.size > 1 else ou_noise[0]**2
        
        # 3. 更新方差估计
        self.var_estimate = (1 - self.var_ema_alpha) * self.var_estimate + \
                           self.var_ema_alpha * current_var
        
        # 4. 归一化OU噪声到理论方差
        scale_factor = np.sqrt(self.ou_theoretical_var / (self.var_estimate + 1e-8))
        ou_noise_scaled = ou_noise * scale_factor
        
        # 5. 生成标准高斯噪声并缩放
        gaussian_noise = self.sigma_gaussian * np.random.randn(self.size)
        
        # 6. 使用理论权重混合噪声
        mixed_noise = (
            np.sqrt(self.alpha) * ou_noise_scaled + 
            np.sqrt(1 - self.alpha) * gaussian_noise
        )
        
        # 7. 确保均值正确
        return mixed_noise + self.mu

# ============================
# 4. 优化后的MADDPG智能体
# ============================

class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, n_agents, agent_id, n_red=1, n_blue=1, 
                 lr_actor=1e-4, lr_critic=1e-3, weight_decay=1e-5, hidden_dim=256, dropout=0.3):
        self.actor = Actor(obs_dim, act_dim, hidden_dim=hidden_dim, dropout=dropout).float()
        self.critic = Critic(obs_dim, act_dim, n_agents, hidden_dim=hidden_dim, dropout=dropout).float()
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim=hidden_dim, dropout=dropout).float()
        self.target_critic = Critic(obs_dim, act_dim, n_agents, hidden_dim=hidden_dim, dropout=dropout).float()
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        
        # 引入学习率调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10000, gamma=0.95)  # 可调整
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10000, gamma=0.95)  # 可调整
        
        self.agent_id = agent_id
        self.noise = MixedNoise(act_dim)  # 使用混合噪声
        
    def select_action(self, obs, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            obs = torch.FloatTensor(obs).unsqueeze(0)  # [1, obs_dim]
            action = self.actor(obs).squeeze(0).numpy()  # [act_dim]
        self.actor.train()
        # 添加探索噪声
        noise = self.noise.sample() * noise_scale
        return np.clip(action + noise, -1, 1)
    
    def reset_noise(self):
        self.noise.reset()
    
    def update_targets(self, tau=0.01):
        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
    def step_schedulers(self):
        """步进学习率调度器"""
        self.actor_scheduler.step()
        self.critic_scheduler.step()

# ============================
# 5. 优化后的MADDPG主类
# ============================

class MADDPG:
    def __init__(self, n_red, n_blue, obs_dim, act_dim, n_step=1, gamma=0.95, 
                 capacity=1000000, alpha=0.6, beta_start=0.4, beta_frames=100000,
                 batch_size=128, lr_actor=1e-4, lr_critic=1e-3, weight_decay=1e-5,
                 dropout=0.3, hidden_dim=256, tau=0.01):
        self.n_agents = n_red + n_blue
        self.n_red = n_red
        self.n_blue = n_blue
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        # 创建红蓝双方智能体
        self.red_agents = [MADDPGAgent(obs_dim, act_dim, self.n_agents, f'red_{i}', 
                                      n_red=n_red, n_blue=n_blue, lr_actor=lr_actor, 
                                      lr_critic=lr_critic, weight_decay=weight_decay, 
                                      hidden_dim=hidden_dim, dropout=dropout) for i in range(n_red)]
        self.blue_agents = [MADDPGAgent(obs_dim, act_dim, self.n_agents, f'blue_{i}', 
                                       n_red=n_red, n_blue=n_blue, lr_actor=lr_actor, 
                                       lr_critic=lr_critic, weight_decay=weight_decay, 
                                       hidden_dim=hidden_dim, dropout=dropout) for i in range(n_blue)]
        self.agents = self.red_agents + self.blue_agents
        
        self.memory = PrioritizedReplayBuffer(
            n_step=n_step, 
            gamma=gamma, 
            n_agents=self.n_agents, 
            n_red=n_red, 
            n_blue=n_blue,
            obs_dim=obs_dim,
            act_dim=act_dim,
            capacity=capacity,
            alpha=alpha
        )
        
        # 基础训练参数
        self.batch_size = batch_size  # 可调整
        self.gamma = gamma  # 可调整
        self.tau = tau  # 可调整
        self.beta_start = beta_start  # 可调整
        self.beta_frames = beta_frames  # 可调整
        self.frame = 1  # 用于beta的线性增长
        
        # 学习率衰减相关
        self.max_frames = beta_frames  # 可调整

    def select_actions(self, obs_dict, noise_scale=0.1):
        actions = {}
        # 红方动作
        for i in range(self.n_red):
            agent_id = f'red_{i}'
            if agent_id in obs_dict:  # 确保智能体还活着
                actions[agent_id] = self.red_agents[i].select_action(obs_dict[agent_id], noise_scale)
                
        # 蓝方动作
        for i in range(self.n_blue):
            agent_id = f'blue_{i}'
            if agent_id in obs_dict:  # 确保智能体还活着
                actions[agent_id] = self.blue_agents[i].select_action(obs_dict[agent_id], noise_scale)
                
        return actions

    def store_transition(self, obs, acts, rews, next_obs, dones):
        self.memory.add(obs, acts, rews, next_obs, dones)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # 动态调整beta值
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, indices, weights = self.memory.sample(self.batch_size, beta)
        
        # obs_batch: [batch_size, n_agents, obs_dim]
        # act_batch: [batch_size, n_agents, act_dim]
        # rew_batch: [batch_size, n_agents]
        # next_obs_batch: [batch_size, n_agents, obs_dim]
        # done_batch: [batch_size, n_agents]
        # weights: [batch_size, 1]
        
        batch_size, n_agents, obs_dim = obs_batch.size()
        _, _, act_dim = act_batch.size()
        
        # 将所有智能体的观测和动作拼接为单一向量
        obs_all_agents = obs_batch.view(batch_size, -1)  # [batch_size, n_agents * obs_dim]
        act_all_agents = act_batch.view(batch_size, -1)  # [batch_size, n_agents * act_dim]
        next_obs_all_agents = next_obs_batch.view(batch_size, -1)  # [batch_size, n_agents * obs_dim]
        
        for agent_i, agent in enumerate(self.agents):
            if agent_i >= self.n_agents:
                continue
                
            # Critic update
            agent.critic_optimizer.zero_grad()
            
            # 计算目标动作
            target_actions = []
            for a_i, target_agent in enumerate(self.agents):
                if a_i >= self.n_agents:
                    continue
                target_act = target_agent.target_actor(next_obs_batch[:, a_i])
                target_actions.append(target_act)
            target_actions = torch.cat(target_actions, dim=1)  # [batch_size, n_agents * act_dim]
            
            # 计算目标Q值
            target_q = rew_batch[:, agent_i].unsqueeze(1) + \
                    (self.gamma ** self.memory.n_step) * agent.target_critic(
                        next_obs_all_agents,
                        target_actions
                    ) * (1 - done_batch[:, agent_i].unsqueeze(1))
            
            # 计算当前Q值
            current_q = agent.critic(
                obs_all_agents,
                act_all_agents
            )
            
            # 计算TD误差
            td_error = current_q - target_q.detach()
            critic_loss = (F.mse_loss(current_q, target_q.detach(), reduction='none') * weights).mean()
            critic_loss.backward()
            
            # 梯度裁剪
            for param in agent.critic.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
            
            agent.critic_optimizer.step()
            
            # 更新优先级
            new_priorities = (td_error.abs().detach().cpu().numpy() + 1e-6).flatten()
            self.memory.update_priorities(indices, new_priorities)
            
            # Actor update
            agent.actor_optimizer.zero_grad()
            current_actions = []
            for a_i, a in enumerate(self.agents):
                if a_i >= self.n_agents:
                    continue
                if a_i == agent_i:
                    current_actions.append(a.actor(obs_batch[:, a_i]))
                else:
                    current_actions.append(act_batch[:, a_i].detach())
            current_actions = torch.cat(current_actions, dim=1)  # [batch_size, n_agents * act_dim]
            
            actor_loss = -agent.critic(
                obs_all_agents,
                current_actions
            ).mean()
            actor_loss.backward()
            
            # 梯度裁剪
            for param in agent.actor.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1.0, 1.0)
            
            agent.actor_optimizer.step()
            
            # 更新目标网络
            agent.update_targets(self.tau)
        
        # 步进学习率调度器
        for agent in self.agents:
            agent.step_schedulers()

    def save(self, path):
        # 保存模型
        save_dict = {
            'red_agents': [agent.actor.state_dict() for agent in self.red_agents],
            'blue_agents': [agent.actor.state_dict() for agent in self.blue_agents],
            'memory_buffer': {
                'observations': [e.obs for e in self.memory.buffer],
                'actions': [e.acts for e in self.memory.buffer],
                'rewards': [e.rews for e in self.memory.buffer],
                'next_observations': [e.next_obs for e in self.memory.buffer],
                'dones': [e.dones for e in self.memory.buffer],
                'priorities': list(self.memory.priorities)
            }
        }
        torch.save(save_dict, path)

    def load(self, path):
        # 加载模型
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        # 加载智能体权重
        for i, state_dict in enumerate(checkpoint['red_agents']):
            self.red_agents[i].actor.load_state_dict(state_dict)
            self.red_agents[i].target_actor.load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['blue_agents']):
            self.blue_agents[i].actor.load_state_dict(state_dict)
            self.blue_agents[i].target_actor.load_state_dict(state_dict)
        
        # 重建记忆缓冲区
        if 'memory_buffer' in checkpoint:
            mb = checkpoint['memory_buffer']
            self.memory.buffer.clear()
            self.memory.priorities.clear()
            
            for i in range(len(mb['observations'])):
                exp = Experience(
                    obs=mb['observations'][i],
                    acts=mb['actions'][i],
                    rews=mb['rewards'][i],
                    next_obs=mb['next_observations'][i],
                    dones=mb['dones'][i],
                    priority=mb['priorities'][i]
                )
                self.memory.buffer.append(exp)
            self.memory.priorities.extend(mb['priorities'])

    def adjust_hyperparameters(self, new_params):
        """动态调整超参数"""
        for param, value in new_params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            for agent in self.agents:
                if hasattr(agent, param):
                    setattr(agent, param, value)
