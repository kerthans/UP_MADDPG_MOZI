import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random

# 基础版Actor网络
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, act_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出范围[-1,1]

# 基础版Critic网络
class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents, hidden_dim=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim * n_agents + act_dim * n_agents, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, obs_all_agents, act_all_agents):
        x = torch.cat([obs_all_agents, act_all_agents], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 基础的经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["obs", "acts", "rews", "next_obs", "dones"])
        
    def add(self, obs, acts, rews, next_obs, dones):
        """添加经验到缓冲区"""
        obs_list = self._dict_to_list(obs)
        next_obs_list = self._dict_to_list(next_obs)
        acts_list = self._dict_to_list(acts)  # 新增
        rews_list = self._dict_to_list(rews)  # 新增
        dones_list = self._dict_to_list(dones)
        
        e = self.experience(obs_list, acts_list, rews_list, next_obs_list, dones_list)
        self.buffer.append(e)
        
    def sample(self, batch_size):
        """从缓冲区采样批量经验"""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")
            
        experiences = random.sample(self.buffer, k=batch_size)
        experiences = [e for e in experiences if e is not None]
        
        # 获取每个经验的维度
        first_exp = experiences[0]
        obs_dim = len(first_exp.obs)
        act_dim = len(first_exp.acts)
        rew_dim = len(first_exp.rews)
        done_dim = len(first_exp.dones)
        
        # 创建适当维度的张量
        obs = torch.FloatTensor([e.obs for e in experiences])  # [batch_size, obs_dim]
        acts = torch.FloatTensor([e.acts for e in experiences])  # [batch_size, act_dim]
        rews = torch.FloatTensor([e.rews for e in experiences])  # [batch_size, rew_dim]
        next_obs = torch.FloatTensor([e.next_obs for e in experiences])  # [batch_size, obs_dim]
        dones = torch.FloatTensor([e.dones for e in experiences])  # [batch_size, done_dim]
        
        return obs, acts, rews, next_obs, dones
    
    def _dict_to_list(self, dict_data):
        """将字典格式的数据转换为列表"""
        if not isinstance(dict_data, dict):
            return self._to_flat_list(dict_data)
            
        result = []
        # 按照固定顺序处理红方和蓝方
        for i in range(100):  # 使用足够大的范围以确保覆盖所有可能的智能体
            red_key = f'red_{i}'
            blue_key = f'blue_{i}'
            
            if red_key in dict_data:
                result.extend(self._to_flat_list(dict_data[red_key]))
            if blue_key in dict_data:
                result.extend(self._to_flat_list(dict_data[blue_key]))
                
            # 如果两个key都不存在，说明已经处理完所有智能体
            if red_key not in dict_data and blue_key not in dict_data:
                break
                
        return result
    
    def _to_flat_list(self, data):
        """确保数据是平坦的列表格式"""
        if isinstance(data, (list, np.ndarray)):
            return list(np.array(data).flatten())
        elif isinstance(data, (int, float)):
            return [float(data)]
        elif isinstance(data, dict):
            # 处理嵌套字典的情况
            return self._dict_to_list(data)
        return [float(data)]
        
    def __len__(self):
        return len(self.buffer)


# MADDPG智能体
class MADDPGAgent:
    def __init__(self, obs_dim, act_dim, n_agents, agent_id, lr_actor=1e-4, lr_critic=1e-3):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim, n_agents)
        self.target_actor = Actor(obs_dim, act_dim)
        self.target_critic = Critic(obs_dim, act_dim, n_agents)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.agent_id = agent_id

    def select_action(self, obs, noise_scale=0.1):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).squeeze().numpy()
        # 添加探索噪声
        action += noise_scale * np.random.randn(*action.shape)
        return np.clip(action, -1, 1)

    def update_targets(self, tau=0.01):
        # 软更新目标网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# MADDPG主类
class MADDPG:
    def __init__(self, n_red, n_blue, obs_dim, act_dim):
        self.n_agents = n_red + n_blue
        self.n_red = n_red
        self.n_blue = n_blue
        
        # 创建红蓝双方智能体
        self.red_agents = [MADDPGAgent(obs_dim, act_dim, self.n_agents, i) for i in range(n_red)]
        self.blue_agents = [MADDPGAgent(obs_dim, act_dim, self.n_agents, i + n_red) for i in range(n_blue)]
        self.agents = self.red_agents + self.blue_agents
        
        self.memory = ReplayBuffer()
        
        # 基础训练参数
        self.batch_size = 128
        self.gamma = 0.95
        self.tau = 0.01
        self.noise_scale = 0.1

    def select_actions(self, obs_dict, noise_scale=None):
        if noise_scale is None:
            noise_scale = self.noise_scale
            
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
                
        # 从经验回放中采样
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = self.memory.sample(self.batch_size)
        n_agents = self.n_red + self.n_blue
        
        # 重塑批次维度以匹配每个智能体的观察
        obs_reshaped = obs_batch.view(self.batch_size, n_agents, -1)  # [batch_size, n_agents, obs_dim]
        next_obs_reshaped = next_obs_batch.view(self.batch_size, n_agents, -1)
        act_reshaped = act_batch.view(self.batch_size, n_agents, -1)
        rew_reshaped = rew_batch.view(self.batch_size, n_agents)
        
        # 正确处理 done_batch 的维度
        if len(done_batch.shape) == 3:
            done_batch = done_batch.squeeze(-1)  # 移除最后一个维度
        if len(done_batch.shape) == 1:
            done_batch = done_batch.unsqueeze(1)
        done_reshaped = done_batch.expand(self.batch_size, n_agents)
        
        # 更新每个智能体
        for agent_i, agent in enumerate(self.agents):
            # 更新critic
            agent.critic_optimizer.zero_grad()
            
            # 收集所有智能体的目标动作
            target_actions = []
            for a_i, target_agent in enumerate(self.agents):
                target_act = target_agent.target_actor(next_obs_reshaped[:, a_i])
                target_actions.append(target_act)
            target_actions = torch.cat(target_actions, dim=1)
            
            # 计算目标Q值
            target_q = rew_reshaped[:, agent_i].unsqueeze(1) + \
                    self.gamma * agent.target_critic(
                        next_obs_reshaped.view(self.batch_size, -1),
                        target_actions
                    ) * (1 - done_reshaped[:, agent_i].unsqueeze(1))
            
            # 计算当前Q值
            current_q = agent.critic(
                obs_reshaped.view(self.batch_size, -1),
                act_reshaped.view(self.batch_size, -1)
            )
            
            # 更新critic
            critic_loss = F.mse_loss(current_q, target_q.detach())
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # 更新actor
            agent.actor_optimizer.zero_grad()
            
            # 收集当前动作
            current_actions = []
            for a_i, a in enumerate(self.agents):
                if a_i == agent_i:
                    current_actions.append(agent.actor(obs_reshaped[:, a_i]))
                else:
                    current_actions.append(act_reshaped[:, a_i].detach())
            current_actions = torch.cat(current_actions, dim=1)
            
            # 使用critic评估actor
            actor_loss = -agent.critic(
                obs_reshaped.view(self.batch_size, -1),
                current_actions
            ).mean()
            
            actor_loss.backward()
            agent.actor_optimizer.step()
            
            # 软更新目标网络
            agent.update_targets(self.tau)


    def save(self, path):
        # 保存模型
        torch.save({
            'red_agents': [agent.actor.state_dict() for agent in self.red_agents],
            'blue_agents': [agent.actor.state_dict() for agent in self.blue_agents]
        }, path)

    def load(self, path):
        # 加载模型
        checkpoint = torch.load(path)
        for i, state_dict in enumerate(checkpoint['red_agents']):
            self.red_agents[i].actor.load_state_dict(state_dict)
            self.red_agents[i].target_actor.load_state_dict(state_dict)
        for i, state_dict in enumerate(checkpoint['blue_agents']):
            self.blue_agents[i].actor.load_state_dict(state_dict)
            self.blue_agents[i].target_actor.load_state_dict(state_dict)