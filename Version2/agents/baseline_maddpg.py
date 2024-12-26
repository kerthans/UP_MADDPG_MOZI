# agents/baseline_maddpg.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleActor(nn.Module):
    """基础Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SimpleActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.net(state)

class SimpleCritic(nn.Module):
    """基础Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SimpleCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class SimpleReplayBuffer:
    """基础经验回放缓冲区"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def size(self):
        return len(self.buffer)

class BaselineMADDPG:
    """基础版MADDPG实现"""
    def __init__(self, num_agents, state_dim, action_dim, lr=1e-3, gamma=0.95, tau=0.01,
                 buffer_size=int(1e6), batch_size=64, hidden_dim=256):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # 初始化网络
        self.actors = [SimpleActor(state_dim, action_dim, hidden_dim) 
                      for _ in range(num_agents)]
        self.critics = [SimpleCritic(state_dim * num_agents, action_dim * num_agents, hidden_dim) 
                       for _ in range(num_agents)]
        self.target_actors = [SimpleActor(state_dim, action_dim, hidden_dim) 
                            for _ in range(num_agents)]
        self.target_critics = [SimpleCritic(state_dim * num_agents, action_dim * num_agents, hidden_dim) 
                             for _ in range(num_agents)]
        
        # 初始化优化器
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) 
                               for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) 
                                for critic in self.critics]
        
        # 初始化经验回放
        self.replay_buffer = SimpleReplayBuffer(buffer_size)
        
        # 初始化目标网络
        self.hard_update_targets()
    
    def select_actions(self, states, noise=0.0):
        """选择动作"""
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            self.actors[i].eval()
            with torch.no_grad():
                action = self.actors[i](state)
            self.actors[i].train()
            action = action.squeeze(0).numpy()
            if noise > 0:
                action += np.random.normal(0, noise, size=action.shape)
                action = np.clip(action, -1, 1)
            actions.append(action)
        return actions

    def update(self):
        """更新网络"""
        if self.replay_buffer.size() < self.batch_size:
            return
            
        # 采样经验
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # 更新每个智能体的网络
        for agent_idx in range(self.num_agents):
            # 更新Critic
            self.critic_optimizers[agent_idx].zero_grad()
            
            next_actions = []
            for i in range(self.num_agents):
                next_state_i = next_states[:, i*self.state_dim:(i+1)*self.state_dim]
                next_actions.append(self.target_actors[i](next_state_i))
            next_actions = torch.cat(next_actions, dim=1)
            
            target_q = rewards[:, agent_idx].unsqueeze(1) + \
                      (1 - dones) * self.gamma * \
                      self.target_critics[agent_idx](next_states, next_actions)
            
            current_q = self.critics[agent_idx](states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            critic_loss.backward()
            self.critic_optimizers[agent_idx].step()
            
            # 更新Actor
            self.actor_optimizers[agent_idx].zero_grad()
            
            current_actions = []
            for i in range(self.num_agents):
                state_i = states[:, i*self.state_dim:(i+1)*self.state_dim]
                if i == agent_idx:
                    current_actions.append(self.actors[i](state_i))
                else:
                    current_actions.append(self.actors[i](state_i).detach())
            current_actions = torch.cat(current_actions, dim=1)
            
            actor_loss = -self.critics[agent_idx](states, current_actions).mean()
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()
        
        # 软更新目标网络
        self.soft_update_targets()

    def hard_update_targets(self):
        """硬更新目标网络"""
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

    def soft_update_targets(self):
        """软更新目标网络"""
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), 
                                         self.actors[i].parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for target_param, param in zip(self.target_critics[i].parameters(), 
                                         self.critics[i].parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )