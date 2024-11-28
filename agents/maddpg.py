# agents/maddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor import Actor
from .critic import Critic
from .utils import ReplayBuffer

class MADDPG:
    def __init__(self, num_agents, state_dim, action_dim, lr=1e-3, gamma=0.95, tau=0.01, 
                 buffer_size=int(1e6), batch_size=64, hidden_dim=256):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.lr = lr
        
        # 初始化网络
        self.actors = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents, hidden_dim) 
                       for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim, hidden_dim) for _ in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents, hidden_dim) 
                             for _ in range(num_agents)]
        
        # 初始化目标网络
        self._hard_update_targets()
        
        # 优化器
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) 
                               for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) 
                                for critic in self.critics]
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 噪声衰减
        self.noise_scale = 1.0
        self.noise_decay = 0.995

    def _hard_update_targets(self):
        """硬更新目标网络"""
        for i in range(self.num_agents):
            self.target_actors[i].load_state_dict(self.actors[i].state_dict())
            self.target_critics[i].load_state_dict(self.critics[i].state_dict())

    def _soft_update_targets(self):
        """软更新目标网络"""
        for i in range(self.num_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), 
                                         self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + 
                                      (1.0 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.target_critics[i].parameters(), 
                                         self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + 
                                      (1.0 - self.tau) * target_param.data)

    def select_actions(self, states, noise=0.0):
        """选择动作"""
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            
            # 评估模式
            self.actors[i].eval()
            with torch.no_grad():
                action = self.actors[i](state)
            self.actors[i].train()
            
            # 添加噪声
            action = action.squeeze(0).numpy()
            if noise > 0:
                noise_scale = noise * self.noise_scale
                action += noise_scale * np.random.randn(*action.shape)
                action = np.clip(action, -1, 1)
            
            actions.append(action)
        
        # 衰减噪声
        self.noise_scale *= self.noise_decay
        return actions

    def update(self):
        """更新网络"""
        if self.replay_buffer.size() < self.batch_size:
            return
        
        # 采样batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # 转换为tensor并调整维度
        # [batch_size, num_agents, state_dim] -> [batch_size, num_agents * state_dim]
        states = torch.FloatTensor(states.reshape(self.batch_size, -1))
        next_states = torch.FloatTensor(next_states.reshape(self.batch_size, -1))
        
        # [batch_size, num_agents, action_dim] -> [batch_size, num_agents * action_dim]
        actions = torch.FloatTensor(actions.reshape(self.batch_size, -1))
        
        # [batch_size, num_agents] -> [batch_size, num_agents]
        rewards = torch.FloatTensor(rewards)
        
        # [batch_size] -> [batch_size, 1]
        dones = torch.FloatTensor(dones).view(-1, 1)
        
        for agent_idx in range(self.num_agents):
            # 更新Critic
            self.critic_optimizers[agent_idx].zero_grad()
            
            # 计算下一个状态的动作
            with torch.no_grad():
                next_actions = []
                for i in range(self.num_agents):
                    next_state_i = next_states[:, i*self.state_dim:(i+1)*self.state_dim]
                    next_action_i = self.target_actors[i](next_state_i)
                    next_actions.append(next_action_i)
                next_actions = torch.cat(next_actions, dim=1)
                
                # 计算目标Q值
                target_q = self.target_critics[agent_idx](next_states, next_actions)
                target_q = rewards[:, agent_idx].unsqueeze(1) + (1 - dones) * self.gamma * target_q
            
            # 计算当前Q值
            current_q = self.critics[agent_idx](states, actions)
            
            # Critic损失
            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 1.0)
            self.critic_optimizers[agent_idx].step()
            
            # 更新Actor
            self.actor_optimizers[agent_idx].zero_grad()
            
            # 当前策略动作
            current_actions = []
            for i in range(self.num_agents):
                if i == agent_idx:
                    state_i = states[:, i*self.state_dim:(i+1)*self.state_dim]
                    current_actions.append(self.actors[i](state_i))
                else:
                    current_actions.append(actions[:, i*self.action_dim:(i+1)*self.action_dim].detach())
            current_actions = torch.cat(current_actions, dim=1)
            
            # Actor损失
            actor_loss = -self.critics[agent_idx](states, current_actions).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
            self.actor_optimizers[agent_idx].step()
        
        # 软更新目标网络
        self._soft_update_targets()

    def save(self, path):
        """保存模型"""
        data = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critics': [critic.state_dict() for critic in self.critics],
            'target_actors': [target_actor.state_dict() for target_actor in self.target_actors],
            'target_critics': [target_critic.state_dict() for target_critic in self.target_critics],
            'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
            'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers]
        }
        torch.save(data, path)

    def load(self, path):
        """加载模型"""
        data = torch.load(path)
        for i in range(self.num_agents):
            self.actors[i].load_state_dict(data['actors'][i])
            self.critics[i].load_state_dict(data['critics'][i])
            self.target_actors[i].load_state_dict(data['target_actors'][i])
            self.target_critics[i].load_state_dict(data['target_critics'][i])
            self.actor_optimizers[i].load_state_dict(data['actor_optimizers'][i])
            self.critic_optimizers[i].load_state_dict(data['critic_optimizers'][i])