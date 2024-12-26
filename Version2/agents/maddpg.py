# agents/maddpg.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .actor import Actor
from .critic import Critic
import random
class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha  # 优先级重要性系数
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0  # 记录最大优先级
        
    def add(self, state, action, reward, next_state, done):
        # 使用当前最大优先级作为新经验的优先级
        max_priority = max(0.000001, self.max_priority)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        # 为新样本设置最高优先级
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None
        
        buffer_len = len(self.buffer)
        if buffer_len < self.capacity:
            priorities = self.priorities[:buffer_len]
        else:
            priorities = self.priorities
        
        # 计算抽样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 抽样和重要性采样权重计算
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (buffer_len * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化权重
        samples = [self.buffer[idx] for idx in indices]
        
        return (samples, indices, weights)
    
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            priority = float(priority)  # 确保是标量
            self.priorities[idx] = max(0.000001, priority)
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)

    def size(self):
        """返回当前缓冲区大小"""
        return len(self.buffer)
            
    def __len__(self):
        return len(self.buffer)
# class PrioritizedReplayBuffer:
#     """优先级经验回放缓冲区"""
#     def __init__(self, capacity, alpha=0.6):
#         self.capacity = capacity
#         self.alpha = alpha
#         self.buffer = []
#         self.position = 0
#         self.priorities = np.zeros(capacity)
    
#     def add(self, state, action, reward, next_state, done):
#         max_priority = max(0.000001, np.max(self.priorities) if self.buffer else 1.0)
        
#         if len(self.buffer) < self.capacity:
#             self.buffer.append((state, action, reward, next_state, done))
#         else:
#             self.buffer[self.position] = (state, action, reward, next_state, done)
#         self.priorities[self.position] = max_priority
#         self.position = (self.position + 1) % self.capacity
    
#     def sample(self, batch_size, beta=0.4):
#         if len(self.buffer) == 0:
#             return None
        
#         buffer_len = len(self.buffer)
#         if buffer_len < self.capacity:
#             priorities = self.priorities[:buffer_len]
#         else:
#             priorities = self.priorities
        
#         probs = priorities ** self.alpha
#         probs /= probs.sum()
        
#         indices = np.random.choice(len(self.buffer), batch_size, p=probs)
#         weights = (buffer_len * probs[indices]) ** (-beta)
#         weights /= weights.max()
#         samples = [self.buffer[idx] for idx in indices]
        
#         return (samples, indices, weights)
    
#     def update_priorities(self, indices, priorities):
#         for idx, priority in zip(indices, priorities):
#             self.priorities[idx] = max(0.000001, priority)

#     def size(self):
#         """返回当前缓冲区大小"""
#         return len(self.buffer)
            
#     def __len__(self):
#         return len(self.buffer)

class MADDPG:
    """MADDPG算法实现"""
    def __init__(self, num_agents, state_dim, action_dim, lr=1e-3, gamma=0.95, tau=0.01,
                 buffer_size=int(1e6), batch_size=64, hidden_dim=256, n_step=3,
                 agent_types=None,device='cpu'):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_step = n_step
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 初始化智能体类型
        self.agent_types = agent_types or ['default'] * num_agents
        
        # 初始化网络并移至指定设备
        self.actors = [Actor(state_dim, action_dim, hidden_dim, agent_type=self.agent_types[i]).to(device) 
                    for i in range(num_agents)]
        self.critics = [Critic(state_dim * num_agents, action_dim * num_agents, hidden_dim, num_agents).to(device) 
                    for _ in range(num_agents)]
        self.target_actors = [Actor(state_dim, action_dim, hidden_dim, agent_type=self.agent_types[i]).to(device) 
                            for i in range(num_agents)]
        self.target_critics = [Critic(state_dim * num_agents, action_dim * num_agents, hidden_dim, num_agents).to(device) 
                            for _ in range(num_agents)]
        
        # 初始化优化器
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=lr) 
                               for actor in self.actors]
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=lr) 
                                for critic in self.critics]
        
        # 初始化经验回放
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        
        # 初始化目标网络
        self.hard_update_targets()
        
        # 噪声参数
        self.noise_scale = 1.0
        self.noise_decay = 0.995
        
        # 创建ICM网络和优化器
        self.icm = ICMNetwork(state_dim, action_dim, hidden_dim)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=lr)

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

    # def select_actions(self, states, noise=0.0):
    #     """选择动作"""
    #     actions = []
    #     for i in range(self.num_agents):
    #         state = torch.FloatTensor(states[i]).unsqueeze(0)
            
    #         # 评估模式
    #         self.actors[i].eval()
    #         with torch.no_grad():
    #             action = self.actors[i](state)
    #         self.actors[i].train()
            
    #         # 添加探索噪声
    #         action = action.squeeze(0).numpy()
    #         if isinstance(noise, (int, float)) and noise > 0:  # 修复噪声判断
    #             noise_value = self._ou_noise(action, noise * self.noise_scale)
    #             action += noise_value
    #             action = np.clip(action, -1, 1)
            
    #         actions.append(action)
        
    #     self.noise_scale *= self.noise_decay
    #     return actions
    def select_actions(self, states, noise=0.0):
        """选择动作"""
        actions = []
        for i in range(self.num_agents):
            state = torch.FloatTensor(states[i]).unsqueeze(0)
            
            # 评估模式
            self.actors[i].eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type if torch.cuda.is_available() else 'cpu'):
                    action = self.actors[i](state)
                    # 确保转换回float32类型
                    action = action.to(dtype=torch.float32)
            self.actors[i].train()
            
            # 添加探索噪声
            action = action.squeeze(0).cpu().numpy()  # 先转到CPU再转numpy
            if isinstance(noise, (int, float)) and noise > 0:
                noise_value = self._ou_noise(action, noise * self.noise_scale)
                action += noise_value
                action = np.clip(action, -1, 1)
            
            actions.append(action)

        self.noise_scale *= self.noise_decay
        return actions
    def _compute_curiosity_rewards(self, states, actions, next_states):
        """计算好奇心奖励（兼容性接口）"""
        return self._compute_intrinsic_rewards(states, actions, next_states)

    def _ou_noise(self, x, scale, mu=0., theta=0.15, sigma=0.2):
        """Ornstein-Uhlenbeck过程噪声"""
        return theta * (mu - x) + sigma * np.random.randn(*x.shape) * scale
    
    def _compute_n_step_returns(self, rewards, next_q_values, dones, gamma):
        """计算n步回报"""
        returns = rewards.clone()
        future_return = next_q_values
        
        # 初始化n步回报
        for i in range(1, self.n_step + 1):
            future_return = gamma * future_return * (1 - dones)
            returns = returns + (gamma ** i) * future_return
        
        # 对于终止状态，只使用即时奖励
        returns = torch.where(dones.bool(), rewards, returns)
        
        return returns

    def update(self):
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 采样经验
        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        
        # 转换采样数据的结构
        states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*samples)
        
        # 转换为numpy数组并确保维度正确
        states = torch.FloatTensor(np.array(states_list)).view(self.batch_size, -1)
        next_states = torch.FloatTensor(np.array(next_states_list)).view(self.batch_size, -1)
        actions = torch.FloatTensor(np.array(actions_list)).view(self.batch_size, -1)
        rewards = torch.FloatTensor(np.array(rewards_list)).view(self.batch_size, -1)
        dones = torch.FloatTensor(np.array(dones_list)).view(self.batch_size, 1)
        weights = torch.FloatTensor(weights).view(self.batch_size, 1)
        
        try:
            # 计算内在奖励
            intrinsic_rewards = self._compute_intrinsic_rewards(
                states.view(self.batch_size, self.num_agents, -1),
                actions.view(self.batch_size, self.num_agents, -1),
                next_states.view(self.batch_size, self.num_agents, -1)
            )
            total_rewards = rewards + 0.01 * intrinsic_rewards
            
            # 更新每个智能体的网络
            for agent_idx in range(self.num_agents):
                # 更新Critic
                self.critic_optimizers[agent_idx].zero_grad()
                
                # 计算目标Q值
                next_actions = self._get_target_actions(next_states)
                target_q = self.target_critics[agent_idx](next_states, next_actions)
                target_q = total_rewards[:, agent_idx].unsqueeze(1) + (1 - dones) * self.gamma * target_q
                
                # 计算当前Q值
                current_q = self.critics[agent_idx](states, actions)
                
                # 使用权重更新Critic
                critic_loss = (weights * torch.square(current_q - target_q.detach())).mean()
                critic_loss.backward()
                self.critic_optimizers[agent_idx].step()
                
                # 更新Actor
                self.actor_optimizers[agent_idx].zero_grad()
                current_actions = self._get_current_actions(states, agent_idx)
                actor_loss = -self.critics[agent_idx](states, current_actions).mean()
                actor_loss.backward()
                self.actor_optimizers[agent_idx].step()
                
            # 更新目标网络
            self.soft_update_targets()
            
            # 更新ICM网络
            self._update_icm(states, actions, next_states)
            
        except Exception as e:
            print(f"Error in update: {e}")
            print(f"States shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
            print(f"Rewards shape: {rewards.shape}")
            raise

    # def _compute_intrinsic_rewards(self, states, actions, next_states):
    #     try:
    #         with torch.no_grad():
    #             # 确保输入为tensor并重塑维度
    #             if isinstance(states, np.ndarray):
    #                 states = torch.FloatTensor(states)
    #             if isinstance(actions, np.ndarray):
    #                 actions = torch.FloatTensor(actions)
    #             if isinstance(next_states, np.ndarray):
    #                 next_states = torch.FloatTensor(next_states)
                
    #             # 重塑维度为批处理格式
    #             batch_size = states.size(0)
    #             states = states.reshape(batch_size, -1)
    #             actions = actions.reshape(batch_size, -1)
    #             next_states = next_states.reshape(batch_size, -1)
                
    #             # 计算预测误差
    #             pred_next_states = self.icm.forward_dynamics(states, actions)
    #             pred_actions = self.icm.inverse_dynamics(states, next_states)
                
    #             # 分别计算每个智能体的奖励
    #             rewards = []
    #             for i in range(self.num_agents):
    #                 start_s = i * self.state_dim
    #                 end_s = (i + 1) * self.state_dim
    #                 start_a = i * self.action_dim
    #                 end_a = (i + 1) * self.action_dim
                    
    #                 forward_error = torch.mean(torch.square(
    #                     pred_next_states[:, start_s:end_s] - next_states[:, start_s:end_s]
    #                 ), dim=1)
                    
    #                 inverse_error = torch.mean(torch.square(
    #                     pred_actions[:, start_a:end_a] - actions[:, start_a:end_a]
    #                 ), dim=1)
                    
    #                 intrinsic_reward = forward_error + 0.2 * inverse_error
    #                 rewards.append(intrinsic_reward)
                
    #             # 堆叠所有智能体的奖励
    #             return torch.stack(rewards, dim=1)
                
    #     except Exception as e:
    #         print(f"Error in compute_intrinsic_rewards: {e}")
    #         print(f"States shape: {states.shape}")
    #         print(f"Actions shape: {actions.shape}")
    #         print(f"Next states shape: {next_states.shape}")
    #         raise
    def _compute_intrinsic_rewards(self, states, actions, next_states):
        """计算内在奖励"""
        try:
            with torch.no_grad():
                # 确保输入为tensor
                if isinstance(states, np.ndarray):
                    states = torch.FloatTensor(states)
                if isinstance(actions, np.ndarray):
                    actions = torch.FloatTensor(actions)
                if isinstance(next_states, np.ndarray):
                    next_states = torch.FloatTensor(next_states)
                
                # 移动到正确的设备
                states = states.to(self.device)
                actions = actions.to(self.device)
                next_states = next_states.to(self.device)
                
                # 重塑维度
                batch_size = states.size(0)
                states = states.view(batch_size, self.num_agents, -1)
                actions = actions.view(batch_size, self.num_agents, -1)
                next_states = next_states.view(batch_size, self.num_agents, -1)
                
                # 计算预测
                pred_next_states = self.icm.forward_dynamics(states, actions)
                pred_actions = self.icm.inverse_dynamics(states, next_states)
                
                # 计算误差
                forward_errors = torch.mean(torch.square(
                    pred_next_states.view(batch_size, self.num_agents, -1) - 
                    next_states.view(batch_size, self.num_agents, -1)
                ), dim=2)
                
                inverse_errors = torch.mean(torch.square(
                    pred_actions.view(batch_size, self.num_agents, -1) - 
                    actions.view(batch_size, self.num_agents, -1)
                ), dim=2)
                
                # 综合奖励
                intrinsic_rewards = forward_errors + 0.2 * inverse_errors
                
                return intrinsic_rewards.to(self.device)
                    
        except Exception as e:
            print(f"Error in compute_intrinsic_rewards: {e}")
            print(f"States shape: {states.shape}")
            print(f"Actions shape: {actions.shape}")
            print(f"Next states shape: {next_states.shape}")
            raise

    def _get_target_actions(self, states):
        """获取目标动作"""
        target_actions = []
        for i in range(self.num_agents):
            state_i = states[:, i*self.state_dim:(i+1)*self.state_dim]
            target_action_i = self.target_actors[i](state_i)
            target_actions.append(target_action_i)
        return torch.cat(target_actions, dim=1)

    def _get_current_actions(self, states, agent_idx):
        """获取当前动作"""
        actions = []
        for i in range(self.num_agents):
            state_i = states[:, i*self.state_dim:(i+1)*self.state_dim]
            if i == agent_idx:
                actions.append(self.actors[i](state_i))
            else:
                actions.append(self.actors[i](state_i).detach())
        return torch.cat(actions, dim=1)

    def _update_icm(self, states, actions, next_states):
        """更新ICM网络"""
        self.icm_optimizer.zero_grad()
        
        # 前向动力学损失
        pred_next_states = self.icm.forward_dynamics(states, actions)
        forward_loss = nn.MSELoss()(pred_next_states, next_states)
        
        # 反向动力学损失
        pred_actions = self.icm.inverse_dynamics(states, next_states)
        inverse_loss = nn.MSELoss()(pred_actions, actions)
        
        # 总损失
        total_loss = forward_loss + inverse_loss
        total_loss.backward()
        self.icm_optimizer.step()
    def save(self, path):
        """保存模型"""
        print("Checking hidden_dim:", hasattr(self, 'hidden_dim'), getattr(self, 'hidden_dim', None))
        
        try:
            hidden_dim = getattr(self, 'hidden_dim', 
                            self.actors[0].hidden_dim if hasattr(self.actors[0], 'hidden_dim') else 256)
            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model_weights = {
                'config': {
                    'hidden_dim': hidden_dim,
                    'num_agents': self.num_agents,
                    'state_dim': self.state_dim,
                    'action_dim': self.action_dim,
                    'agent_types': self.agent_types
                },
                'actors': [actor.state_dict() for actor in self.actors],
                'critics': [critic.state_dict() for critic in self.critics],
                'target_actors': [target_actor.state_dict() for target_actor in self.target_actors],
                'target_critics': [target_critic.state_dict() for target_critic in self.target_critics],
                'icm': self.icm.state_dict()
            }
            
            # 分开保存优化器状态
            optimizer_states = {
                'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
                'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
                'icm_optimizer': self.icm_optimizer.state_dict()
            }
            
            torch.save(model_weights, path)
            torch.save(optimizer_states, path + '.opt')  # 优化器状态单独保存
            print(f"Model and optimizer states saved successfully to {path}")
            
        except Exception as e:
            print(f"Error saving model to {path}: {e}")
            import traceback
            traceback.print_exc()
            raise

    def load(self, path):
        """加载模型"""
        try:
            # 加载模型权重
            data = torch.load(path)
            
            # 验证配置
            config = data.get('config', {})
            if config.get('hidden_dim') != self.hidden_dim:
                print(f"Warning: Hidden dim mismatch. Saved: {config.get('hidden_dim')}, Current: {self.hidden_dim}")
            
            # 加载网络权重
            for i in range(self.num_agents):
                self.actors[i].load_state_dict(data['actors'][i])
                self.critics[i].load_state_dict(data['critics'][i])
                self.target_actors[i].load_state_dict(data['target_actors'][i])
                self.target_critics[i].load_state_dict(data['target_critics'][i])
            self.icm.load_state_dict(data['icm'])
            
            # 尝试加载优化器状态（如果存在）
            opt_path = path + '.opt'
            if os.path.exists(opt_path):
                try:
                    opt_data = torch.load(opt_path)
                    for i in range(self.num_agents):
                        self.actor_optimizers[i].load_state_dict(opt_data['actor_optimizers'][i])
                        self.critic_optimizers[i].load_state_dict(opt_data['critic_optimizers'][i])
                    self.icm_optimizer.load_state_dict(opt_data['icm_optimizer'])
                except Exception as e:
                    print(f"Warning: Could not load optimizer states: {e}")
                    print("Continuing with reinitialized optimizers...")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    # def save(self, path):
    #     """保存模型"""
    #     # 添加调试打印
    #     print("Checking hidden_dim:", hasattr(self, 'hidden_dim'), getattr(self, 'hidden_dim', None))
        
    #     try:
    #         # 如果 hidden_dim 不存在，可以尝试从其他地方获取
    #         hidden_dim = getattr(self, 'hidden_dim', 
    #                             self.actors[0].hidden_dim if hasattr(self.actors[0], 'hidden_dim') else 256)
    #         os.makedirs(os.path.dirname(path), exist_ok=True)
    #         data = {
    #             'config': {
    #                 'hidden_dim': hidden_dim,  # 使用获取的 hidden_dim
    #                 'num_agents': self.num_agents,
    #                 'state_dim': self.state_dim,
    #                 'action_dim': self.action_dim,
    #                 'agent_types': self.agent_types
    #             },
    #             'actors': [actor.state_dict() for actor in self.actors],
    #             'critics': [critic.state_dict() for critic in self.critics],
    #             'target_actors': [target_actor.state_dict() for target_actor in self.target_actors],
    #             'target_critics': [target_critic.state_dict() for target_critic in self.target_critics],
    #             'actor_optimizers': [opt.state_dict() for opt in self.actor_optimizers],
    #             'critic_optimizers': [opt.state_dict() for opt in self.critic_optimizers],
    #             'icm': self.icm.state_dict(),
    #             'icm_optimizer': self.icm_optimizer.state_dict()
    #         }
    #         torch.save(data, path, pickle_protocol=4)
    #         print(f"Model saved successfully to {path}")
    #     except Exception as e:
    #         print(f"Error saving model to {path}: {e}")
    #         import traceback
    #         traceback.print_exc()  # 打印完整的错误堆栈
    #         raise



    # def load(self, path):
    #     """加载模型"""
    #     data = torch.load(path)
        
    #     # 验证配置
    #     config = data.get('config', {})
    #     if config.get('hidden_dim') != self.hidden_dim:
    #         print(f"Warning: Hidden dim mismatch. Saved: {config.get('hidden_dim')}, Current: {self.hidden_dim}")
        
    #     try:
    #         for i in range(self.num_agents):
    #             self.actors[i].load_state_dict(data['actors'][i])
    #             self.critics[i].load_state_dict(data['critics'][i])
    #             self.target_actors[i].load_state_dict(data['target_actors'][i])
    #             self.target_critics[i].load_state_dict(data['target_critics'][i])
    #             self.actor_optimizers[i].load_state_dict(data['actor_optimizers'][i])
    #             self.critic_optimizers[i].load_state_dict(data['critic_optimizers'][i])
    #         self.icm.load_state_dict(data['icm'])
    #         self.icm_optimizer.load_state_dict(data['icm_optimizer'])
    #     except Exception as e:
    #         print(f"Error loading model: {e}")
    #         raise


# class ICMNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim):
#         super().__init__()
#         self.state_dim = state_dim
#         self.action_dim = action_dim 
#         self.hidden_dim = hidden_dim
        
#         # 特征编码器 - 根据输入维度调整
#         self.feature_encoder = nn.Sequential(
#             nn.Linear(state_dim * 2, hidden_dim), # 修改输入维度
#             nn.LayerNorm(hidden_dim),  # 添加归一化
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU()
#         )
        
#         # 前向动力学网络
#         self.forward_model = nn.Sequential(
#             nn.Linear(hidden_dim + action_dim * 2, hidden_dim), # 修改输入维度
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, state_dim * 2) # 修改输出维度
#         )
        
#         # 反向动力学网络
#         self.inverse_model = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim * 2) # 修改输出维度
#         )

#     def forward_dynamics(self, states, actions):
#         batch_size = states.size(0)
        
#         # 重塑输入维度
#         states = states.reshape(batch_size, -1)  # 展平状态
#         actions = actions.reshape(batch_size, -1) # 展平动作
        
#         state_features = self.feature_encoder(states)
#         combined = torch.cat([state_features, actions], dim=-1)
#         next_states_pred = self.forward_model(combined)
        
#         return next_states_pred.reshape(batch_size, -1)

#     def inverse_dynamics(self, states, next_states):
#         batch_size = states.size(0)
        
#         # 重塑输入维度
#         states = states.reshape(batch_size, -1)
#         next_states = next_states.reshape(batch_size, -1)
        
#         state_features = self.feature_encoder(states)
#         next_state_features = self.feature_encoder(next_states)
#         combined = torch.cat([state_features, next_state_features], dim=-1)
#         actions_pred = self.inverse_model(combined)
        
#         return actions_pred.reshape(batch_size, -1)

#     def forward(self, states, actions):
#         """兼容旧接口"""
#         return self.forward_dynamics(states, actions)
class ICMNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.hidden_dim = hidden_dim
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 前向动力学网络
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 反向动力学网络
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def encode_state(self, state, num_agents):
        """统一状态编码"""
        batch_size = state.size(0)
        features = []
        
        # 如果状态已经是按智能体分组的形式 [batch, agents, dim]
        if len(state.size()) == 3:
            for i in range(num_agents):
                agent_state = state[:, i]
                features.append(self.feature_encoder(agent_state))
        else:
            # 如果状态是平铺的形式 [batch, agents*dim]
            reshaped_state = state.view(batch_size * num_agents, -1)
            features = self.feature_encoder(reshaped_state)
            features = features.view(batch_size, num_agents, -1)
            
        return features if isinstance(features, torch.Tensor) else torch.stack(features, dim=1)

    def forward_dynamics(self, states, actions):
        batch_size = states.size(0)
        num_agents = states.size(1) if len(states.size()) > 2 else states.shape[1] // self.state_dim
        
        # 编码状态
        state_features = self.encode_state(states, num_agents)
        
        # 处理动作
        reshaped_actions = actions.view(batch_size * num_agents, -1) if len(actions.size()) < 3 else actions.view(batch_size * num_agents, -1)
        
        # 编码状态动作对
        combined = torch.cat([
            state_features.view(batch_size * num_agents, -1),
            reshaped_actions
        ], dim=-1)
        
        # 预测下一个状态
        next_states = self.forward_model(combined)
        return next_states.view(batch_size, num_agents * self.state_dim)

    def inverse_dynamics(self, states, next_states):
        batch_size = states.size(0)
        num_agents = states.size(1) if len(states.size()) > 2 else states.shape[1] // self.state_dim
        
        # 编码当前状态和下一个状态
        current_features = self.encode_state(states, num_agents)
        next_features = self.encode_state(next_states, num_agents)
        
        # 拼接特征
        combined = torch.cat([
            current_features.view(batch_size * num_agents, -1),
            next_features.view(batch_size * num_agents, -1)
        ], dim=-1)
        
        # 预测动作
        actions = self.inverse_model(combined)
        return actions.view(batch_size, num_agents * self.action_dim)

    def forward(self, states, actions):
        """兼容旧接口"""
        return self.forward_dynamics(states, actions)