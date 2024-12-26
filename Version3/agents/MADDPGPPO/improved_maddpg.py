# improved_maddpg.py

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
from typing import Dict, List, Tuple
import logging
from agents.MADDPGPPO.networks import PPOActor, Critic, ValueNet, compute_gae

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 1000000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = 0.4
        self.beta_increment = 0.001

        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", 
            field_names=["obs", "acts", "rews", "next_obs", "dones", "log_probs"])
        
        self._return_log_probs = False  # 控制是否返回 log_probs

    def add(self, obs, acts, rews, next_obs, dones, log_probs=None):
        """添加新经验"""
        obs_list = self._dict_to_list(obs)
        next_obs_list = self._dict_to_list(next_obs)
        dones_list = self._dict_to_list(dones)
        
        e = self.experience(obs_list, acts, rews, next_obs_list, dones_list, log_probs)
        self.buffer.append(e)
        
        # 新样本设置最大优先级
        max_priority = max([p for p in self.priorities]) if self.priorities else 1.0
        self.priorities.append(max_priority)
        
        logging.debug(f"Added experience. Buffer size: {len(self.buffer)}")
    
    def sample(self, batch_size: int):
        """从缓冲区采样批量经验"""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in buffer")
            
        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        # 确保所有经验都是完整的
        experiences = [e for e in experiences if e is not None]
        
        # 处理观察和下一个观察
        obs = torch.FloatTensor([e.obs for e in experiences])
        next_obs = torch.FloatTensor([e.next_obs for e in experiences])
        
        # 处理动作
        acts = torch.FloatTensor([self._dict_to_list(e.acts) for e in experiences])
        
        # 处理奖励
        rews = torch.FloatTensor([self._dict_to_list(e.rews) for e in experiences])
        
        # 处理完成状态
        dones_list = [self._dict_to_list(e.dones) for e in experiences]
        max_len = max(len(d) for d in dones_list)
        dones_list = [d + [0.0] * (max_len - len(d)) for d in dones_list]
        dones = torch.FloatTensor(dones_list)
        
        # 处理 log_probs
        if self._return_log_probs:
            if experiences[0].log_probs is not None:
                log_probs = torch.FloatTensor([self._dict_to_list(e.log_probs) for e in experiences])
            else:
                log_probs = None
            logging.debug("Sampled batch with log_probs")
            return obs, acts, rews, next_obs, dones, indices, weights, log_probs
        else:
            logging.debug("Sampled batch without log_probs")
            return obs, acts, rews, next_obs, dones, indices, weights

    def _to_flat_list(self, data):
        """确保数据是平坦的列表格式"""
        if isinstance(data, (list, np.ndarray)):
            return list(np.array(data).flatten())
        return [data]
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority) ** 2 + 1e-6  # 使用 TD 误差的平方并避免零优先级
    
    def _dict_to_list(self, dict_data):
        if not isinstance(dict_data, dict):
            return dict_data
                
        result = []
        for i in range(100):
            red_key = f'red_{i}'
            blue_key = f'blue_{i}'
            
            if red_key in dict_data:
                result.extend(self._to_flat_list(dict_data[red_key]))
            if blue_key in dict_data:
                result.extend(self._to_flat_list(dict_data[blue_key]))
                
            if red_key not in dict_data and blue_key not in dict_data:
                break
        return result
    
    def __len__(self):
        return len(self.buffer)

class ImprovedMADDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, n_agents: int, agent_id: int,
                 lr_actor: float = 1e-4, lr_critic: float = 1e-3, lr_value: float = 1e-3):
        self.actor = PPOActor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim, n_agents)
        self.value_net = ValueNet(obs_dim)
        
        self.target_critic = Critic(obs_dim, act_dim, n_agents)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        self.agent_id = agent_id
        
        # PPO超参数
        self.clip_param = 0.2
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        obs = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob = self.actor.sample_action(obs, deterministic)
        return action.squeeze().numpy(), log_prob.squeeze().item()

    def update_ppo(self, obs: torch.Tensor, actions: torch.Tensor, 
                  advantages: torch.Tensor, old_log_probs: torch.Tensor,
                  returns: torch.Tensor, weights: torch.Tensor):
        """
        更新 PPO Actor 和 Value 网络
        """
        # 获取当前策略的均值和标准差
        mean, log_std = self.actor(obs)
        std = log_std.exp()
        
        # 根据当前策略分布采样新的动作和计算对应的 log_prob
        new_actions, new_log_probs = self.actor.sample_action(obs)
        
        # 计算策略比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 裁剪策略比率
        surr1 = ratio * advantages * weights
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages * weights
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 计算策略分布的熵
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().sum(-1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # 总损失为策略损失加熵正则化
        total_loss = actor_loss + entropy_loss
        
        # 反向传播和优化
        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # 更新 Value 网络
        value_pred = self.value_net(obs)
        value_loss = F.mse_loss(value_pred, returns)
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return actor_loss.item(), value_loss.item()

class ImprovedMADDPG:
    def __init__(self, n_red: int, n_blue: int, obs_dim: int, act_dim: int):
        self.n_agents = n_red + n_blue
        self.n_red = n_red
        self.n_blue = n_blue
        
        self.red_agents = [ImprovedMADDPGAgent(obs_dim, act_dim, self.n_agents, i) 
                          for i in range(n_red)]
        self.blue_agents = [ImprovedMADDPGAgent(obs_dim, act_dim, self.n_agents, i + n_red)
                           for i in range(n_blue)]
        self.agents = self.red_agents + self.blue_agents
        
        self.memory = PrioritizedReplayBuffer()
        
        # 训练参数
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.01
        self.gae_lambda = 0.95
        self.ppo_epochs = 3
        
    def select_actions(self, obs_dict: Dict, deterministic: bool = False) -> Dict:
        actions = {}
        log_probs = {}
        
        # 红方动作
        for i in range(self.n_red):
            agent_id = f'red_{i}'
            if agent_id in obs_dict:
                action, log_prob = self.red_agents[i].select_action(
                    obs_dict[agent_id], deterministic)
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
        
        # 蓝方动作
        for i in range(self.n_blue):
            agent_id = f'blue_{i}'
            if agent_id in obs_dict:
                action, log_prob = self.blue_agents[i].select_action(
                    obs_dict[agent_id], deterministic)
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
        
        return actions, log_probs
    
    def compute_shaped_rewards(self, obs_dict: Dict, actions: Dict, 
                                next_obs_dict: Dict, base_rewards: Dict) -> Dict:
        """平滑的奖励塑形函数"""
        shaped_rewards = {}
        
        # 提取位置和状态信息
        red_positions = {k: np.array(obs_dict[k][:2]) for k in obs_dict if k.startswith('red_')}
        blue_positions = {k: np.array(obs_dict[k][:2]) for k in obs_dict if k.startswith('blue_')}
        
        # 计算初始和当前存活数量
        initial_red = self.n_red
        initial_blue = self.n_blue
        current_red = len(red_positions)
        current_blue = len(blue_positions)
        
        # 归一化参数
        MAX_DIST = 200.0
        OPTIMAL_RANGE = 50.0  # 最佳射击距离
        DANGER_RANGE = 30.0   # 危险距离
        
        for agent_id in base_rewards.keys():
            is_red = agent_id.startswith('red_')
            pos = red_positions[agent_id] if is_red else blue_positions[agent_id]
            enemy_positions = blue_positions if is_red else red_positions
            
            # 1. 基础奖励
            base_reward = np.clip(base_rewards[agent_id], -1.0, 1.0)
            
            # 2. 距离奖励
            if enemy_positions:
                enemy_dists = [np.linalg.norm(pos - e_pos) for e_pos in enemy_positions.values()]
                min_dist = min(enemy_dists)
                
                if is_red:
                    if min_dist < DANGER_RANGE:
                        dist_reward = 1.0  # 接近敌人
                    elif min_dist < OPTIMAL_RANGE:
                        dist_reward = 0.5
                    else:
                        dist_reward = -0.5 * (min_dist - OPTIMAL_RANGE) / MAX_DIST
                else:
                    if min_dist < DANGER_RANGE:
                        dist_reward = -1.0
                    elif min_dist < OPTIMAL_RANGE * 1.5:
                        dist_reward = 0.5
                    else:
                        dist_reward = 0.0
            else:
                dist_reward = -0.5
            
            # 3. 击杀奖励
            if is_red:
                kill_reward = 2.0 * (initial_blue - current_blue) / initial_blue
                death_penalty = -1.5 * (initial_red - current_red) / initial_red
            else:
                kill_reward = 1.5 * (initial_red - current_red) / initial_red
                death_penalty = -2.0 * (initial_blue - current_blue) / initial_blue
            
            # 4. 生存时间奖励
            survival_bonus = 0.1
            
            # 5. 攻击意愿奖励
            attack_willingness = 0.0
            if is_red and min_dist < OPTIMAL_RANGE * 1.2:
                attack_willingness = 1.0
            elif not is_red and DANGER_RANGE < min_dist < OPTIMAL_RANGE * 1.5:
                attack_willingness = 0.5
            
            # 组合所有奖励
            total_reward = (
                0.2 * base_reward +
                0.25 * dist_reward +
                0.25 * (kill_reward + death_penalty) +
                0.15 * survival_bonus +
                0.15 * attack_willingness
            )
            
            # 稳定性处理
            total_reward = np.clip(total_reward, -3.0, 3.0)
            if np.isnan(total_reward):
                total_reward = -0.5
            
            shaped_rewards[agent_id] = float(total_reward)
        
        return shaped_rewards

    def store_transition(self, obs: Dict, acts: Dict, rews: Dict, 
                        next_obs: Dict, dones: Dict, log_probs: Dict = None):
        """存储经验"""
        self.memory.add(obs, acts, rews, next_obs, dones, log_probs)
    
    def train(self):
        """训练算法"""
        if len(self.memory) < self.batch_size:
            return
        
        # 设置标志以获取完整返回值
        self.memory._return_log_probs = True
        
        # 从优先级经验回放中采样
        sample_results = self.memory.sample(self.batch_size)
        if len(sample_results) == 8:
            obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, indices, weights, log_probs_batch = sample_results
        else:
            raise ValueError("Sampled results do not contain log_probs")
            
        n_agents = self.n_red + self.n_blue
        
        # 重塑批次维度，确保维度正确
        obs_reshaped = obs_batch.view(self.batch_size, n_agents, -1)
        next_obs_reshaped = next_obs_batch.view(self.batch_size, n_agents, -1)
        act_reshaped = act_batch.view(self.batch_size, n_agents, -1)
        
        # 处理 log_probs
        if log_probs_batch is not None:
            log_probs_reshaped = log_probs_batch.view(self.batch_size, n_agents, -1)
        else:
            raise ValueError("Log_probs are not available in the sampled batch")
        
        # 确保done_batch维度正确
        if done_batch.size(1) != n_agents:
            done_batch = done_batch.repeat(1, n_agents // done_batch.size(1))
        done_reshaped = done_batch.view(self.batch_size, n_agents)
        
        rew_batch = rew_batch.view(self.batch_size, n_agents)
        
        # 计算每个智能体的TD误差和进行更新
        td_errors = []
        for agent_i, agent in enumerate(self.agents):
            # 1. 更新critic
            agent.critic_optimizer.zero_grad()
            
            # 获取目标动作
            target_actions = []
            for a_i, target_agent in enumerate(self.agents):
                with torch.no_grad():
                    action, _ = target_agent.actor.sample_action(
                        next_obs_reshaped[:, a_i], deterministic=True)
                target_actions.append(action)
            target_actions = torch.cat(target_actions, dim=1)
            
            # 计算目标Q值
            target_q = rew_batch[:, agent_i].unsqueeze(1) + \
                    self.gamma * agent.target_critic(
                        next_obs_reshaped.view(self.batch_size, -1),
                        target_actions
                    ) * (1 - done_reshaped[:, agent_i].unsqueeze(1))
            
            # 计算当前Q值
            current_q = agent.critic(
                obs_reshaped.view(self.batch_size, -1),
                act_reshaped.view(self.batch_size, -1)
            )
            
            # 计算TD误差
            td_error = (target_q.detach() - current_q).abs().mean(dim=1).detach().cpu().numpy()
            td_errors.append(td_error)
            
            # 应用重要性采样权重
            critic_loss = (weights * F.mse_loss(current_q, target_q.detach(), reduction='none')).mean()
            
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # 2. PPO更新
            values = agent.value_net(obs_reshaped[:, agent_i])
            next_values = agent.value_net(next_obs_reshaped[:, agent_i])
            advantages, returns = compute_gae(
                rew_batch[:, agent_i].unsqueeze(1),
                values.detach(),
                next_values.detach(),
                done_reshaped[:, agent_i].unsqueeze(1),
                self.gamma,
                self.gae_lambda
            )
            
            # 多次PPO更新
            actor_loss_total = 0.0
            value_loss_total = 0.0
            for _ in range(self.ppo_epochs):
                actor_loss, value_loss = agent.update_ppo(
                    obs_reshaped[:, agent_i],
                    act_reshaped[:, agent_i],
                    advantages,
                    log_probs_reshaped[:, agent_i].squeeze(-1),  # 正确传递 old_log_probs
                    returns,
                    weights
                )
                actor_loss_total += actor_loss
                value_loss_total += value_loss
            
            # 记录平均损失
            logging.info(f"Agent {agent_i} - Actor Loss: {actor_loss_total / self.ppo_epochs:.4f}, Value Loss: {value_loss_total / self.ppo_epochs:.4f}")
            
            # 软更新目标网络
            for target_param, param in zip(agent.target_critic.parameters(), 
                                            agent.critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )
            logging.info(f"Agent {agent_i} - Target Critic updated with tau={self.tau}")
        
        # 更新优先级
        td_errors = np.mean(np.stack(td_errors), axis=0)
        self.memory.update_priorities(indices, td_errors)
        
        # 记录平均TD误差
        avg_td_error = np.mean(td_errors)
        logging.info(f"Average TD Error: {avg_td_error:.4f}")

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'red_agents': [
                {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'value': agent.value_net.state_dict()
                } for agent in self.red_agents
            ],
            'blue_agents': [
                {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'value': agent.value_net.state_dict()
                } for agent in self.blue_agents
            ],
            'memory_buffer': self.memory.buffer,
            'memory_priorities': self.memory.priorities
        }, path)
        logging.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        
        for i, state_dict in enumerate(checkpoint['red_agents']):
            self.red_agents[i].actor.load_state_dict(state_dict['actor'])
            self.red_agents[i].critic.load_state_dict(state_dict['critic'])
            self.red_agents[i].value_net.load_state_dict(state_dict['value'])
            self.red_agents[i].target_critic.load_state_dict(state_dict['critic'])
            
        for i, state_dict in enumerate(checkpoint['blue_agents']):
            self.blue_agents[i].actor.load_state_dict(state_dict['actor'])
            self.blue_agents[i].critic.load_state_dict(state_dict['critic'])
            self.blue_agents[i].value_net.load_state_dict(state_dict['value'])
            self.blue_agents[i].target_critic.load_state_dict(state_dict['critic'])
        
        # 恢复缓冲区
        self.memory.buffer = checkpoint['memory_buffer']
        self.memory.priorities = checkpoint['memory_priorities']
        logging.info("Model loaded successfully")

