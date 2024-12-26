import numpy as np
from collections import deque
from typing import Dict, List, Tuple

class CombatMechanics:
    def __init__(self, config):
        self.config = config
        self.episode_stats = deque(maxlen=100)
        self.current_step = 0
        self.reward_weights = {
            'hit': 350.0,
            'be_hit': -120.0,
            'distance': -0.06,
            'survival': 0.12,
            'win': 500.0,
            'lose': -130.0,
            'formation': 0.10,
            'efficiency': 0.30
        }

    def process_attacks(self, actions, red_states, blue_states):
        red_hits = []
        blue_hits = []
        
        # 处理红方攻击
        for i in range(self.config['num_red']):
            if red_states[i, 4] < 0.5:
                continue
                
            action = actions[f'red_{i}']
            if action[2] > 0.3:
                for j in range(self.config['num_blue']):
                    if blue_states[j, 4] < 0.5:
                        continue
                        
                    if self._check_attack_range(red_states[i], blue_states[j]):
                        if np.random.random() < self.config['hit_probability']:
                            blue_states[j, 4] = 0.0
                            red_hits.append(j)
        
        # 处理蓝方攻击
        for i in range(self.config['num_blue']):
            if blue_states[i, 4] < 0.5:
                continue
                
            action = actions[f'blue_{i}']
            if action[2] > 0.5:
                for j in range(self.config['num_red']):
                    if red_states[j, 4] < 0.5:
                        continue
                        
                    if self._check_attack_range(blue_states[i], red_states[j]):
                        if np.random.random() < self.config['hit_probability']:
                            red_states[j, 4] = 0.0
                            blue_hits.append(j)
        
        return red_hits, blue_hits


    def _update_reward_weights(self):
        """极度保守的奖励权重更新策略"""
        if len(self.episode_stats) < 10:
            return
        
        all_stats = list(self.episode_stats)
        recent_stats = all_stats[-min(30, len(all_stats)):]  # 使用更长的历史窗口
        
        avg_steps = np.mean([stat['steps'] for stat in recent_stats])
        avg_red_reward = np.mean([stat['red_reward'] for stat in recent_stats])
        avg_blue_reward = np.mean([stat['blue_reward'] for stat in recent_stats])
        
        # 更保守的性能指标计算
        step_efficiency = 0.5 * np.tanh(avg_steps / self.config['max_steps'])  # 减半效率影响
        reward_balance = np.clip((avg_red_reward - avg_blue_reward) / 8000, -0.08, 0.08)  # 进一步收紧平衡范围
        win_rate = len([s for s in recent_stats if s['red_reward'] > 0]) / len(recent_stats)
        
        # 极小的调整率
        adaptation_rate = np.clip(0.02 + 0.08 * (1 - win_rate), 0.02, 0.1)  # 减半调整幅度
        
        # 大幅降低基础权重
        base_weights = {
            'hit': 200.0,        # 进一步降低击中奖励
            'be_hit': -60.0,     # 减小惩罚
            'distance': -0.04,   # 减小距离权重
            'survival': 0.1,    # 降低存活奖励
            'win': 200.0,        # 降低胜利奖励
            'lose': -80.0,      # 减小失败惩罚
            'formation': 0.08,   # 减小编队权重
            'efficiency': 0.2   # 降低效率权重
        }
        
        # 更保守的动态调整
        for key, base_value in base_weights.items():
            if key in ['hit', 'win']:
                adjustment = 1 + adaptation_rate * (0.5 - win_rate)
                self.reward_weights[key] = np.clip(
                    base_value * adjustment,
                    0.85 * base_value,  # 更窄的调整范围
                    1.15 * base_value
                )
            elif key in ['be_hit', 'lose']:
                adjustment = 1 - adaptation_rate * (win_rate - 0.5)
                self.reward_weights[key] = np.clip(
                    base_value * adjustment,
                    1.15 * base_value,
                    0.85 * base_value
                )
            elif key in ['distance', 'formation', 'efficiency']:
                self.reward_weights[key] = base_value * (1 + 0.15 * step_efficiency)  # 减半效率影响
            else:
                self.reward_weights[key] = base_value
        
        # 更严格的权重范围限制
        for key, value in self.reward_weights.items():
            self.reward_weights[key] = np.clip(float(value), -150.0, 150.0)  # 收紧绝对范围

    def compute_rewards(self, red_states: np.ndarray, blue_states: np.ndarray,
                    red_hits: List[int], blue_hits: List[int]) -> Dict[str, float]:
        """计算红蓝双方的奖励，采用极端的非线性处理和归一化"""
        rewards = {}
        
        # 预计算存活状态
        red_alive = red_states[:, 4] > 0.5
        blue_alive = blue_states[:, 4] > 0.5
        red_alive_count = np.sum(red_alive)
        blue_alive_count = np.sum(blue_alive)
        
        # 预计算距离矩阵
        red_pos = red_states[:, :2]
        blue_pos = blue_states[:, :2]
        distances = np.linalg.norm(red_pos[:, np.newaxis] - blue_pos, axis=2)
        
        # 更极端的距离评估函数
        # def evaluate_distance(dist):
        #     optimal_dist = self.config['attack_range'] * 0.6  # 降低最优距离
        #     x1 = (dist - optimal_dist * 0.4) / (optimal_dist * 0.15)  # 放宽惩罚曲线
        #     x2 = (dist - optimal_dist) / (optimal_dist * 0.2)
        #     # distance_reward = np.tanh(1.5 / (1 + np.exp(x1 * x1)) - 0.8 / (1 + np.exp(-x2 * x2)))
        #     distance_reward = np.tanh(1.5 / (1 + np.exp(x1 * x1)) - 0.8 / (1 + np.exp(-x2 * x2)))
            
        #     # 添加额外的近距离奖励
        #     close_range_bonus = np.exp(-dist / (self.config['attack_range'] * 0.3))
        #     return distance_reward + 0.3 * close_range_bonus
        def evaluate_distance(dist):
            optimal_dist = self.config['attack_range'] * 0.6
            x1 = np.clip((dist - optimal_dist * 0.4) / (optimal_dist * 0.15), -10, 10)
            x2 = np.clip((dist - optimal_dist) / (optimal_dist * 0.2), -10, 10)
            return np.tanh(1.5 / (1 + np.exp(x1)) - 0.8 / (1 + np.exp(-x2)))
        
        # 更激进的非线性累加
        def accumulate_rewards(*rewards):
            # 双重tanh压缩
            return np.tanh(np.tanh(sum(rewards)) * 0.7)
        
        # 更严格的标准化
        def normalize_reward(reward, scale=1.0):
            # 双重压缩
            return np.tanh(np.clip(reward / scale, -1.5, 1.5)) * 0.7
        
        base_hit_reward = self.reward_weights['hit'] * 0.25  # 进一步降低基础奖励
        base_survival_bonus = self.reward_weights['survival'] * 0.15
        
        # 计算红方奖励
        for i in range(self.config['num_red']):
            if red_alive[i]:
                rewards_components = []
                
                # 击中奖励
                if i in red_hits:
                    hit_bonus = 0.5 * np.tanh(len(red_hits) * 0.3)  # 减半连击奖励
                    rewards_components.append(normalize_reward(base_hit_reward * (1 + hit_bonus)))
                
                # 被击中惩罚
                if i in blue_hits:
                    survival_factor = 0.5 * np.tanh(red_alive_count / self.config['num_red'])
                    rewards_components.append(normalize_reward(
                        self.reward_weights['be_hit'] * survival_factor * 0.25))
                
                # 距离奖励
                valid_distances = distances[i][blue_alive]
                if len(valid_distances) > 0:
                    dist_rewards = evaluate_distance(valid_distances)
                    rewards_components.append(normalize_reward(
                        self.reward_weights['distance'] * np.mean(dist_rewards) * 0.25))
                
                # 编队奖励
                formation_reward = self.calculate_formation_reward(i, True, red_states, blue_states)
                team_factor = 0.5 * np.sqrt(red_alive_count / self.config['num_red'])
                rewards_components.append(normalize_reward(
                    self.reward_weights['formation'] * formation_reward * team_factor * 0.15))
                
                # 效率奖励
                efficiency_reward = self._calculate_efficiency()
                rewards_components.append(normalize_reward(
                    self.reward_weights['efficiency'] * efficiency_reward * 0.2))
                
                # 存活奖励
                survival_bonus = 0.5 + 0.5 * np.tanh((self.config['num_red'] - red_alive_count) * 0.1)
                rewards_components.append(normalize_reward(base_survival_bonus * survival_bonus))
                
                # 非线性累加并缩放
                rewards[f'red_{i}'] = accumulate_rewards(*rewards_components) * 50  # 减半最终缩放
            else:
                rewards[f'red_{i}'] = self.reward_weights['be_hit'] * 0.125  # 减半死亡惩罚
        
        # 计算蓝方奖励
        for i in range(self.config['num_blue']):
            if blue_alive[i]:
                rewards_components = []
                
                if i in blue_hits:
                    hit_bonus = 0.5 * np.tanh(len(blue_hits) * 0.25)
                    rewards_components.append(normalize_reward(base_hit_reward * (1 + hit_bonus) * 1.05))
                
                if i in red_hits:
                    survival_factor = 0.5 * np.tanh(blue_alive_count / self.config['num_blue'])
                    rewards_components.append(normalize_reward(
                        self.reward_weights['be_hit'] * survival_factor * 0.25))
                
                valid_distances = distances[:, i][red_alive]
                if len(valid_distances) > 0:
                    dist_rewards = evaluate_distance(valid_distances)
                    rewards_components.append(normalize_reward(
                        self.reward_weights['distance'] * np.mean(dist_rewards) * 0.275))
                
                formation_reward = self.calculate_formation_reward(i, False, red_states, blue_states)
                team_factor = 0.5 * np.sqrt(blue_alive_count / self.config['num_blue'])
                rewards_components.append(normalize_reward(
                    self.reward_weights['formation'] * formation_reward * team_factor * 0.15))
                
                efficiency_reward = self._calculate_efficiency()
                rewards_components.append(normalize_reward(
                    self.reward_weights['efficiency'] * efficiency_reward * 0.2))
                
                survival_bonus = 0.5 + 0.5 * np.tanh((self.config['num_blue'] - blue_alive_count) * 0.125)
                rewards_components.append(normalize_reward(base_survival_bonus * survival_bonus))
                
                rewards[f'blue_{i}'] = accumulate_rewards(*rewards_components) * 50
            else:
                rewards[f'blue_{i}'] = self.reward_weights['be_hit'] * 0.125
        
        # 更保守的胜负奖励
        def victory_reward(alive_count, total_count, base_reward):
            ratio = alive_count / total_count
            return base_reward * 0.25 * (1 + np.tanh(ratio))  # 减半胜利奖励
        
        if blue_alive_count == 0:  # 红方胜利
            victory_value = victory_reward(red_alive_count, self.config['num_red'],
                                        self.reward_weights['win'] * 0.2)
            for i in range(self.config['num_red']):
                if red_alive[i]:
                    rewards[f'red_{i}'] += victory_value
                    
        elif red_alive_count == 0:  # 蓝方胜利
            victory_value = victory_reward(blue_alive_count, self.config['num_blue'],
                                        self.reward_weights['win'] * 0.225)
            for i in range(self.config['num_blue']):
                if blue_alive[i]:
                    rewards[f'blue_{i}'] += victory_value
        
        return rewards


    def _calculate_efficiency(self) -> float:
        """计算效率指标"""
        if len(self.episode_stats) == 0:
            return 0.0
            
        # 计算最近episodes的统计
        recent_stats = list(self.episode_stats)
        avg_steps = np.mean([stat['steps'] for stat in recent_stats])
        avg_red_reward = np.mean([stat['red_reward'] for stat in recent_stats])
        
        # 计算效率指标
        step_efficiency = 1.0 - (avg_steps / self.config['max_steps'])
        reward_efficiency = np.clip((avg_red_reward + 1000) / 2000, 0, 1)
        
        return (step_efficiency + reward_efficiency) / 2.0
    def update_episode_stats(self, rewards: Dict[str, float], current_step: int = None):
        """更新episode统计信息"""
        if current_step is not None:
            self.current_step = current_step
            
        # 计算奖励
        red_rewards = [rewards.get(f'red_{i}', 0.0) for i in range(self.config['num_red'])]
        blue_rewards = [rewards.get(f'blue_{i}', 0.0) for i in range(self.config['num_blue'])]
        
        stats = {
            'steps': self.current_step,
            'red_reward': sum(red_rewards),
            'blue_reward': sum(blue_rewards)
        }
        
        self.episode_stats.append(stats)
        
        # 动态更新奖励权重
        if len(self.episode_stats) >= 10:
            self._update_reward_weights()

    def _check_attack_range(self, attacker, target):
        dist = np.linalg.norm(attacker[:2] - target[:2])
        if dist < 1e-10:
            return True
        return dist <= self.config['attack_range']

    def calculate_formation_reward(self, agent_idx: int, is_red: bool, red_states: np.ndarray, blue_states: np.ndarray) -> float:
        """优化的编队奖励计算
        
        Args:
            agent_idx: 智能体索引
            is_red: 是否为红方智能体
            red_states: 红方状态矩阵
            blue_states: 蓝方状态矩阵
            
        Returns:
            float: 编队奖励值
        """
        states = red_states if is_red else blue_states
        num_agents = self.config['num_red'] if is_red else self.config['num_blue']
        
        if num_agents < 2:
            return 0.0
            
        # 计算编队中心
        alive_agents = states[:, 4] > 0.5
        if not np.any(alive_agents):
            return 0.0
            
        formation_center = np.mean(states[alive_agents, :2], axis=0)
        
        # 计算与编队中心的距离
        agent_pos = states[agent_idx, :2]
        dist_to_center = np.linalg.norm(agent_pos - formation_center)
        
        # 计算理想编队半径（基于攻击范围和队伍规模）
        ideal_radius = self.config['attack_range'] * (0.3 + 0.1 * np.log2(num_agents))
        
        # 计算编队奖励
        formation_reward = 1.0 - np.clip(abs(dist_to_center - ideal_radius) / ideal_radius, 0, 1)
        
        # 添加朝向一致性奖励
        heading_consistency = np.mean(np.cos(states[alive_agents, 2] - states[agent_idx, 2]))
        
        return 0.7 * formation_reward + 0.3 * heading_consistency

    def calculate_threat_level(self, own_state, opponent_state):
        """计算威胁等级（避免除零）"""
        rel_pos = opponent_state[:2] - own_state[:2]
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-10:  # 避免除零
            return 1.0
            
        distance_threat = 1.0 - np.clip(dist / (2 * self.attack_range), 0, 1)
        
        rel_velocity = opponent_state[3] * np.array([
            np.cos(opponent_state[2]),
            np.sin(opponent_state[2])
        ])
        own_velocity = own_state[3] * np.array([
            np.cos(own_state[2]),
            np.sin(own_state[2])
        ])
        
        # 安全的速度威胁计算
        closing_speed = -np.dot(rel_pos/max(dist, 1e-10), rel_velocity - own_velocity)
        speed_threat = np.clip(closing_speed / (2 * self.max_speed), -1, 1)
        
        return (distance_threat + max(0, speed_threat)) / 2.0


    def _validate_config(self):
        """验证配置参数的有效性"""
        required_keys = ['num_red', 'num_blue', 'max_steps', 'field_size', 
                        'attack_range', 'min_speed', 'max_speed', 'max_turn_rate']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config parameter: {key}")