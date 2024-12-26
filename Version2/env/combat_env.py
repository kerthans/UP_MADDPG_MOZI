# env/combat_env.py

import numpy as np
import gym 
from gym import spaces
from typing import Dict, List, Tuple, Optional

class CombatEnv(gym.Env):
    """
    多智能体空战环境，支持异构智能体
    """
    def __init__(self, 
                num_red: int = 3,
                num_blue: int = 3,
                state_dim: int = 9,  # 更新为9维状态空间
                action_dim: int = 2,
                max_steps: int = 200,
                red_agent_types: Optional[List[str]] = None,
                blue_agent_types: Optional[List[str]] = None,
                heterogeneous: bool = False):
        """
        参数初始化
        :param num_red: 红方智能体数量
        :param num_blue: 蓝方智能体数量
        :param state_dim: 状态空间维度
        :param action_dim: 动作空间维度
        :param max_steps: 最大步数
        :param red_agent_types: 红方智能体类型列表
        :param blue_agent_types: 蓝方智能体类型列表
        :param heterogeneous: 是否启用异构
        """
        super().__init__()
        self.num_red = num_red
        self.num_blue = num_blue
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_agents = num_red + num_blue
        self.max_steps = max_steps
        self.heterogeneous = heterogeneous

        # 初始化上一步速度记录(用于计算加速度)
        self.last_red_velocities = None
        self.last_blue_velocities = None

        # 异构智能体参数
        self.agent_types = {
            'scout': {
                'max_velocity': 6.0,    # 侦察机速度快
                'attack_range': 3.0,    # 攻击范围小
                'kill_probability': 0.6, # 击杀概率低
                'observation_range': 8.0 # 观察范围大
            },
            'fighter': {
                'max_velocity': 5.0,    # 战斗机速度适中
                'attack_range': 4.0,    # 攻击范围适中
                'kill_probability': 0.8, # 击杀概率高
                'observation_range': 6.0 # 观察范围适中
            },
            'bomber': {
                'max_velocity': 4.0,    # 轰炸机速度慢
                'attack_range': 5.0,    # 攻击范围大
                'kill_probability': 0.9, # 击杀概率很高
                'observation_range': 5.0 # 观察范围小
            }
        }

        # 初始化智能体类型
        if heterogeneous:
            self.red_agent_types = red_agent_types or ['fighter'] * num_red
            self.blue_agent_types = blue_agent_types or ['fighter'] * num_blue
        else:
            self.red_agent_types = ['fighter'] * num_red
            self.blue_agent_types = ['fighter'] * num_blue

        # 动作空间定义
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # 观察空间定义
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim * self.total_agents,),
            dtype=np.float32
        )

        # 环境参数
        self.boundary = 12.0
        self.kill_distance = 2.0
        self.safe_distance = 5.0
        
        # 奖励系统参数
        self.reward_weights = {
            'kill': 10.0,           # 击杀奖励
            'death': -8.0,          # 死亡惩罚
            'victory': 20.0,        # 胜利奖励
            'survive': 0.05,        # 生存奖励
            'approach': 0.1,        # 接近奖励
            'team': 0.15,           # 团队协作奖励
            'boundary': -0.3,       # 边界惩罚
            'energy': -0.01,        # 能量消耗惩罚
            'strategy': 0.2         # 战术位置奖励
        }

    def reset(self) -> np.ndarray:
        """
        重置环境状态
        :return: 初始观察值
        """
        self.steps_count = 0
        
        # 初始化位置 - 使用更合理的初始化策略
        self.red_positions = self._initialize_formation(
            num_agents=self.num_red,
            center=np.array([-5.0, 0.0]),
            spread=3.0
        )
        self.blue_positions = self._initialize_formation(
            num_agents=self.num_blue,
            center=np.array([5.0, 0.0]),
            spread=3.0
        )
        
        # 初始化速度和存活状态
        self.red_velocities = np.zeros((self.num_red, 2), dtype=np.float32)
        self.blue_velocities = np.zeros((self.num_blue, 2), dtype=np.float32)
        self.red_alive = np.ones(self.num_red, dtype=bool)
        self.blue_alive = np.ones(self.num_blue, dtype=bool)
        
        # 初始化能量值(新增)
        self.red_energy = np.ones(self.num_red, dtype=np.float32) * 100
        self.blue_energy = np.ones(self.num_blue, dtype=np.float32) * 100
        # 重置速度历史
        self.last_red_velocities = np.zeros((self.num_red, 2), dtype=np.float32)
        self.last_blue_velocities = np.zeros((self.num_blue, 2), dtype=np.float32)
        
        return self._get_obs()

    def _initialize_formation(self, num_agents: int, center: np.ndarray, spread: float) -> np.ndarray:
        """
        初始化智能体的阵型
        :param num_agents: 智能体数量
        :param center: 阵型中心位置
        :param spread: 分散程度
        :return: 智能体位置数组
        """
        if num_agents == 1:
            return np.array([center], dtype=np.float32)
            
        positions = []
        if num_agents <= 3:
            # 三角形阵型
            angles = np.linspace(0, 2*np.pi, num_agents, endpoint=False)
            for angle in angles:
                pos = center + spread * np.array([np.cos(angle), np.sin(angle)])
                positions.append(pos)
        else:
            # 矩形阵型
            rows = int(np.ceil(np.sqrt(num_agents)))
            for i in range(num_agents):
                row = i // rows
                col = i % rows
                pos = center + spread * np.array([row - rows/2, col - rows/2])
                positions.append(pos)
                
        return np.array(positions, dtype=np.float32)

    def step(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, bool, Dict]:
        """
        环境步进
        :param actions: 智能体动作列表
        :return: (观察值, 奖励值, 是否结束, 信息字典)
        """
        self.steps_count += 1
        
        # 动作处理
        actions = np.array(actions, dtype=np.float32)
        red_actions = actions[:self.num_red]
        blue_actions = actions[self.num_red:]
        
        # 更新状态
        self._update_velocities(red_actions, blue_actions)
        self._update_positions()
        self._update_energy()
        
        # 计算奖励和游戏状态
        rewards = self._compute_rewards()
        done = self._check_done()
        
        info = {
            'red_alive': np.sum(self.red_alive),
            'blue_alive': np.sum(self.blue_alive),
            'steps': self.steps_count,
            'red_positions': self.red_positions.copy(),
            'blue_positions': self.blue_positions.copy(),
            'red_energy': self.red_energy.copy(),
            'blue_energy': self.blue_energy.copy()
        }
        
        return self._get_obs(), rewards, done, info

    def _update_velocities(self, red_actions: np.ndarray, blue_actions: np.ndarray):
        """
        更新速度
        :param red_actions: 红方动作
        :param blue_actions: 蓝方动作
        """
        # 根据智能体类型获取对应的最大速度
        red_max_velocities = np.array([self.agent_types[agent_type]['max_velocity'] 
                                     for agent_type in self.red_agent_types])
        blue_max_velocities = np.array([self.agent_types[agent_type]['max_velocity'] 
                                      for agent_type in self.blue_agent_types])

        # 更新红方速度
        red_actions = red_actions.reshape(self.num_red, 2)
        self.red_velocities += red_actions * self.red_alive.reshape(-1, 1)
        for i in range(self.num_red):
            if self.red_alive[i]:
                velocity_magnitude = np.linalg.norm(self.red_velocities[i])
                if velocity_magnitude > red_max_velocities[i]:
                    self.red_velocities[i] *= red_max_velocities[i] / velocity_magnitude

        # 更新蓝方速度
        blue_actions = blue_actions.reshape(self.num_blue, 2)
        self.blue_velocities += blue_actions * self.blue_alive.reshape(-1, 1)
        for i in range(self.num_blue):
            if self.blue_alive[i]:
                velocity_magnitude = np.linalg.norm(self.blue_velocities[i])
                if velocity_magnitude > blue_max_velocities[i]:
                    self.blue_velocities[i] *= blue_max_velocities[i] / velocity_magnitude

    def _update_positions(self):
        """更新位置并处理边界"""
        # 更新红方位置
        self.red_positions += self.red_velocities * self.red_alive.reshape(-1, 1)
        np.clip(self.red_positions, -self.boundary, self.boundary, out=self.red_positions)
        
        # 更新蓝方位置
        self.blue_positions += self.blue_velocities * self.blue_alive.reshape(-1, 1)
        np.clip(self.blue_positions, -self.boundary, self.boundary, out=self.blue_positions)

    def _update_energy(self):
        """更新能量消耗"""
        # 根据速度和机动计算能量消耗
        red_energy_cost = np.sum(np.square(self.red_velocities), axis=1) * 0.01
        blue_energy_cost = np.sum(np.square(self.blue_velocities), axis=1) * 0.01
        
        self.red_energy -= red_energy_cost * self.red_alive
        self.blue_energy -= blue_energy_cost * self.blue_alive
        
        # 能量耗尽则死亡
        self.red_alive &= (self.red_energy > 0)
        self.blue_alive &= (self.blue_energy > 0)

    def _compute_rewards(self) -> np.ndarray:
        """
        计算奖励
        :return: 奖励数组
        """
        rewards = np.zeros(self.total_agents, dtype=np.float32)
        
        # 计算红方中心位置
        red_center = np.mean(self.red_positions[self.red_alive], axis=0) if np.any(self.red_alive) else np.zeros(2)
        
        # 计算蓝方中心位置
        blue_center = np.mean(self.blue_positions[self.blue_alive], axis=0) if np.any(self.blue_alive) else np.zeros(2)
        
        # 红方奖励计算
        for i in range(self.num_red):
            if not self.red_alive[i]:
                continue
                
            agent_type = self.red_agent_types[i]
            attack_range = self.agent_types[agent_type]['attack_range']
            kill_prob = self.agent_types[agent_type]['kill_probability']
            
            # 基础生存奖励
            rewards[i] = self.reward_weights['survive']
            
            # 团队协作奖励：基于与队伍中心的距离
            dist_to_center = np.linalg.norm(self.red_positions[i] - red_center)
            rewards[i] += self.reward_weights['team'] * np.exp(-0.5 * dist_to_center)
            
            # 战术位置奖励：基于位置优势
            tactical_advantage = self._calculate_tactical_advantage(
                self.red_positions[i], 
                self.red_velocities[i],
                blue_center
            )
            rewards[i] += self.reward_weights['strategy'] * tactical_advantage
            
            # 对敌奖励计算
            for j in range(self.num_blue):
                if not self.blue_alive[j]:
                    continue
                
                dist = np.linalg.norm(self.red_positions[i] - self.blue_positions[j])
                
                # 在攻击范围内的奖励
                if dist < self.kill_distance:
                    # 击杀判定：根据智能体类型的击杀概率
                    if np.random.random() < kill_prob:
                        rewards[i] += self.reward_weights['kill']
                        rewards[self.num_red + j] += self.reward_weights['death']
                        self.blue_alive[j] = False
                elif dist < attack_range:
                    # 接近目标奖励：距离越近奖励越大
                    progress = (attack_range - dist) / attack_range
                    rewards[i] += self.reward_weights['approach'] * progress * progress
            
            # 边界惩罚(平滑过渡)
            pos = self.red_positions[i]
            dist_to_boundary = np.maximum(np.abs(pos) - (self.boundary - 2.0), 0)
            rewards[i] += self.reward_weights['boundary'] * np.sum(np.square(dist_to_boundary))
            
            # 能量消耗惩罚
            energy_cost = np.sum(np.square(self.red_velocities[i])) * self.reward_weights['energy']
            rewards[i] -= energy_cost
        
        # 蓝方奖励计算(类似红方，但有所简化)
        for i in range(self.num_blue):
            if not self.blue_alive[i]:
                continue
                
            agent_type = self.blue_agent_types[i]
            attack_range = self.agent_types[agent_type]['attack_range']
            kill_prob = self.agent_types[agent_type]['kill_probability']
            
            idx = self.num_red + i
            rewards[idx] = self.reward_weights['survive']
            
            # 对敌奖励
            for j in range(self.num_red):
                if not self.red_alive[j]:
                    continue
                    
                dist = np.linalg.norm(self.blue_positions[i] - self.red_positions[j])
                
                if dist < self.kill_distance:
                    if np.random.random() < kill_prob:
                        rewards[idx] += self.reward_weights['kill']
                        rewards[j] += self.reward_weights['death']
                        self.red_alive[j] = False
                elif dist < attack_range:
                    progress = (attack_range - dist) / attack_range
                    rewards[idx] += self.reward_weights['approach'] * progress * progress
            
            # 边界惩罚
            pos = self.blue_positions[i]
            dist_to_boundary = np.maximum(np.abs(pos) - (self.boundary - 2.0), 0)
            rewards[idx] += self.reward_weights['boundary'] * np.sum(np.square(dist_to_boundary))
            
            # 能量消耗
            energy_cost = np.sum(np.square(self.blue_velocities[i])) * self.reward_weights['energy']
            rewards[idx] -= energy_cost
        
        # 胜利奖励：基于存活比例
        if not np.any(self.blue_alive):
            alive_ratio = np.mean(self.red_alive)
            rewards[:self.num_red] += self.reward_weights['victory'] * alive_ratio
        elif not np.any(self.red_alive):
            alive_ratio = np.mean(self.blue_alive)
            rewards[self.num_red:] += self.reward_weights['victory'] * alive_ratio
        
        return rewards

    def _calculate_tactical_advantage(self, position: np.ndarray, velocity: np.ndarray, 
                                    target_center: np.ndarray) -> float:
        """计算战术位置优势"""
        to_target = target_center - position
        dist_to_target = np.linalg.norm(to_target)
        if dist_to_target < 1e-6:
            return 0.0
            
        target_direction = to_target / dist_to_target
        
        # 速度方向
        speed = np.linalg.norm(velocity)
        velocity_direction = velocity / max(speed, 1e-6)
        
        # 计算攻击角度
        attack_angle = np.arccos(np.clip(np.dot(target_direction, velocity_direction), -1.0, 1.0))
        optimal_attack_angle = np.pi/6  # 30度为最佳攻击角
        angle_advantage = np.exp(-2.0 * (attack_angle - optimal_attack_angle)**2)
        
        # 综合考虑距离、角度和攻击角的优势
        distance_factor = np.exp(-0.1 * dist_to_target)
        angle_factor = (1 + np.dot(target_direction, velocity_direction)) / 2
        
        return distance_factor * angle_factor * angle_advantage

    def _get_obs(self) -> np.ndarray:
        """
        获取环境观察值
        返回：9维观察值数组 = 位置(2) + 速度(2) + 相对速度(2) + 加速度(2) + 状态(1)
        """
        obs = np.zeros(self.total_agents * self.state_dim, dtype=np.float32)
        
        # 计算相对速度
        red_relative_velocities = np.zeros((self.num_red, 2), dtype=np.float32)
        blue_relative_velocities = np.zeros((self.num_blue, 2), dtype=np.float32)
        
        if np.any(self.red_alive) and np.any(self.blue_alive):
            # 计算红方相对速度
            for i in range(self.num_red):
                if self.red_alive[i]:
                    distances = [np.linalg.norm(self.red_positions[i] - self.blue_positions[j]) 
                            for j in range(self.num_blue) if self.blue_alive[j]]
                    if distances:
                        nearest_enemy = np.argmin(distances)
                        red_relative_velocities[i] = (self.red_velocities[i] - 
                                                    self.blue_velocities[nearest_enemy])

            # 计算蓝方相对速度
            for i in range(self.num_blue):
                if self.blue_alive[i]:
                    distances = [np.linalg.norm(self.blue_positions[i] - self.red_positions[j]) 
                            for j in range(self.num_red) if self.red_alive[j]]
                    if distances:
                        nearest_enemy = np.argmin(distances)
                        blue_relative_velocities[i] = (self.blue_velocities[i] - 
                                                    self.red_velocities[nearest_enemy])
        
        # 计算加速度
        red_acceleration = np.zeros((self.num_red, 2), dtype=np.float32)
        blue_acceleration = np.zeros((self.num_blue, 2), dtype=np.float32)
        
        if self.last_red_velocities is not None:
            red_acceleration = self.red_velocities - self.last_red_velocities
        if self.last_blue_velocities is not None:
            blue_acceleration = self.blue_velocities - self.last_blue_velocities
        
        # 填充红方观察值
        for i in range(self.num_red):
            start_idx = i * self.state_dim
            
            # 位置信息 (2维)
            obs[start_idx:start_idx+2] = self.red_positions[i]
            
            # 速度信息 (2维)
            obs[start_idx+2:start_idx+4] = self.red_velocities[i]
            
            # 相对速度信息 (2维)
            obs[start_idx+4:start_idx+6] = red_relative_velocities[i]
            
            # 加速度信息 (2维)
            obs[start_idx+6:start_idx+8] = red_acceleration[i]
            
            # 存活状态 (1维)
            obs[start_idx+8] = float(self.red_alive[i])
        
        # 填充蓝方观察值
        for i in range(self.num_blue):
            start_idx = (i + self.num_red) * self.state_dim
            
            # 位置信息 (2维)
            obs[start_idx:start_idx+2] = self.blue_positions[i]
            
            # 速度信息 (2维)
            obs[start_idx+2:start_idx+4] = self.blue_velocities[i]
            
            # 相对速度信息 (2维)
            obs[start_idx+4:start_idx+6] = blue_relative_velocities[i]
            
            # 加速度信息 (2维)
            obs[start_idx+6:start_idx+8] = blue_acceleration[i]
            
            # 存活状态 (1维)
            obs[start_idx+8] = float(self.blue_alive[i])
        
        # 更新速度历史
        self.last_red_velocities = self.red_velocities.copy()
        self.last_blue_velocities = self.blue_velocities.copy()
        
        return obs.astype(np.float32)

    def _check_done(self) -> bool:
        """
        检查游戏是否结束
        :return: 是否结束
        """
        # 检查步数限制
        if self.steps_count >= self.max_steps:
            return True
        
        # 检查胜负
        red_all_dead = np.all(self.red_alive == False)
        blue_all_dead = np.all(self.blue_alive == False)
        
        # 任意一方全部阵亡则游戏结束
        return red_all_dead or blue_all_dead

    def render(self, mode='human'):
        """
        渲染环境（可选实现）
        """
        pass

    def seed(self, seed=None):
        """
        设置随机种子
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]