# env/combat_env.py

import numpy as np
import gym
from gym import spaces

class CombatEnv(gym.Env):
    def __init__(self, num_red=3, num_blue=3, state_dim=5, action_dim=2, max_velocity=5.0, max_steps=300, debug=False):
        super(CombatEnv, self).__init__()
        self.num_red = num_red
        self.num_blue = num_blue
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.total_agents = self.num_red + self.num_blue
        self.max_velocity = max_velocity
        self.max_steps = max_steps
        self.debug = debug
        
        # 环境参数
        self.boundary = 10.0
        self.kill_distance = 1.0      # 击落距离
        self.attack_range = 2.0       # 攻击区域范围
        self.safe_distance = 3.0      # 安全距离
        
        # 奖励参数
        self.kill_reward = 20.0       # 击落奖励
        self.death_penalty = -15.0    # 被击落惩罚
        self.victory_reward = 50.0    # 胜利奖励
        self.survive_reward = 0.1     # 生存奖励
        self.boundary_penalty = -1.0  # 边界惩罚
        
        # 空间定义
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.total_agents * self.state_dim,),
            dtype=np.float32
        )
        
        self.steps_count = 0

    def reset(self):
        self.steps_count = 0
        
        # 初始化位置：红蓝双方在各自区域随机分布
        self.red_positions = np.random.uniform(-8, -3, (self.num_red, 2)).astype(np.float32)
        self.blue_positions = np.random.uniform(3, 8, (self.num_blue, 2)).astype(np.float32)
        
        # 初始化速度和存活状态
        self.red_velocities = np.zeros((self.num_red, 2), dtype=np.float32)
        self.blue_velocities = np.zeros((self.num_blue, 2), dtype=np.float32)
        self.red_alive = np.ones(self.num_red, dtype=np.float32)
        self.blue_alive = np.ones(self.num_blue, dtype=np.float32)
        
        if self.debug:
            print("环境已重置")
        return self._get_obs()

    def step(self, actions):
        self.steps_count += 1
        info = {}
        
        # 动作执行
        actions = np.array(actions, dtype=np.float32)
        red_actions = actions[:self.num_red]
        blue_actions = actions[self.num_red:]

        # 更新速度和位置
        self._update_velocities(red_actions, blue_actions)
        self._update_positions()
        
        # 计算奖励和状态
        rewards = self._compute_rewards()
        done = self._check_done()
        
        # 记录信息
        info['red_alive'] = np.sum(self.red_alive)
        info['blue_alive'] = np.sum(self.blue_alive)
        info['steps'] = self.steps_count
        
        return self._get_obs(), rewards, done, info

    def _update_velocities(self, red_actions, blue_actions):
        """更新速度"""
        self.red_velocities = (self.red_velocities + red_actions[:, :2] * self.red_alive[:, None]).astype(np.float32)
        self.blue_velocities = (self.blue_velocities + blue_actions[:, :2] * self.blue_alive[:, None]).astype(np.float32)
        
        # 速度限制
        self.red_velocities = np.clip(self.red_velocities, -self.max_velocity, self.max_velocity)
        self.blue_velocities = np.clip(self.blue_velocities, -self.max_velocity, self.max_velocity)

    def _update_positions(self):
        """更新位置"""
        self.red_positions = (self.red_positions + self.red_velocities * self.red_alive[:, None]).astype(np.float32)
        self.blue_positions = (self.blue_positions + self.blue_velocities * self.blue_alive[:, None]).astype(np.float32)
        
        # 边界检查
        self.red_positions = np.clip(self.red_positions, -self.boundary, self.boundary)
        self.blue_positions = np.clip(self.blue_positions, -self.boundary, self.boundary)

    def _compute_rewards(self):
        """计算奖励"""
        rewards = np.zeros(self.total_agents, dtype=np.float32)
        
        # 计算红蓝双方交互奖励
        for i in range(self.num_red):
            if self.red_alive[i] == 0:
                continue
                
            # 生存奖励
            rewards[i] += self.survive_reward
            
            for j in range(self.num_blue):
                if self.blue_alive[j] == 0:
                    continue
                    
                distance = np.linalg.norm(self.red_positions[i] - self.blue_positions[j])
                
                # 击落奖励
                if distance < self.kill_distance:
                    rewards[i] += self.kill_reward
                    rewards[self.num_red + j] += self.death_penalty
                    self.blue_alive[j] = 0
                    if self.debug:
                        print(f"红方{i}击落蓝方{j}，距离{distance:.2f}")
                
                # 距离奖励（攻击区域）
                elif distance < self.attack_range:
                    rewards[i] += 2.0 * (self.attack_range - distance)
                    rewards[self.num_red + j] -= (self.attack_range - distance)
                
                # 追击/逃离奖励
                else:
                    pursuit_reward = 1.0 / (1.0 + distance)
                    rewards[i] += pursuit_reward
                    rewards[self.num_red + j] -= pursuit_reward * 0.5
        
        # 边界惩罚
        red_boundary_dist = np.maximum(np.abs(self.red_positions) - (self.boundary - 1.0), 0)
        blue_boundary_dist = np.maximum(np.abs(self.blue_positions) - (self.boundary - 1.0), 0)
        
        rewards[:self.num_red] += self.boundary_penalty * np.sum(red_boundary_dist, axis=1)
        rewards[self.num_red:] += self.boundary_penalty * np.sum(blue_boundary_dist, axis=1)
        
        # 胜利奖励
        if np.all(self.blue_alive == 0):
            rewards[:self.num_red] += self.victory_reward
            
        return rewards

    def _check_done(self):
        """检查是否结束"""
        if self.steps_count >= self.max_steps:
            return True
        if np.all(self.blue_alive == 0):
            return True
        return False

    def _get_obs(self):
        """获取观察"""
        red_obs = np.hstack([
            self.red_positions,
            self.red_velocities,
            self.red_alive[:, None]
        ]).flatten()
        
        blue_obs = np.hstack([
            self.blue_positions,
            self.blue_velocities,
            self.blue_alive[:, None]
        ]).flatten()
        
        return np.concatenate([red_obs, blue_obs]).astype(np.float32)