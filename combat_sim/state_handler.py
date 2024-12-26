import math
import numpy as np
from collections import deque
from typing import Dict, List, Tuple

class StateHandler:
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.red_states = None
        self.blue_states = None
        self.state_buffer = deque(maxlen=5)
        self.thread_pool = None  # 添加线程池属性

    def set_thread_pool(self, thread_pool):
        """设置线程池"""
        self.thread_pool = thread_pool

    def reset(self):
        """重置环境状态"""
        self.current_step = 0
        self.state_buffer.clear()
        
        # 初始化红方和蓝方状态
        self.red_states = self._initialize_team(True)
        self.blue_states = self._initialize_team(False)
        
        self.state_buffer.append((self.red_states.copy(), self.blue_states.copy()))
        return self.get_obs()

    def calculate_obs_dim(self):
        # 实现观察空间维度计算
        own_state = 6
        trend_features = 5
        teammates_dim = max(self.config['num_red'] - 1, self.config['num_blue'] - 1) * 7
        opponents_dim = max(self.config['num_red'], self.config['num_blue']) * 8
        return own_state + trend_features + teammates_dim + opponents_dim


    def _initialize_team(self, is_red):
        """优化的队伍初始化函数，确保智能体严格在指定区域内
        
        Args:
            is_red: 是否为红方
                
        Returns:
            np.ndarray: 初始化的状态矩阵
        """
        num_agents = self.config['num_red'] if is_red else self.config['num_blue']
        states = np.zeros((num_agents, 5))
        field_size = self.config['field_size']
        
        # 计算安全的编队半径，确保不会超出区域限制
        max_radius = min(
            field_size/8,  # 确保x方向不超出
            field_size/4   # 确保y方向不超出
        )
        formation_radius = min(self.config['attack_range'] * 0.8, max_radius)
        
        # 确定基准位置：对红方在左1/8处，对蓝方在右7/8处
        base_x = field_size * (0.125 if is_red else 0.875)  # 从1/4改为1/8，给编队留出空间
        base_y = field_size * 0.5
        
        for i in range(num_agents):
            # 计算编队位置角度，根据队伍数量均匀分布
            formation_angle = (i * 2 * np.pi / num_agents)
            if is_red:
                formation_angle += np.pi/2  # 红方朝右
            else:
                formation_angle += -np.pi/2  # 蓝方朝左
                
            # 计算位置，确保在限定范围内
            x = base_x + formation_radius * np.cos(formation_angle)
            y = base_y + formation_radius * np.sin(formation_angle)
            
            # 确保x坐标严格在指定范围内
            if is_red:
                x = np.clip(x, 0, field_size/4)  # 红方在左四分之一
            else:
                x = np.clip(x, 3*field_size/4, field_size)  # 蓝方在右四分之一
                
            # 确保y坐标在场地范围内
            y = np.clip(y, 0, field_size)
            
            # 设置初始朝向：红方朝右(0)，蓝方朝左(pi)
            heading = 0 if is_red else np.pi
            
            # 设置初始状态
            states[i] = [
                x,
                y,
                heading,
                self.config['max_speed'] * 0.8,  # 初始速度设为最大速度的80%
                1.0  # 存活状态
            ]
        
        return states

    def update_states(self, actions, thread_pool):
        self.current_step += 1
        self.state_buffer.append((self.red_states.copy(), self.blue_states.copy()))
        
        # 使用线程池更新状态
        futures = []
        for i in range(self.config['num_red']):
            if self.red_states[i, 4] > 0.5:
                futures.append(
                    thread_pool.submit(
                        self._update_agent_state,
                        (i, True, actions[f'red_{i}'])
                    )
                )
        
        for i in range(self.config['num_blue']):
            if self.blue_states[i, 4] > 0.5:
                futures.append(
                    thread_pool.submit(
                        self._update_agent_state,
                        (i, False, actions[f'blue_{i}'])
                    )
                )
        
        for future in futures:
            future.result()

    def _update_agent_state(self, args):
        """优化的智能体状态更新
        
        Args:
            args: (idx, is_red, action) 元组
        """
        idx, is_red, action = args
        states = self.red_states if is_red else self.blue_states
        
        if states[idx, 4] < 0.5:
            return
            
        # 计算最优转向
        current_heading = states[idx, 2]
        current_pos = states[idx, :2]
        opponent_states = self.blue_states if is_red else self.red_states
        
        # 寻找最近的活着的对手
        nearest_dist = float('inf')
        nearest_angle = 0
        
        for opp_state in opponent_states:
            if opp_state[4] > 0.5:  # 只考虑活着的对手
                rel_pos = opp_state[:2] - current_pos
                dist = np.linalg.norm(rel_pos)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_angle = np.arctan2(rel_pos[1], rel_pos[0])
        
        # 优化转向控制
        angle_diff = (nearest_angle - current_heading + np.pi) % (2 * np.pi) - np.pi
        turn_rate = np.clip(
            angle_diff / self.config['max_turn_rate'],
            -1,
            1
        ) * self.config['max_turn_rate'] * action[0]
        
        # 优化速度控制
        target_speed = self.config['min_speed'] + (
            self.config['max_speed'] - self.config['min_speed']
        ) * (0.5 + 0.5 * action[1])
        
        # 更新状态
        states[idx, 2] = (current_heading + turn_rate) % (2 * np.pi)
        states[idx, 3] = np.clip(
            states[idx, 3] + (target_speed - states[idx, 3]) * 0.2,
            self.config['min_speed'],
            self.config['max_speed']
        )
        
        # 更新位置，增加智能避障
        new_pos = states[idx, :2] + states[idx, 3] * np.array([
            np.cos(states[idx, 2]),
            np.sin(states[idx, 2])
        ])
        
        # 碰撞避免
        for j, state in enumerate(states):
            if j != idx and state[4] > 0.5:
                rel_pos = new_pos - state[:2]
                dist = np.linalg.norm(rel_pos)
                if dist < self.config['attack_range'] * 0.5:
                    new_pos += rel_pos / dist * (
                        self.config['attack_range'] * 0.5 - dist
                    )
        
        # 边界处理
        states[idx, :2] = np.clip(new_pos, 0, self.config['field_size'])

    def get_obs(self, thread_pool=None):
        """获取观察，使用类的thread_pool或传入的thread_pool"""
        thread_pool = thread_pool or self.thread_pool
        if thread_pool is None:
            raise ValueError("Thread pool not set. Please set thread pool before calling get_obs")
            
        observations = {}
        
        # 并行计算观察
        red_obs_futures = [
            thread_pool.submit(self._get_agent_obs, i, True)
            for i in range(self.config['num_red'])
        ]
        
        blue_obs_futures = [
            thread_pool.submit(self._get_agent_obs, i, False)
            for i in range(self.config['num_blue'])
        ]
        
        # 收集结果
        for i, future in enumerate(red_obs_futures):
            observations[f'red_{i}'] = future.result()
        for i, future in enumerate(blue_obs_futures):
            observations[f'blue_{i}'] = future.result()
            
        return observations

    def _get_agent_obs(self, idx: int, is_red: bool) -> np.ndarray:
        """获取单个智能体的观察向量
        
        Args:
            idx: 智能体索引
            is_red: 是否为红方智能体
            
        Returns:
            np.ndarray: 标准化的观察向量
        """
        if is_red:
            own_state = self.red_states[idx]
            own_pos = own_state[:2]
            team_states = self.red_states
            opponent_states = self.blue_states
            max_teammates = max(self.config['num_red'] - 1, self.config['num_blue'] - 1)
            max_opponents = max(self.config['num_red'], self.config['num_blue'])
        else:
            own_state = self.blue_states[idx]
            own_pos = own_state[:2]
            team_states = self.blue_states
            opponent_states = self.red_states
            max_teammates = max(self.config['num_red'] - 1, self.config['num_blue'] - 1)
            max_opponents = max(self.config['num_red'], self.config['num_blue'])
        
        # 1. 标准化自身状态
        normalized_own_state = [
            own_state[0]/self.config['field_size'],
            own_state[1]/self.config['field_size'],
            np.cos(own_state[2]),
            np.sin(own_state[2]),
            own_state[3]/self.config['max_speed'],
            own_state[4]
        ]
        
        # 2. 计算趋势特征
        trend_features = self._calculate_trend_features(idx, is_red)
        
        # 3. 计算队友相对信息（补齐到最大维度）
        teammate_obs = []
        teammate_count = 0
        for i, state in enumerate(team_states):
            if i != idx and teammate_count < max_teammates:
                rel_pos = state[:2] - own_pos
                dist = np.linalg.norm(rel_pos)
                rel_speed = state[3] - own_state[3]
                rel_heading = state[2] - own_state[2]
                teammate_obs.extend([
                    rel_pos[0]/self.config['field_size'],
                    rel_pos[1]/self.config['field_size'],
                    rel_speed/self.config['max_speed'],
                    state[4],
                    np.cos(rel_heading),
                    np.sin(rel_heading),
                    dist/self.config['field_size']
                ])
                teammate_count += 1
        
        # 补齐未使用的队友维度
        padding_size = (max_teammates - teammate_count) * 7
        teammate_obs.extend([0.0] * padding_size)
        
        # 4. 计算对手相对信息（补齐到最大维度）
        opponent_obs = []
        opponent_count = 0
        for state in opponent_states:
            if opponent_count < max_opponents:
                rel_pos = state[:2] - own_pos
                dist = np.linalg.norm(rel_pos)
                angle = np.arctan2(rel_pos[1], rel_pos[0]) - own_state[2]
                in_range = float(dist <= self.config['attack_range'])
                threat_level = self._calculate_threat_level(own_state, state)
                
                opponent_obs.extend([
                    rel_pos[0]/self.config['field_size'],
                    rel_pos[1]/self.config['field_size'],
                    state[3]/self.config['max_speed'],
                    state[4],
                    angle/np.pi,
                    in_range,
                    threat_level,
                    dist/self.config['field_size']
                ])
                opponent_count += 1
                
        # 补齐未使用的对手维度
        padding_size = (max_opponents - opponent_count) * 8
        opponent_obs.extend([0.0] * padding_size)
        
        # 组合所有特征
        obs = np.concatenate([
            normalized_own_state,
            trend_features,
            np.array(teammate_obs),
            np.array(opponent_obs)
        ])
        
        return obs.astype(np.float32)

    def check_done(self):
        red_alive = np.sum(self.red_states[:, 4]) > 0
        blue_alive = np.sum(self.blue_states[:, 4]) > 0
        return bool(
            (self.current_step >= self.config['max_steps']) |
            ((~red_alive) | (~blue_alive))
        )
    
    def _calculate_trend_features(self, idx: int, is_red: bool) -> np.ndarray:
        """计算状态趋势特征
        
        Args:
            idx: 智能体索引
            is_red: 是否为红方智能体
            
        Returns:
            np.ndarray: 趋势特征向量[dx, dy, dspeed, cos(dheading), sin(dheading)]
        """
        if len(self.state_buffer) < 2:
            return np.zeros(5)
            
        current_states = self.state_buffer[-1][0 if is_red else 1]
        prev_states = self.state_buffer[-2][0 if is_red else 1]
        
        pos_diff = current_states[idx, :2] - prev_states[idx, :2]
        speed_diff = current_states[idx, 3] - prev_states[idx, 3]
        heading_diff = current_states[idx, 2] - prev_states[idx, 2]
        
        return np.array([
            pos_diff[0]/self.config['field_size'],
            pos_diff[1]/self.config['field_size'],
            speed_diff/self.config['max_speed'],
            np.cos(heading_diff),
            np.sin(heading_diff)
        ])
    
    def _calculate_threat_level(self, own_state, opponent_state):
        """计算威胁等级（避免除零）"""
        rel_pos = opponent_state[:2] - own_state[:2]
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-10:  # 避免除零
            return 1.0
            
        distance_threat = 1.0 - np.clip(dist / (2 * self.config['attack_range']), 0, 1)
        
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
        speed_threat = np.clip(closing_speed / (2 * self.config['max_speed']), -1, 1)
        
        return (distance_threat + max(0, speed_threat)) / 2.0
    def _validate_states(self):
        """验证状态的有效性"""
        if self.red_states is None or self.blue_states is None:
            raise ValueError("States not properly initialized")
        if self.red_states.shape[0] != self.config['num_red']:
            raise ValueError(f"Red states shape mismatch: {self.red_states.shape}")
        if self.blue_states.shape[0] != self.config['num_blue']:
            raise ValueError(f"Blue states shape mismatch: {self.blue_states.shape}")