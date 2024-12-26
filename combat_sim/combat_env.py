import random
import gym
import numpy as np
from gym import spaces
import math
from typing import Dict, List, Tuple
from collections import deque
from combat_sim.state_handler import StateHandler
from concurrent.futures import ThreadPoolExecutor
from combat_sim.combat_mechanics import CombatMechanics

class CombatEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, num_red=3, num_blue=2, max_steps=120, field_size=1000.0,
                attack_range=100.0, min_speed=10.0, max_speed=30.0,
                max_turn_rate=math.pi/6, hit_probability=0.8, num_threads=8):
        super().__init__()
        
        # 保存配置参数
        self.config = {
            'num_red': num_red,
            'num_blue': num_blue,
            'max_steps': max_steps,
            'field_size': field_size,
            'attack_range': attack_range,
            'min_speed': min_speed,
            'max_speed': max_speed,
            'max_turn_rate': max_turn_rate,
            'hit_probability': hit_probability,
            'num_threads': num_threads
        }
        
        # 初始化动作空间
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # 初始化线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=num_threads)
        
        # 初始化状态处理器和战斗机制
        self.state_handler = StateHandler(self.config)
        self.state_handler.set_thread_pool(self.thread_pool)
        self.combat_mechanics = CombatMechanics(self.config)
        
        # 初始化观察空间
        obs_dim = self.state_handler.calculate_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 直接初始化状态
        initial_obs = self.reset()
        self.red_states = self.state_handler.red_states
        self.blue_states = self.state_handler.blue_states
    @property
    def field_size(self):
        """获取场地大小"""
        return self.config['field_size']

    @property 
    def max_steps(self):
        """获取最大步数"""
        return self.config['max_steps']
    @property 
    def min_speed(self):
        return self.config['min_speed']
    @property 
    def max_speed(self):
        return self.config['max_speed']
    @property
    def num_red(self):
        return self.config['num_red']
        
    @property
    def num_blue(self):
        return self.config['num_blue']
        
    @property
    def reward_weights(self):
        return self.combat_mechanics.reward_weights
        
    def _calculate_obs_dim(self):
        return self.state_handler.calculate_obs_dim()
    def reset(self):
        return self.state_handler.reset()

    def step(self, actions):
        # 更新状态
        self.state_handler.update_states(actions, self.thread_pool)
        
        # 处理战斗
        red_hits, blue_hits = self.combat_mechanics.process_attacks(
            actions, 
            self.state_handler.red_states, 
            self.state_handler.blue_states
        )
        
        # 计算奖励
        rewards = self.combat_mechanics.compute_rewards(
            self.state_handler.red_states,
            self.state_handler.blue_states,
            red_hits,
            blue_hits
        )
        
        # 获取新观察
        next_obs = self.state_handler.get_obs(self.thread_pool)
        
        # 检查终止
        done = self.state_handler.check_done()
        
        # 更新统计
        if done:
            self.combat_mechanics.update_episode_stats(rewards, self.state_handler.current_step)
        
        info = {
            'red_hits': len(red_hits),
            'blue_hits': len(blue_hits),
            'red_alive': np.sum(self.state_handler.red_states[:, 4]),
            'blue_alive': np.sum(self.state_handler.blue_states[:, 4])
        }
        
        return next_obs, rewards, done, info

    def close(self):
        self.thread_pool.shutdown()
        
    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        return [seed]