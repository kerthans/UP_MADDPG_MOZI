# agents/utils.py

import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition):
        """
        添加一个转换到缓冲区
        transition: tuple (states, actions, rewards, next_states, done)
        states: list of agent states
        actions: list of agent actions
        rewards: numpy array of rewards
        next_states: list of next states
        done: bool
        """
        # 确保数据类型正确
        states = [np.array(state, dtype=np.float32) for state in transition[0]]
        actions = [np.array(action, dtype=np.float32) for action in transition[1]]
        rewards = np.array(transition[2], dtype=np.float32)
        next_states = [np.array(state, dtype=np.float32) for state in transition[3]]
        done = bool(transition[4])
        
        self.buffer.append((states, actions, rewards, next_states, done))
    
    def sample(self, batch_size):
        """采样一个批次的数据"""
        batch = random.sample(self.buffer, batch_size)
        
        # 分解批次数据
        states_batch = []
        actions_batch = []
        rewards_batch = []
        next_states_batch = []
        dones_batch = []
        
        for states, actions, rewards, next_states, done in batch:
            states_batch.append(states)
            actions_batch.append(actions)
            rewards_batch.append(rewards)
            next_states_batch.append(next_states)
            dones_batch.append(done)
            
        # 转换为正确的格式
        # [batch_size, num_agents, state_dim]
        states = np.array(states_batch)
        # [batch_size, num_agents, action_dim]
        actions = np.array(actions_batch)
        # [batch_size, num_agents]
        rewards = np.array(rewards_batch)
        # [batch_size, num_agents, state_dim]
        next_states = np.array(next_states_batch)
        # [batch_size]
        dones = np.array(dones_batch)
        
        return states, actions, rewards, next_states, dones
    
    def size(self):
        return len(self.buffer)