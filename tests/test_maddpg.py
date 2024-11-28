# tests/test_maddpg.py

import unittest
import numpy as np
import sys
import os

# 动态添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agents.maddpg import MADDPG

class TestMADDPG(unittest.TestCase):
    def test_initialization(self):
        num_agents = 2
        state_dim = 5  # 每个智能体的状态维度
        action_dim = 2
        maddpg = MADDPG(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim)
        self.assertEqual(len(maddpg.actors), num_agents, f"Expected {num_agents} actors, got {len(maddpg.actors)}")
        self.assertEqual(len(maddpg.critics), num_agents, f"Expected {num_agents} critics, got {len(maddpg.critics)}")
        
    def test_select_actions(self):
        num_agents = 2
        state_dim = 5
        action_dim = 2
        maddpg = MADDPG(num_agents=num_agents, state_dim=state_dim, action_dim=action_dim)
        states = [np.random.randn(state_dim).astype(np.float32) for _ in range(num_agents)]
        actions = maddpg.select_actions(states, noise=0.0)  # 设置噪声为0，便于测试
        self.assertEqual(len(actions), num_agents, f"Expected {num_agents} actions, got {len(actions)}")
        for action in actions:
            self.assertEqual(action.shape, (action_dim,), f"Expected action shape ({action_dim},), got {action.shape}")

if __name__ == '__main__':
    unittest.main()
