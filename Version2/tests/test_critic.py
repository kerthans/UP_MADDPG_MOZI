# tests/test_critic.py

import unittest
import torch
import sys
import os

# 动态添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agents.critic import Critic

class TestCritic(unittest.TestCase):
    def test_forward(self):
        num_agents = 2
        state_dim = 5
        action_dim = 2
        total_state_dim = state_dim * num_agents  # 10
        total_action_dim = action_dim * num_agents  # 4
        critic = Critic(total_state_dim, total_action_dim)
        state = torch.randn((1, total_state_dim), dtype=torch.float32)
        action = torch.randn((1, total_action_dim), dtype=torch.float32)
        q = critic(state, action)
        self.assertEqual(q.shape, (1, 1), f"Expected Q-value shape (1, 1), got {q.shape}")

if __name__ == '__main__':
    unittest.main()
