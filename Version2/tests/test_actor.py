# tests/test_actor.py

import unittest
import torch
import sys
import os

# 动态添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agents.actor import Actor

class TestActor(unittest.TestCase):
    def test_forward(self):
        state_dim = 5
        action_dim = 2
        actor = Actor(state_dim, action_dim)
        state = torch.randn((1, state_dim), dtype=torch.float32)
        action = actor(state)
        self.assertEqual(action.shape, (1, action_dim), f"Expected action shape (1, {action_dim}), got {action.shape}")

if __name__ == '__main__':
    unittest.main()
