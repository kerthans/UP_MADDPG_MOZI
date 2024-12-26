# tests/test_suite.py

import unittest
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.combat_env import CombatEnv
from agents.maddpg import MADDPG
from agents.actor import Actor
from agents.critic import Critic

class TestEnvironment(unittest.TestCase):
    """测试环境的基本功能"""
    
    def setUp(self):
        self.env = CombatEnv(
            num_red=2, 
            num_blue=2, 
            state_dim=5, 
            action_dim=2,
            max_steps=200
        )
        
    def test_env_initialization(self):
        """测试环境初始化"""
        print("\n测试环境初始化...")
        # 测试观察空间
        obs = self.env.reset()
        self.assertEqual(len(obs), self.env.total_agents * self.env.state_dim)
        print(f"√ 观察空间维度正确: {len(obs)}")
        
        # 检查观察值类型
        self.assertEqual(obs.dtype, np.float32)
        print("√ 观察值类型正确: float32")
        
        # 测试动作空间
        self.assertEqual(self.env.action_space.shape[0], self.env.action_dim)
        print(f"√ 动作空间维度正确: {self.env.action_dim}")
        
    def test_env_step(self):
        """测试环境步进"""
        print("\n测试环境步进...")
        obs = self.env.reset()
        
        # 创建合适类型的动作
        actions = [np.array([0.5, 0.5], dtype=np.float32) for _ in range(self.env.total_agents)]
        
        # 执行步进
        next_obs, rewards, done, _ = self.env.step(actions)
        
        # 检查返回值
        self.assertEqual(len(next_obs), self.env.total_agents * self.env.state_dim)
        self.assertEqual(next_obs.dtype, np.float32)
        self.assertEqual(len(rewards), self.env.total_agents)
        self.assertIsInstance(done, bool)
        
        print(f"√ 环境步进返回值格式正确")
        print(f"√ 观察值类型正确: {next_obs.dtype}")
        print(f"√ 奖励维度正确: {len(rewards)}")
        
    def test_reward_mechanism(self):
        """测试奖励机制"""
        print("\n测试奖励机制...")
        self.env.reset()
        
        # 设置测试场景（确保使用float32类型）
        self.env.red_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        self.env.blue_positions = np.array([[0.5, 0.0]], dtype=np.float32)
        self.env.red_velocities = np.zeros((1, 2), dtype=np.float32)
        self.env.blue_velocities = np.zeros((1, 2), dtype=np.float32)
        
        # 创建测试动作
        actions = [np.array([0.1, 0.0], dtype=np.float32) for _ in range(self.env.total_agents)]
        
        # 执行步进
        _, rewards, _, _ = self.env.step(actions)
        
        # 验证奖励
        self.assertTrue(any(r > 0 for r in rewards[:self.env.num_red]))
        print(f"√ 奖励机制工作正常")
        print(f"  红方奖励: {rewards[:self.env.num_red]}")
        print(f"  蓝方奖励: {rewards[self.env.num_red:]}")

class TestNetworks(unittest.TestCase):
    """测试神经网络结构"""
    
    def setUp(self):
        self.state_dim = 5
        self.action_dim = 2
        self.batch_size = 32
        self.hidden_dim = 256
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = Critic(self.state_dim * 4, self.action_dim * 4, self.hidden_dim)
        
    def test_actor_network(self):
        """测试Actor网络"""
        print("\n测试Actor网络...")
        
        # 创建测试输入
        state = torch.randn(self.batch_size, self.state_dim)
        
        # 测试前向传播
        action = self.actor(state)
        
        # 检查输出
        self.assertEqual(action.shape, (self.batch_size, self.action_dim))
        self.assertTrue(torch.all(action >= -1) and torch.all(action <= 1))
        print(f"√ Actor输出维度正确: {action.shape}")
        print(f"√ Actor输出范围正确: [{action.min().item():.2f}, {action.max().item():.2f}]")
        
    def test_critic_network(self):
        """测试Critic网络"""
        print("\n测试Critic网络...")
        
        # 创建测试输入
        state = torch.randn(self.batch_size, self.state_dim * 4)
        action = torch.randn(self.batch_size, self.action_dim * 4)
        
        # 测试前向传播
        q_value = self.critic(state, action)
        
        # 检查输出
        self.assertEqual(q_value.shape, (self.batch_size, 1))
        print(f"√ Critic输出维度正确: {q_value.shape}")
        print(f"√ Q值范围: [{q_value.min().item():.2f}, {q_value.max().item():.2f}]")

class TestMADDPG(unittest.TestCase):
    """测试MADDPG算法"""
    
    def setUp(self):
        self.num_agents = 4
        self.state_dim = 5
        self.action_dim = 2
        self.batch_size = 32
        self.hidden_dim = 256
        self.maddpg = MADDPG(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            lr=1e-3,
            gamma=0.95,
            tau=0.01,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_dim
        )
        
    def test_action_selection(self):
        """测试动作选择"""
        print("\n测试动作选择...")
        
        # 创建测试状态
        states = [np.random.randn(self.state_dim).astype(np.float32) 
                 for _ in range(self.num_agents)]
        
        # 测试不同噪声水平
        noise_levels = [0.0, 0.1, 0.5]
        for noise in noise_levels:
            actions = self.maddpg.select_actions(states, noise=noise)
            
            # 检查动作
            self.assertEqual(len(actions), self.num_agents)
            self.assertTrue(all(isinstance(a, np.ndarray) for a in actions))
            self.assertTrue(all(a.shape == (self.action_dim,) for a in actions))
            self.assertTrue(all(a.dtype == np.float32 for a in actions))
            
            print(f"√ 噪声水平 {noise}: 动作范围 [{min(min(a) for a in actions):.2f}, "
                  f"{max(max(a) for a in actions):.2f}]")
    
    def test_training_step(self):
        """测试训练步骤"""
        print("\n测试训练步骤...")
        
        # 填充经验回放缓冲区
        for _ in range(self.batch_size + 10):
            states = [np.random.randn(self.state_dim).astype(np.float32) 
                     for _ in range(self.num_agents)]
            actions = [np.random.randn(self.action_dim).astype(np.float32) 
                      for _ in range(self.num_agents)]
            rewards = np.random.randn(self.num_agents).astype(np.float32)
            next_states = [np.random.randn(self.state_dim).astype(np.float32) 
                          for _ in range(self.num_agents)]
            
            self.maddpg.replay_buffer.add((states, actions, rewards, next_states, False))
        
        # 记录初始参数
        initial_params = sum(p.sum().item() for p in self.maddpg.actors[0].parameters())
        
        # 执行更新
        self.maddpg.update()
        
        # 记录更新后的参数
        final_params = sum(p.sum().item() for p in self.maddpg.actors[0].parameters())
        
        # 验证参数已更新
        self.assertNotEqual(initial_params, final_params)
        print(f"√ 网络参数已更新")
        print(f"  参数变化: {abs(final_params - initial_params):.6f}")

class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def setUp(self):
        self.env = CombatEnv(
            num_red=2,
            num_blue=2,
            state_dim=5,
            action_dim=2,
            max_steps=100
        )
        self.maddpg = MADDPG(
            num_agents=4,
            state_dim=5,
            action_dim=2,
            lr=1e-3,
            gamma=0.95,
            tau=0.01,
            batch_size=32,
            hidden_dim=256
        )
        
    def test_training_episode(self):
        """测试完整的训练回合"""
        print("\n测试训练回合...")
        
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        
        while steps < self.env.max_steps:
            # 获取每个智能体的观察
            per_agent_obs = []
            for i in range(self.maddpg.num_agents):
                agent_obs = obs[i * self.env.state_dim:(i + 1) * self.env.state_dim]
                per_agent_obs.append(agent_obs)
            
            # 选择动作
            actions = self.maddpg.select_actions(per_agent_obs, noise=0.1)
            
            # 执行动作
            next_obs, rewards, done, _ = self.env.step(actions)
            
            # 存储经验
            self.maddpg.replay_buffer.add((per_agent_obs, actions, rewards, next_obs, done))
            
            # 如果有足够的样本，进行更新
            if self.maddpg.replay_buffer.size() > self.maddpg.batch_size:
                self.maddpg.update()
            
            total_reward += sum(rewards)
            obs = next_obs
            steps += 1
            
            if done:
                break
        
        print(f"√ 成功完成训练回合")
        print(f"  总步数: {steps}")
        print(f"  总奖励: {total_reward:.2f}")
        print(f"  回合是否提前结束: {done}")
        print(f"  经验回放缓冲区大小: {self.maddpg.replay_buffer.size()}")

def run_tests():
    """运行所有测试"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnvironment)
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNetworks))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMADDPG))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegration))
    
    # 运行测试
    print("\n" + "="*50)
    print("开始运行MADDPG测试套件")
    print("="*50)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_tests()