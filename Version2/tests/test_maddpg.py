# tests/test_maddpg.py

import unittest
import numpy as np
import torch
import sys
import os
import shutil

# 动态添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from agents.maddpg import MADDPG, PrioritizedReplayBuffer, ICMNetwork

class TestMADDPG(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.num_agents = 2
        self.state_dim = 5
        self.action_dim = 2
        self.agent_types = ['scout', 'fighter']
        self.batch_size = 4
        self.hidden_dim = 64
        
        # 创建测试用MADDPG实例
        self.maddpg = MADDPG(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            agent_types=self.agent_types,
            buffer_size=100,
            batch_size=self.batch_size,
            hidden_dim=self.hidden_dim,
            n_step=3
        )
        
        # 创建测试目录
        self.test_dir = "test_models"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """清理测试环境"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """测试初始化配置"""
        # 基础配置测试
        self.assertEqual(len(self.maddpg.actors), self.num_agents)
        self.assertEqual(len(self.maddpg.critics), self.num_agents)
        self.assertEqual(self.maddpg.agent_types, self.agent_types)
        
        # 网络架构测试
        for i, actor in enumerate(self.maddpg.actors):
            self.assertTrue(hasattr(actor, 'attention'))
            self.assertEqual(actor.agent_type, self.agent_types[i])
            
        # 默认配置测试
        default_maddpg = MADDPG(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        self.assertEqual(default_maddpg.agent_types, ['default'] * self.num_agents)

    def test_priority_replay_buffer(self):
        """测试优先级经验回放"""
        buffer = PrioritizedReplayBuffer(capacity=100)
        
        # 基础功能测试
        for i in range(10):
            state = np.zeros(self.state_dim)
            action = np.zeros(self.action_dim)
            reward = float(i)
            next_state = np.zeros(self.state_dim)
            done = False
            buffer.add(state, action, reward, next_state, done)
            
        batch, indices, weights = buffer.sample(4)
        self.assertEqual(len(batch), 4)
        self.assertEqual(len(weights), 4)
        
        # 优先级更新测试
        new_priorities = np.array([10.0] * len(indices))
        buffer.update_priorities(indices, new_priorities)
        
        # 验证优先级效果
        _, _, new_weights = buffer.sample(4)
        self.assertTrue(np.any(new_weights != weights))

    def test_n_step_returns(self):
        """测试n步回报计算"""
        # 常规情况测试
        rewards = torch.ones(self.batch_size, 1)
        next_q_values = torch.ones(self.batch_size, 1) * 0.5
        dones = torch.zeros(self.batch_size, 1)
        
        returns = self.maddpg._compute_n_step_returns(
            rewards, next_q_values, dones, self.maddpg.gamma
        )
        
        self.assertEqual(returns.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(returns > rewards))

        # 终止状态测试
        dones = torch.ones(self.batch_size, 1)
        terminal_returns = self.maddpg._compute_n_step_returns(
            rewards, next_q_values, dones, self.maddpg.gamma
        )
        self.assertTrue(torch.allclose(terminal_returns, rewards))

    def test_curiosity_mechanism(self):
        """测试好奇心机制"""
        batch_size = self.batch_size
        
        # 测试ICM网络
        states = torch.randn(batch_size, self.num_agents * self.state_dim)
        actions = torch.randn(batch_size, self.num_agents * self.action_dim)
        next_states = torch.randn(batch_size, self.num_agents * self.state_dim)
        
        # 内在奖励测试
        intrinsic_rewards = self.maddpg._compute_curiosity_rewards(
            states, actions, next_states
        )
        
        self.assertEqual(intrinsic_rewards.shape, (batch_size, self.num_agents))
        self.assertTrue(torch.all(intrinsic_rewards >= 0))

        # ICM学习测试
        try:
            self.maddpg._update_icm(states, actions, next_states)
        except Exception as e:
            self.fail(f"ICM update failed with error: {str(e)}")

    def test_attention_mechanism(self):
        """测试注意力机制"""
        batch_size = 4
        seq_len = 3
        
        # 测试Actor的注意力机制
        for actor in self.maddpg.actors:
            x = torch.randn(batch_size, seq_len, self.hidden_dim)
            try:
                output = actor.attention(x)
                self.assertEqual(output.shape, x.shape)
            except Exception as e:
                self.fail(f"Attention mechanism failed with error: {str(e)}")

    def test_heterogeneous_agents(self):
        """测试异构智能体"""
        state = torch.randn(1, self.state_dim)
        
        # 测试不同类型智能体的动作生成
        scout_action = self.maddpg.actors[0](state)
        fighter_action = self.maddpg.actors[1](state)
        
        self.assertTrue(torch.all(torch.abs(scout_action) <= 1.0))
        self.assertTrue(torch.all(torch.abs(fighter_action) <= 1.0))
        
        # 验证网络参数差异性
        scout_params = list(self.maddpg.actors[0].parameters())
        fighter_params = list(self.maddpg.actors[1].parameters())
        
        params_different = False
        for p1, p2 in zip(scout_params, fighter_params):
            if not torch.allclose(p1, p2):
                params_different = True
                break
        self.assertTrue(params_different)

    def test_full_training_cycle(self):
        """测试完整训练周期"""
        # 准备训练数据
        for _ in range(self.batch_size * 2):
            states = []
            actions = []
            rewards = []
            next_states = []
            
            for _ in range(self.num_agents):
                states.append(np.random.randn(self.state_dim))
                actions.append(np.random.randn(self.action_dim))
                rewards.append(np.random.rand())
                next_states.append(np.random.randn(self.state_dim))
            
            states = np.concatenate(states).astype(np.float32)
            actions = np.concatenate(actions).astype(np.float32)
            rewards = np.array(rewards).astype(np.float32)
            next_states = np.concatenate(next_states).astype(np.float32)
            
            self.maddpg.replay_buffer.add(states, actions, rewards, next_states, False)
        
        # 测试更新过程
        try:
            self.maddpg.update()
        except Exception as e:
            self.fail(f"Training update failed with error: {str(e)}")

    def test_save_load(self):
        """测试模型保存和加载"""
        save_path = os.path.join(self.test_dir, "model.pt")
        
        # 保存模型
        self.maddpg.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # 加载模型到新实例
        new_maddpg = MADDPG(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            agent_types=self.agent_types
        )
        
        try:
            new_maddpg.load(save_path)
        except Exception as e:
            self.fail(f"Model loading failed with error: {str(e)}")
        
        # 额外的一致性检查
        self.assertEqual(len(self.maddpg.actors), len(new_maddpg.actors))
        self.assertEqual(len(self.maddpg.critics), len(new_maddpg.critics))
        
        # 验证加载后的行为一致性
        test_state = torch.randn(1, self.state_dim)
        original_action = self.maddpg.actors[0](test_state)
        loaded_action = new_maddpg.actors[0](test_state)
        
        # 使用更宽松的误差容忍度
        self.assertTrue(torch.allclose(original_action, loaded_action, atol=1e-5, rtol=1e-4))

def run_maddpg_tests():
    """运行所有MADDPG测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMADDPG)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    run_maddpg_tests()