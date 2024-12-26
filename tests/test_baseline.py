import unittest
import numpy as np
import torch
import os
import json
from collections import defaultdict
import time
from combat_sim.combat_env import CombatEnv
from agents.baseline import MADDPG, Actor, Critic, ReplayBuffer
import torch.nn as nn

class TestMADDPGBaseline(unittest.TestCase):
    """测试MADDPG baseline的完整功能和性能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境和基本参数"""
        print("\n开始MADDPG Baseline测试...")
        cls.env_config = {
            'num_red': 3,
            'num_blue': 2,
            'max_steps': 120,
            'field_size': 1000.0,
            'attack_range': 100.0,
            'min_speed': 10.0,
            'max_speed': 30.0,
            'max_turn_rate': np.pi/6,
            'hit_probability': 0.8,
            'num_threads': 8
        }
        cls.env = CombatEnv(**cls.env_config)
        
        # 获取观察空间维度
        obs = cls.env.reset()
        cls.obs_dim = len(list(obs.values())[0])
        cls.act_dim = 3  # [转向，速度，开火]
        
        # 创建保存目录
        cls.save_dir = './test_results'
        os.makedirs(cls.save_dir, exist_ok=True)

    def setUp(self):
        """每个测试用例开始前的设置"""
        self.maddpg = MADDPG(
            n_red=self.env_config['num_red'],
            n_blue=self.env_config['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
        )

    def test_01_model_initialization(self):
        """测试模型初始化"""
        print("\n测试模型初始化...")
        
        # 测试Actor网络
        actor = Actor(self.obs_dim, self.act_dim)
        self.assertIsInstance(actor, nn.Module)
        
        # 测试Critic网络
        critic = Critic(self.obs_dim, self.act_dim, self.env_config['num_red'] + self.env_config['num_blue'])
        self.assertIsInstance(critic, nn.Module)
        
        # 测试网络输出维度
        dummy_obs = torch.randn(1, self.obs_dim)
        actor_out = actor(dummy_obs)
        self.assertEqual(actor_out.shape, (1, self.act_dim))
        
        # 测试经验回放缓冲区
        buffer = ReplayBuffer()
        self.assertEqual(len(buffer), 0)
        
        print("模型初始化测试完成")

    def test_02_action_selection(self):
        """测试动作选择功能"""
        print("\n测试动作选择...")
        
        obs = self.env.reset()
        
        # 测试动作选择
        actions = self.maddpg.select_actions(obs)
        
        # 验证动作格式和范围
        for agent_id, action in actions.items():
            self.assertEqual(len(action), self.act_dim)
            self.assertTrue(np.all(action >= -1))
            self.assertTrue(np.all(action <= 1))
        
        # 验证动作噪声
        actions_with_noise = self.maddpg.select_actions(obs, noise_scale=0.1)
        actions_without_noise = self.maddpg.select_actions(obs, noise_scale=0.0)
        
        # 确保噪声产生了不同的动作
        self.assertFalse(np.all(actions_with_noise[list(actions_with_noise.keys())[0]] ==
                               actions_without_noise[list(actions_without_noise.keys())[0]]))
        
        print("动作选择测试完成")

    def test_03_experience_storage(self):
        """测试经验存储功能"""
        print("\n测试经验存储...")
        
        obs = self.env.reset()
        actions = self.maddpg.select_actions(obs)
        next_obs, rewards, done, _ = self.env.step(actions)
        
        # 测试经验存储
        self.maddpg.store_transition(
            obs,
            self._flatten_dict(actions),
            self._flatten_dict(rewards),
            next_obs,
            {k: float(done) for k in obs.keys()}
        )
        
        # 验证经验回放缓冲区
        self.assertGreater(len(self.maddpg.memory), 0)
        
        print("经验存储测试完成")

    def test_04_training_step(self):
        """测试训练步骤"""
        print("\n测试训练步骤...")
        
        # 填充一些经验
        for _ in range(self.maddpg.batch_size + 10):
            obs = self.env.reset()
            actions = self.maddpg.select_actions(obs)
            next_obs, rewards, done, _ = self.env.step(actions)
            self.maddpg.store_transition(
                obs,
                self._flatten_dict(actions),
                self._flatten_dict(rewards),
                next_obs,
                {k: float(done) for k in obs.keys()}
            )
        
        # 测试训练步骤
        self.maddpg.train()
        
        print("训练步骤测试完成")

    def test_05_short_training_session(self):
        """测试短期训练会话"""
        print("\n测试短期训练会话...")
        
        n_episodes = 5
        stats = self._run_training_session(n_episodes)
        
        # 验证基本训练效果
        self.assertEqual(len(stats['episode_rewards']), n_episodes)
        self.assertTrue(all(isinstance(r, (int, float)) for r in stats['episode_rewards']))
        
        print("短期训练会话测试完成")

    def test_06_model_save_load(self):
        """测试模型保存和加载"""
        print("\n测试模型保存和加载...")
        
        save_path = os.path.join(self.save_dir, 'test_model.pt')
        
        # 保存模型
        self.maddpg.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # 加载模型
        new_maddpg = MADDPG(
            n_red=self.env_config['num_red'],
            n_blue=self.env_config['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
        )
        new_maddpg.load(save_path)
        
        # 验证加载后的动作一致性
        obs = self.env.reset()
        actions1 = self.maddpg.select_actions(obs, noise_scale=0.0)
        actions2 = new_maddpg.select_actions(obs, noise_scale=0.0)
        
        for agent_id in actions1:
            np.testing.assert_array_almost_equal(actions1[agent_id], actions2[agent_id])
        
        print("模型保存和加载测试完成")

    def test_07_performance_metrics(self):
        """测试性能指标计算"""
        print("\n测试性能指标计算...")
        
        # 运行短期训练获取性能指标
        n_episodes = 10
        stats = self._run_training_session(n_episodes)
        
        # 验证基本指标
        self.assertTrue(len(stats['episode_rewards']) == n_episodes)
        self.assertTrue(len(stats['episode_lengths']) == n_episodes)
        self.assertTrue(len(stats['red_win_rates']) == n_episodes)
        self.assertTrue(len(stats['red_survival_rates']) == n_episodes)
        self.assertTrue(len(stats['blue_survival_rates']) == n_episodes)
        
        # 检查指标范围
        self.assertTrue(all(0 <= rate <= 1 for rate in stats['red_win_rates']))
        self.assertTrue(all(0 <= rate <= 1 for rate in stats['red_survival_rates']))
        self.assertTrue(all(0 <= rate <= 1 for rate in stats['blue_survival_rates']))
        
        print("性能指标计算测试完成")

    def _run_training_session(self, n_episodes):
        """运行训练会话并返回统计数据"""
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'red_win_rates': [],
            'red_survival_rates': [],
            'blue_survival_rates': []
        }
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            steps = 0
            
            while True:
                actions = self.maddpg.select_actions(obs)
                next_obs, rewards, done, info = self.env.step(actions)
                
                self.maddpg.store_transition(
                    obs,
                    self._flatten_dict(actions),
                    self._flatten_dict(rewards),
                    next_obs,
                    {k: float(done) for k in obs.keys()}
                )
                
                self.maddpg.train()
                
                episode_reward += sum(rewards.values())
                steps += 1
                obs = next_obs
                
                if done:
                    break
            
            # 更新统计数据
            stats['episode_rewards'].append(episode_reward)
            stats['episode_lengths'].append(steps)
            
            # 计算红方胜率
            red_win = all(info.get(f'blue_{i}', 0) < 0.5 for i in range(self.env_config['num_blue']))
            stats['red_win_rates'].append(float(red_win))
            
            # 计算存活率
            red_alive = sum(1 for i in range(self.env_config['num_red']) 
                          if info.get(f'red_{i}', 0) > 0.5)
            blue_alive = sum(1 for i in range(self.env_config['num_blue']) 
                           if info.get(f'blue_{i}', 0) > 0.5)
            
            stats['red_survival_rates'].append(red_alive / self.env_config['num_red'])
            stats['blue_survival_rates'].append(blue_alive / self.env_config['num_blue'])
        
        return stats

    def _flatten_dict(self, d):
        """将字典展平为列表"""
        result = []
        for i in range(self.env_config['num_red']):
            key = f'red_{i}'
            if key in d:
                result.append(d[key])
        for i in range(self.env_config['num_blue']):
            key = f'blue_{i}'
            if key in d:
                result.append(d[key])
        return result

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        cls.env.close()
        print("\nMADDPG Baseline测试完成!")

def main():
    # 运行所有测试
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()