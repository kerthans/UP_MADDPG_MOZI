# tests/test_main.py

import unittest
import numpy as np
import torch
import os
import json
from datetime import datetime
import pandas as pd
from main import plot_training_curves, save_evaluation_report
from env.combat_env import CombatEnv
from agents.maddpg import MADDPG

class TestMainFunction(unittest.TestCase):
    """测试主函数功能"""
    
    def setUp(self):
        """测试初始化"""
        self.test_params = {
            'num_episodes': 10,
            'max_steps': 300,
            'num_red': 3,
            'num_blue': 3,
            'state_dim': 5,
            'action_dim': 2,
            'lr': 1e-4,
            'gamma': 0.99,
            'tau': 0.001,
            'initial_noise': 0.5,
            'noise_decay': 0.9999,
            'min_noise': 0.05,
            'hidden_dim': 256,
            'batch_size': 128,
            'buffer_size': int(1e6),
            'max_velocity': 5.0,
            'seed': 42
        }
        
        # 设置随机种子
        np.random.seed(self.test_params['seed'])
        torch.manual_seed(self.test_params['seed'])
        
        # 创建测试目录
        self.test_dir = os.path.join("test_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 创建环境和智能体
        self.env = CombatEnv(**{k: v for k, v in self.test_params.items() 
                               if k in ['num_red', 'num_blue', 'state_dim', 'action_dim', 
                                      'max_velocity', 'max_steps']})
        
        self.maddpg = MADDPG(
            num_agents=self.test_params['num_red'] + self.test_params['num_blue'],
            state_dim=self.test_params['state_dim'],
            action_dim=self.test_params['action_dim'],
            lr=self.test_params['lr'],
            gamma=self.test_params['gamma'],
            tau=self.test_params['tau'],
            buffer_size=self.test_params['buffer_size'],
            batch_size=self.test_params['batch_size'],
            hidden_dim=self.test_params['hidden_dim']
        )

    def test_environment_setup(self):
        """测试环境设置"""
        obs = self.env.reset()
        self.assertEqual(len(obs), self.env.total_agents * self.env.state_dim)
        print("✓ 观察空间维度正确")
        self.assertEqual(self.maddpg.num_agents, self.env.total_agents)
        print("✓ 智能体数量正确")

    def test_training_step(self):
        """测试单个训练步骤"""
        obs = self.env.reset()
        per_agent_obs = []
        for i in range(self.maddpg.num_agents):
            agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
            per_agent_obs.append(agent_obs)
        
        actions = self.maddpg.select_actions(per_agent_obs, noise=0.1)
        next_obs, rewards, done, info = self.env.step(actions)
        
        self.assertEqual(len(actions), self.maddpg.num_agents)
        self.assertEqual(len(rewards), self.maddpg.num_agents)
        self.assertTrue(isinstance(done, bool))
        
        print("✓ 单步训练测试通过")
        print(f"- 奖励: {rewards}")
        print(f"- 存活情况: 红方 {info['red_alive']}/{self.env.num_red}, "
              f"蓝方 {info['blue_alive']}/{self.env.num_blue}")

    def test_experience_replay(self):
        """测试经验回放"""
        obs = self.env.reset()
        
        # 存储一些经验
        for _ in range(self.test_params['batch_size'] * 2):
            per_agent_obs = []
            for i in range(self.maddpg.num_agents):
                agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
                per_agent_obs.append(agent_obs)
            
            actions = self.maddpg.select_actions(per_agent_obs, noise=0.1)
            next_obs, rewards, done, _ = self.env.step(actions)
            
            # 将next_obs分割成每个智能体的观察
            next_per_agent_obs = []
            for i in range(self.maddpg.num_agents):
                next_agent_obs = next_obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
                next_per_agent_obs.append(next_agent_obs)
            
            self.maddpg.replay_buffer.add((per_agent_obs, actions, rewards, next_per_agent_obs, done))
            obs = next_obs
            
            if done:
                obs = self.env.reset()
        
        print(f"✓ 经验回放缓冲区大小: {self.maddpg.replay_buffer.size()}")
        
        # 测试采样的数据格式
        batch = self.maddpg.replay_buffer.sample(self.test_params['batch_size'])
        states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = batch
        
        # 验证采样数据的形状
        self.assertEqual(states_batch.shape[0], self.test_params['batch_size'])
        self.assertEqual(states_batch.shape[2], self.test_params['state_dim'])
        self.assertEqual(actions_batch.shape[0], self.test_params['batch_size'])
        self.assertEqual(actions_batch.shape[2], self.test_params['action_dim'])
        
        print("✓ 经验回放采样正确")
        print(f"- 状态批次形状: {states_batch.shape}")
        print(f"- 动作批次形状: {actions_batch.shape}")
        print(f"- 奖励批次形状: {rewards_batch.shape}")

    def test_short_training(self):
        """测试短期训练效果"""
        metrics = {
            'rewards_per_episode': [],
            'steps_per_episode': [],
            'success_per_episode': [],
            'noise_values': [],
        }
        
        current_noise = self.test_params['initial_noise']
        print("\n开始短期训练测试...")
        
        for episode in range(5):
            obs = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < self.test_params['max_steps']:
                per_agent_obs = []
                for i in range(self.maddpg.num_agents):
                    agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
                    per_agent_obs.append(agent_obs)
                
                actions = self.maddpg.select_actions(per_agent_obs, noise=current_noise)
                next_obs, rewards, done, info = self.env.step(actions)
                
                next_per_agent_obs = []
                for i in range(self.maddpg.num_agents):
                    next_agent_obs = next_obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
                    next_per_agent_obs.append(next_agent_obs)
                
                self.maddpg.replay_buffer.add((per_agent_obs, actions, rewards, next_per_agent_obs, done))
                
                if self.maddpg.replay_buffer.size() > self.test_params['batch_size']:
                    self.maddpg.update()
                
                episode_reward += sum(rewards)
                obs = next_obs
                step_count += 1
                
                if done:
                    break
            
            current_noise = max(
                self.test_params['min_noise'],
                current_noise * self.test_params['noise_decay']
            )
            
            metrics['rewards_per_episode'].append(float(episode_reward))
            metrics['steps_per_episode'].append(step_count)
            metrics['success_per_episode'].append(done)
            metrics['noise_values'].append(float(current_noise))
            
            print(f"Episode {episode + 1}: 奖励={episode_reward:.2f}, "
                  f"步数={step_count}, 成功={done}")
        
        metrics['moving_avg_rewards'] = pd.Series(
            metrics['rewards_per_episode']
        ).rolling(window=3, min_periods=1).mean().tolist()
        
        save_evaluation_report(self.test_dir, metrics, self.test_params)
        plot_training_curves(self.test_dir, metrics)
        
        print("\n测试训练结果:")
        print(f"- 平均奖励: {np.mean(metrics['rewards_per_episode']):.2f}")
        print(f"- 平均步数: {np.mean(metrics['steps_per_episode']):.2f}")
        print(f"- 成功率: {np.mean(metrics['success_per_episode']) * 100:.2f}%")

    def test_model_save_load(self):
        """测试模型保存和加载"""
        save_path = os.path.join(self.test_dir, 'test_save.pt')
        self.maddpg.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        new_maddpg = MADDPG(
            num_agents=self.test_params['num_red'] + self.test_params['num_blue'],
            state_dim=self.test_params['state_dim'],
            action_dim=self.test_params['action_dim'],
            lr=self.test_params['lr'],
            gamma=self.test_params['gamma'],
            tau=self.test_params['tau'],
            buffer_size=self.test_params['buffer_size'],
            batch_size=self.test_params['batch_size'],
            hidden_dim=self.test_params['hidden_dim']
        )
        
        new_maddpg.load(save_path)
        
        obs = self.env.reset()
        per_agent_obs = []
        for i in range(new_maddpg.num_agents):
            agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
            per_agent_obs.append(agent_obs)
        
        actions = new_maddpg.select_actions(per_agent_obs, noise=0.0)
        self.assertEqual(len(actions), new_maddpg.num_agents)
        print("✓ 模型保存加载测试通过")

def run_main_tests():
    """运行所有主函数测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMainFunction)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    run_main_tests()