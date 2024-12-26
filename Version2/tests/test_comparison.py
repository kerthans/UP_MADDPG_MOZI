# test_comparison.py

import unittest
import torch
import numpy as np
from comparison_experiment import BaselineMADDPG, MADDPG
from env.combat_env import CombatEnv

class TestMADDPGComparison(unittest.TestCase):
    """测试MADDPG版本比较的测试类"""
    
    def setUp(self):
        """测试前的初始化设置"""
        self.config = {
            'num_episodes': 10,  # 用较小的回合数进行测试
            'max_steps': 50,
            'num_red': 3,
            'num_blue': 3,
            'state_dim': 9,
            'action_dim': 2,
            'hidden_dim': 256,
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'batch_size': 32,
            'buffer_size': int(1e4),
            'heterogeneous': True,
            'device': 'cpu'
        }
        
        # 设置随机种子以确保可重复性
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建环境和智能体
        self.env = CombatEnv(
            num_red=self.config['num_red'],
            num_blue=self.config['num_blue'],
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            max_steps=self.config['max_steps']
        )
        
        # 创建改进版和基础版MADDPG
        self.improved_agent = MADDPG(
            num_agents=self.config['num_red'] + self.config['num_blue'],
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            lr=self.config['lr'],
            gamma=self.config['gamma'],
            tau=self.config['tau'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            hidden_dim=self.config['hidden_dim'],
            device=self.config['device']
        )
        
        self.baseline_agent = BaselineMADDPG(
            num_agents=self.config['num_red'] + self.config['num_blue'],
            state_dim=self.config['state_dim'],
            action_dim=self.config['action_dim'],
            lr=self.config['lr'],
            gamma=self.config['gamma'],
            tau=self.config['tau'],
            buffer_size=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            hidden_dim=self.config['hidden_dim'],
            device=self.config['device']
        )
    
    def test_network_structure(self):
        """测试两个版本的网络结构是否不同"""
        # 验证基础版没有ICM网络
        self.assertIsNone(self.baseline_agent.icm)
        self.assertIsNone(self.baseline_agent.icm_optimizer)
        
        # 验证改进版有ICM网络
        self.assertIsNotNone(self.improved_agent.icm)
        self.assertIsNotNone(self.improved_agent.icm_optimizer)
        
        print("Network structure test passed!")

    def test_intrinsic_rewards(self):
        """测试内在奖励计算是否不同"""
        # 创建测试数据
        batch_size = 4
        states = torch.randn(batch_size, self.config['state_dim'] * (self.config['num_red'] + self.config['num_blue']))
        actions = torch.randn(batch_size, self.config['action_dim'] * (self.config['num_red'] + self.config['num_blue']))
        next_states = torch.randn(batch_size, self.config['state_dim'] * (self.config['num_red'] + self.config['num_blue']))
        
        # 计算两个版本的内在奖励
        baseline_rewards = self.baseline_agent._compute_intrinsic_rewards(states, actions, next_states)
        improved_rewards = self.improved_agent._compute_intrinsic_rewards(states, actions, next_states)
        
        # 验证基础版的内在奖励为零
        self.assertTrue(torch.all(baseline_rewards == 0))
        # 验证改进版的内在奖励不为零
        self.assertFalse(torch.all(improved_rewards == 0))
        
        print("Intrinsic rewards test passed!")

    def test_update_mechanism(self):
        """测试更新机制是否不同"""
        # 填充一些经验到回放缓冲区
        obs = self.env.reset()
        for _ in range(self.config['batch_size'] * 2):
            per_agent_obs = []
            for i in range(self.improved_agent.num_agents):
                agent_obs = obs[i * self.config['state_dim']:(i + 1) * self.config['state_dim']]
                per_agent_obs.append(agent_obs)
            
            # 获取动作
            improved_actions = self.improved_agent.select_actions(per_agent_obs, noise=0.1)
            baseline_actions = self.baseline_agent.select_actions(per_agent_obs, noise=0.1)
            
            # 执行动作
            next_obs, rewards, done, _ = self.env.step(improved_actions)
            
            # 存储经验
            self.improved_agent.replay_buffer.add(obs, improved_actions, rewards, next_obs, done)
            self.baseline_agent.replay_buffer.add(obs, baseline_actions, rewards, next_obs, done)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
        
        # 记录更新前的参数
        improved_params_before = self._get_network_params(self.improved_agent)
        baseline_params_before = self._get_network_params(self.baseline_agent)
        
        # 执行更新
        self.improved_agent.update()
        self.baseline_agent.update()
        
        # 记录更新后的参数
        improved_params_after = self._get_network_params(self.improved_agent)
        baseline_params_after = self._get_network_params(self.baseline_agent)
        
        # 验证参数变化是否不同
        improved_param_changes = self._compute_param_changes(improved_params_before, improved_params_after)
        baseline_param_changes = self._compute_param_changes(baseline_params_before, baseline_params_after)
        
        self.assertNotEqual(improved_param_changes, baseline_param_changes)
        print("Update mechanism test passed!")

    def _get_network_params(self, agent):
        """获取网络参数"""
        params = {}
        for i, actor in enumerate(agent.actors):
            params[f'actor_{i}'] = {name: param.data.clone() 
                                  for name, param in actor.named_parameters()}
        for i, critic in enumerate(agent.critics):
            params[f'critic_{i}'] = {name: param.data.clone() 
                                   for name, param in critic.named_parameters()}
        return params

    def _compute_param_changes(self, params_before, params_after):
        """计算参数变化的总量"""
        total_change = 0
        for key in params_before:
            for name in params_before[key]:
                total_change += torch.sum(torch.abs(
                    params_after[key][name] - params_before[key][name]
                )).item()
        return total_change

    def test_action_selection(self):
        """测试动作选择是否不同"""
        obs = self.env.reset()
        per_agent_obs = []
        for i in range(self.improved_agent.num_agents):
            agent_obs = obs[i * self.config['state_dim']:(i + 1) * self.config['state_dim']]
            per_agent_obs.append(agent_obs)
        
        improved_actions = self.improved_agent.select_actions(per_agent_obs, noise=0.1)
        baseline_actions = self.baseline_agent.select_actions(per_agent_obs, noise=0.1)
        
        # 验证动作不完全相同
        actions_diff = np.mean(np.abs(np.array(improved_actions) - np.array(baseline_actions)))
        self.assertGreater(actions_diff, 0)
        print("Action selection test passed!")

    def test_training_loop(self):
        """测试完整的训练循环"""
        # 训练几个回合
        improved_rewards = []
        baseline_rewards = []
        
        for episode in range(5):  # 测试5个回合
            # 改进版训练
            obs = self.env.reset()
            episode_reward = 0
            for step in range(20):  # 每个回合20步
                per_agent_obs = []
                for i in range(self.improved_agent.num_agents):
                    agent_obs = obs[i * self.config['state_dim']:(i + 1) * self.config['state_dim']]
                    per_agent_obs.append(agent_obs)
                
                actions = self.improved_agent.select_actions(per_agent_obs, noise=0.1)
                next_obs, rewards, done, _ = self.env.step(actions)
                episode_reward += sum(rewards)
                
                self.improved_agent.replay_buffer.add(obs, actions, rewards, next_obs, done)
                if self.improved_agent.replay_buffer.size() >= self.config['batch_size']:
                    self.improved_agent.update()
                
                if done:
                    break
                obs = next_obs
            improved_rewards.append(episode_reward)
            
            # 基础版训练
            obs = self.env.reset()
            episode_reward = 0
            for step in range(20):
                per_agent_obs = []
                for i in range(self.baseline_agent.num_agents):
                    agent_obs = obs[i * self.config['state_dim']:(i + 1) * self.config['state_dim']]
                    per_agent_obs.append(agent_obs)
                
                actions = self.baseline_agent.select_actions(per_agent_obs, noise=0.1)
                next_obs, rewards, done, _ = self.env.step(actions)
                episode_reward += sum(rewards)
                
                self.baseline_agent.replay_buffer.add(obs, actions, rewards, next_obs, done)
                if self.baseline_agent.replay_buffer.size() >= self.config['batch_size']:
                    self.baseline_agent.update()
                
                if done:
                    break
                obs = next_obs
            baseline_rewards.append(episode_reward)
        
        # 验证两个版本的表现不同
        improved_mean = np.mean(improved_rewards)
        baseline_mean = np.mean(baseline_rewards)
        self.assertNotEqual(improved_mean, baseline_mean)
        print("Training loop test passed!")

def run_tests():
    """运行所有测试"""
    # 创建测试套件
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMADDPGComparison)
    
    # 运行测试
    print("\nRunning MADDPG comparison tests...")
    print("="*50)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print("="*50)

if __name__ == '__main__':
    run_tests()