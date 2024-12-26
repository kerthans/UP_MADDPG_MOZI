import unittest
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import sys
import os
import time
from typing import Dict, List, Tuple

# 修改导入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combat_sim.combat_env import CombatEnv
from agents.up import MADDPG, Actor, Critic

class TestCombatSystem(unittest.TestCase):
    """增强版测试套件 - 测试强化学习环境和网络"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        cls.num_red = 2
        cls.num_blue = 3
        cls.obs_dim = 49
        cls.act_dim = 3
        
        # 环境配置
        cls.env_config = {
            'num_red': cls.num_red,
            'num_blue': cls.num_blue,
            'max_steps': 200,
            'field_size': 1000.0,
            'attack_range': 100.0,
            'min_speed': 10.0,
            'max_speed': 30.0,
            'hit_probability': 0.8
        }
        
        # 网络配置 - 修改 buffer_size 为 capacity
        cls.network_config = {
            'n_red': cls.num_red,
            'n_blue': cls.num_blue,
            'obs_dim': cls.obs_dim,
            'act_dim': cls.act_dim,
            'gamma': 0.99,
            'lr_actor': 3e-5,
            'lr_critic': 1e-4,
            'batch_size': 256,
            'capacity': 1000000  # 原来是 buffer_size
        }
        
        cls.env = CombatEnv(**cls.env_config)
        cls.maddpg = MADDPG(**cls.network_config)
        
        # 测试结果存储
        cls.test_results = defaultdict(list)

    def setUp(self):
        """每个测试方法前运行"""
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)

    def test_environment_initialization(self):
        """测试环境初始化"""
        print("\n测试环境初始化...")
        
        # 测试重置
        obs = self.env.reset()
        
        # 检查观察空间
        for agent_id, agent_obs in obs.items():
            self.assertEqual(len(agent_obs), self.obs_dim, 
                           f"{agent_id} 观察维度错误: {len(agent_obs)} != {self.obs_dim}")
            self.assertTrue(np.all(np.isfinite(agent_obs)), 
                          f"{agent_id} 观察值包含无效数值")
        
        # 检查初始状态
        self.assertEqual(np.sum(self.env.red_states[:, 4]), self.num_red,
                        "红方初始存活数量错误")
        self.assertEqual(np.sum(self.env.blue_states[:, 4]), self.num_blue,
                        "蓝方初始存活数量错误")
        
        # 检查位置边界
        red_positions = self.env.red_states[:, :2]
        blue_positions = self.env.blue_states[:, :2]
        
        self.assertTrue(np.all(red_positions[:, 0] <= self.env_config['field_size'] / 4),
                       "红方初始x坐标超出范围")
        self.assertTrue(np.all(blue_positions[:, 0] >= 3 * self.env_config['field_size'] / 4),
                       "蓝方初始x坐标超出范围")

    def test_environment_dynamics(self):
        """测试环境动力学"""
        print("\n测试环境动力学...")
        
        obs = self.env.reset()
        initial_states = {
            'red': self.env.red_states.copy(),
            'blue': self.env.blue_states.copy()
        }
        
        # 测试动作执行
        actions = {}
        for agent_id in obs.keys():
            # 测试特定动作
            actions[agent_id] = np.array([0.5, 0.8, 0])  # 转向和速度变化，不开火
        
        next_obs, rewards, done, info = self.env.step(actions)
        
        # 检查状态更新
        for team in ['red', 'blue']:
            current_states = self.env.red_states if team == 'red' else self.env.blue_states
            initial_team_states = initial_states[team]
            
            # 检查位置更新
            position_changes = np.linalg.norm(
                current_states[:, :2] - initial_team_states[:, :2],
                axis=1
            )
            
            # 检查速度约束
            self.assertTrue(np.all(current_states[:, 3] >= self.env_config['min_speed']), 
                          f"{team}方速度小于最小速度")
            self.assertTrue(np.all(current_states[:, 3] <= self.env_config['max_speed']),
                          f"{team}方速度超过最大速度")

    # def test_reward_mechanism(self):
    #     """测试奖励机制"""
    #     print("\n测试奖励机制...")
    #     n_episodes = 50
    #     reward_stats = defaultdict(lambda: defaultdict(list))
        
    #     for episode in range(n_episodes):
    #         obs = self.env.reset()
    #         done = False
    #         episode_info = defaultdict(lambda: defaultdict(float))
            
    #         while not done:
    #             # 使用不同的动作策略
    #             if episode % 3 == 0:  # 随机动作
    #                 actions = {agent_id: self.env.action_space.sample() 
    #                          for agent_id in obs.keys()}
    #             elif episode % 3 == 1:  # 靠近对手
    #                 actions = self._generate_approach_actions(obs)
    #             else:  # 远离对手
    #                 actions = self._generate_retreat_actions(obs)
                
    #             next_obs, rewards, done, info = self.env.step(actions)
                
    #             # 记录各类奖励组成
    #             for agent_id, reward in rewards.items():
    #                 episode_info[agent_id]['total'] += reward
    #                 # 记录击中奖励、距离奖励等...
                
    #             obs = next_obs
            
    #         # 统计每个智能体的奖励
    #         for agent_id, stats in episode_info.items():
    #             for key, value in stats.items():
    #                 reward_stats[agent_id][key].append(value)
        
    #     # 分析奖励统计
    #     self._analyze_rewards(reward_stats)

    def test_network_architecture(self):
        """测试网络架构"""
        print("\n测试网络架构...")
        
        # 测试Actor网络
        actor = Actor(self.obs_dim, self.act_dim)
        test_obs = torch.randn(4, self.obs_dim)  # batch_size = 4
        actor_output = actor(test_obs)
        
        self.assertEqual(actor_output.shape, (4, self.act_dim),
                        "Actor输出维度错误")
        self.assertTrue(torch.all(actor_output >= -1) and torch.all(actor_output <= 1),
                       "Actor输出范围错误")
        
        # 测试Critic网络
        critic = Critic(self.obs_dim, self.act_dim, self.num_red + self.num_blue)
        test_obs_all = torch.randn(4, (self.num_red + self.num_blue) * self.obs_dim)
        test_act_all = torch.randn(4, (self.num_red + self.num_blue) * self.act_dim)
        critic_output = critic(test_obs_all, test_act_all)
        
        self.assertEqual(critic_output.shape, (4, 1),
                        "Critic输出维度错误")

    # def test_training_stability(self):
    #     """测试训练稳定性"""
    #     print("\n测试训练稳定性...")
    #     n_episodes = 20
    #     training_stats = defaultdict(list)
        
    #     # 收集初始经验
    #     self._collect_initial_experience(1000)
        
    #     # 训练循环
    #     for episode in range(n_episodes):
    #         episode_start = time.time()
    #         obs = self.env.reset()
    #         episode_reward = 0
    #         step_count = 0
    #         training_time = 0
    #         action_time = 0
            
    #         while True:
    #             # 计时动作选择
    #             action_start = time.time()
    #             actions = self.maddpg.select_actions(obs, noise_scale=0.1)
    #             action_time += time.time() - action_start
                
    #             next_obs, rewards, done, info = self.env.step(actions)
    #             step_count += 1
                
    #             # 计时训练过程
    #             train_start = time.time()
    #             self.maddpg.store_transition(obs, actions, rewards, next_obs, done)
    #             self.maddpg.train()
    #             training_time += time.time() - train_start
                
    #             episode_reward += sum(rewards.values())
    #             obs = next_obs
                
    #             if done:
    #                 break
            
    #         # 记录统计信息
    #         episode_time = time.time() - episode_start
    #         training_stats['rewards'].append(episode_reward)
    #         training_stats['steps'].append(step_count)
    #         training_stats['total_time'].append(episode_time)
    #         training_stats['training_time'].append(training_time)
    #         training_stats['action_time'].append(action_time)
            
    #         print(f"Episode {episode + 1}/{n_episodes}:")
    #         print(f"  Reward: {episode_reward:.2f}")
    #         print(f"  Steps: {step_count}")
    #         print(f"  Training time: {training_time:.2f}s")
    #         print(f"  Action selection time: {action_time:.2f}s")
    #         print(f"  Total time: {episode_time:.2f}s")
        
    #     self._analyze_training_stability(training_stats)

    def test_integrated_performance(self):
        """测试系统整体性能"""
        print("\n测试系统整体性能...")
        n_episodes = 30
        test_scenarios = [
            ('standard', {}),
            ('high_speed', {'max_speed': 50.0}),
            ('large_field', {'field_size': 2000.0}),
            ('dense_agents', {'num_red': 4, 'num_blue': 4})
        ]
        
        for scenario_name, config_changes in test_scenarios:
            print(f"\n测试场景: {scenario_name}")
            # 更新环境配置
            temp_config = self.env_config.copy()
            temp_config.update(config_changes)
            temp_env = CombatEnv(**temp_config)
            
            scenario_results = self._evaluate_scenario(temp_env, n_episodes)
            self.test_results[scenario_name] = scenario_results

    def _generate_approach_actions(self, obs):
        """生成接近对手的动作"""
        actions = {}
        for agent_id, agent_obs in obs.items():
            # 根据观察计算最近对手方向
            actions[agent_id] = np.array([0.8, 1.0, 0])  # 简化版本
        return actions

    def _generate_retreat_actions(self, obs):
        """生成远离对手的动作"""
        actions = {}
        for agent_id, agent_obs in obs.items():
            actions[agent_id] = np.array([-0.8, 1.0, 0])  # 简化版本
        return actions

    def _collect_initial_experience(self, n_steps):
        """收集初始经验"""
        obs = self.env.reset()
        for _ in range(n_steps):
            actions = {agent_id: self.env.action_space.sample() 
                      for agent_id in obs.keys()}
            next_obs, rewards, done, _ = self.env.step(actions)
            self.maddpg.store_transition(obs, actions, rewards, next_obs, done)
            obs = next_obs if not done else self.env.reset()

    def _analyze_rewards(self, reward_stats):
        """分析奖励统计"""
        print("\n奖励统计分析:")
        for agent_id, stats in reward_stats.items():
            print(f"\n{agent_id}:")
            for key, values in stats.items():
                mean = np.mean(values)
                std = np.std(values)
                print(f"  {key}: {mean:.2f} ± {std:.2f}")

    def _analyze_training_stability(self, stats):
        """分析训练稳定性"""
        print("\n训练稳定性分析:")
        
        # 奖励趋势
        rewards = np.array(stats['rewards'])
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        print(f"平均奖励: {reward_mean:.2f} ± {reward_std:.2f}")
        
        # 计算收敛性指标
        window_size = 5
        rolling_mean = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        convergence_speed = np.abs(np.diff(rolling_mean)).mean()
        print(f"收敛速度指标: {convergence_speed:.4f}")
        
        # 时间效率
        avg_step_time = np.mean(stats['total_time']) / np.mean(stats['steps'])
        print(f"平均每步时间: {avg_step_time:.4f}s")
        
        # 训练/推理时间比
        train_inference_ratio = np.mean(stats['training_time']) / np.mean(stats['action_time'])
        print(f"训练/推理时间比: {train_inference_ratio:.2f}")

    def _evaluate_scenario(self, env, n_episodes) -> Dict:
        """评估特定场景下的性能"""
        results = defaultdict(list)
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            hits = {'red': 0, 'blue': 0}
            
            while True:
                actions = self.maddpg.select_actions(obs, noise_scale=0.1)
                next_obs, rewards, done, info = env.step(actions)
                episode_reward += sum(rewards.values())
                hits['red'] += info['red_hits']
                hits['blue'] += info['blue_hits']
                step_count += 1
                
                obs = next_obs
                if done:
                    break
            
            # 记录结果
            results['rewards'].append(episode_reward)
            results['steps'].append(step_count)
            results['red_hits'].append(hits['red'])
            results['blue_hits'].append(hits['blue'])
            results['red_alive'].append(info['red_alive'])
            results['blue_alive'].append(info['blue_alive'])
            
            print(f"Episode {episode + 1}:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Steps: {step_count}")
            print(f"  Hits - Red: {hits['red']}, Blue: {hits['blue']}")
            print(f"  Alive - Red: {info['red_alive']}, Blue: {info['blue_alive']}")
        
        # 计算统计数据
        stats = {
            'mean_reward': np.mean(results['rewards']),
            'std_reward': np.std(results['rewards']),
            'mean_steps': np.mean(results['steps']),
            'mean_red_hits': np.mean(results['red_hits']),
            'mean_blue_hits': np.mean(results['blue_hits']),
            'mean_red_survival': np.mean(results['red_alive']),
            'mean_blue_survival': np.mean(results['blue_alive'])
        }
        
        return stats

    def generate_test_report(self):
        """生成综合测试报告"""
        print("\n=====================================")
        print("       综合测试报告")
        print("=====================================\n")
        
        # 1. 环境性能分析
        print("1. 环境性能分析")
        print("-----------------")
        print("基础环境参数:")
        for key, value in self.env_config.items():
            print(f"  {key}: {value}")
        print("\n")
        
        # 2. 网络性能分析
        print("2. 网络性能分析")
        print("-----------------")
        print("网络配置:")
        for key, value in self.network_config.items():
            print(f"  {key}: {value}")
        print("\n")
        
        # 3. 场景测试结果
        print("3. 场景测试结果")
        print("-----------------")
        for scenario, results in self.test_results.items():
            print(f"\n场景: {scenario}")
            print("  性能指标:")
            for metric, value in results.items():
                print(f"    {metric}: {value:.2f}")
        print("\n")
        
        # 4. 性能建议
        print("4. 性能改进建议")
        print("-----------------")
        recommendations = self._generate_recommendations()
        for area, suggestions in recommendations.items():
            print(f"\n{area}:")
            for suggestion in suggestions:
                print(f"  - {suggestion}")

    def _generate_recommendations(self) -> Dict[str, List[str]]:
        """根据测试结果生成改进建议"""
        recommendations = defaultdict(list)
        
        # 分析奖励稳定性
        reward_variations = [stats['std_reward'] for stats in self.test_results.values()]
        if np.mean(reward_variations) > 300:
            recommendations['奖励系统'].append("考虑降低奖励波动，可能需要重新设计奖励缩放机制")
            recommendations['奖励系统'].append("建议实施奖励归一化和裁剪")
        
        # 分析训练效率
        if hasattr(self, 'training_stats'):
            avg_episode_time = np.mean(self.training_stats['total_time'])
            if avg_episode_time > 2.0:  # 假设2秒为阈值
                recommendations['训练效率'].append("考虑优化计算密集型操作，可能需要简化网络结构")
                recommendations['训练效率'].append("建议增加batch size以提高训练效率")
        
        # 分析智能体行为
        for scenario_results in self.test_results.values():
            if scenario_results.get('mean_steps', 0) >= self.env_config['max_steps'] * 0.9:
                recommendations['智能体行为'].append("智能体可能存在决策效率问题，建议调整动作空间或奖励设计")
                recommendations['智能体行为'].append("考虑添加时间压力相关的奖励项")
        
        # 分析整体平衡性
        for results in self.test_results.values():
            red_effectiveness = results.get('mean_red_hits', 0) / max(results.get('mean_steps', 1), 1)
            blue_effectiveness = results.get('mean_blue_hits', 0) / max(results.get('mean_steps', 1), 1)
            if abs(red_effectiveness - blue_effectiveness) > 0.2:
                recommendations['系统平衡性'].append("红蓝双方效能差异较大，建议调整初始配置或奖励机制")
                recommendations['系统平衡性'].append("考虑修改环境参数以平衡双方优势")
        
        return recommendations

if __name__ == '__main__':
    # 运行所有测试
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestCombatSystem))
    
    runner = unittest.TextTestRunner(verbosity=2)
    test_results = runner.run(test_suite)
    
    # 生成测试报告
    if test_results.wasSuccessful():
        test = TestCombatSystem()
        test.setUpClass()
        test.test_integrated_performance()  # 运行综合性能测试
        test.generate_test_report()  # 生成报告