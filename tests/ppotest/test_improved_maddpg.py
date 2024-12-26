# test_improved_maddpg.py

import unittest
import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from combat_sim.combat_env import CombatEnv  # 假设 CombatEnv 已经支持多线程
from agents.MADDPGPPO.improved_maddpg import ImprovedMADDPG
from agents.MADDPGPPO.networks import PPOActor, Critic, ValueNet, compute_gae
import logging
import multiprocessing as mp

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestImprovedMADDPG(unittest.TestCase):
    """测试改进版MADDPG的完整功能和性能"""
    
    @classmethod
    def setUpClass(cls):
        """设置测试环境和基本参数"""
        print("\n开始改进版MADDPG测试...")
        
        # 环境配置
        cls.env_config = {
            'num_red': 2,
            'num_blue': 3,
            'max_steps': 200,
            'field_size': 1000.0,
            'attack_range': 100.0,
            'min_speed': 10.0,
            'max_speed': 30.0,
            'max_turn_rate': np.pi/6,
            'hit_probability': 0.8,
            'num_threads': 8  # 假设支持多线程
        }
        
        cls.env = CombatEnv(**cls.env_config)
        
        # 获取观察空间维度
        cls.obs = cls.env.reset()
        cls.obs_dim = len(list(cls.obs.values())[0])
        cls.act_dim = 3  # [转向，速度，开火]
        
        # 创建保存目录
        cls.save_dir = './test_results'
        os.makedirs(cls.save_dir, exist_ok=True)

    def setUp(self):
        """每个测试用例开始前的设置"""
        self.maddpg = ImprovedMADDPG(
            n_red=self.env_config['num_red'],
            n_blue=self.env_config['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
        )
        
        # 记录性能指标
        self.metrics = defaultdict(list)

    def test_01_network_architecture(self):
        """测试网络架构"""
        print("\n测试网络架构...")
        
        # 测试PPOActor
        actor = PPOActor(self.obs_dim, self.act_dim)
        test_obs = torch.randn(1, self.obs_dim)
        mean, log_std = actor(test_obs)
        
        self.assertEqual(mean.shape, (1, self.act_dim))
        self.assertEqual(log_std.shape, (1, self.act_dim))
        
        # 测试动作采样
        action, log_prob = actor.sample_action(test_obs)
        self.assertEqual(action.shape, (1, self.act_dim))
        self.assertEqual(log_prob.shape, (1, 1))
        
        # 测试Critic
        critic = Critic(self.obs_dim, self.act_dim, self.env_config['num_red'] + self.env_config['num_blue'])
        test_actions = torch.randn(1, self.act_dim * (self.env_config['num_red'] + self.env_config['num_blue']))
        test_obs_all = torch.randn(1, self.obs_dim * (self.env_config['num_red'] + self.env_config['num_blue']))
        q_val = critic(test_obs_all, test_actions)
        self.assertEqual(q_val.shape, (1, 1))
        
        # 测试ValueNet
        value_net = ValueNet(self.obs_dim)
        value = value_net(test_obs)
        self.assertEqual(value.shape, (1, 1))
        
        print("网络架构测试完成")

    def test_02_action_selection(self):
        """测试动作选择"""
        print("\n测试动作选择...")
        
        obs = self.env.reset()
        
        # 测试确定性和随机动作
        actions_det, log_probs_det = self.maddpg.select_actions(obs, deterministic=True)
        actions_stoch, log_probs_stoch = self.maddpg.select_actions(obs, deterministic=False)
        
        # 验证动作和log概率的格式
        for agent_id in obs.keys():
            self.assertEqual(len(actions_det[agent_id]), self.act_dim)
            self.assertTrue(isinstance(log_probs_det[agent_id], float))
            self.assertTrue(np.all(actions_det[agent_id] >= -1))
            self.assertTrue(np.all(actions_det[agent_id] <= 1))
        
        # 验证随机性
        for agent_id in obs.keys():
            self.assertFalse(np.allclose(actions_det[agent_id], actions_stoch[agent_id]))
        
        print("动作选择测试完成")

    def test_03_memory_and_sampling(self):
        """测试优先级经验回放"""
        print("\n测试优先级经验回放...")
        
        # 存储一些经验
        for _ in range(self.maddpg.batch_size * 2):
            obs = self.env.reset()
            actions, log_probs = self.maddpg.select_actions(obs)
            next_obs, rewards, done, _ = self.env.step(actions)
            shaped_rewards = self.maddpg.compute_shaped_rewards(obs, actions, next_obs, rewards)
            self.maddpg.store_transition(obs, actions, shaped_rewards, next_obs, 
                                    {k: float(done) for k in obs.keys()}, log_probs)
        
        # 设置返回 log_probs
        self.maddpg.memory._return_log_probs = True
        
        # 测试采样
        self.assertGreater(len(self.maddpg.memory), self.maddpg.batch_size)
        
        sample_results = self.maddpg.memory.sample(self.maddpg.batch_size)
        self.assertEqual(len(sample_results), 8)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch, indices, weights, log_probs_batch = sample_results
        
        # 验证数据格式
        self.assertEqual(obs_batch.shape[0], self.maddpg.batch_size)
        self.assertEqual(weights.shape[0], self.maddpg.batch_size)
        self.assertEqual(log_probs_batch.shape[0], self.maddpg.batch_size)
        
        # 验证优先级更新
        td_errors = np.random.rand(self.maddpg.batch_size)
        self.maddpg.memory.update_priorities(indices, td_errors)
        
        print("优先级经验回放测试完成")



    def test_04_reward_shaping(self):
        """测试奖励塑形"""
        print("\n测试奖励塑形...")
        
        obs = self.env.reset()
        actions, _ = self.maddpg.select_actions(obs)
        next_obs, base_rewards, done, _ = self.env.step(actions)
        
        # 计算塑形后的奖励
        shaped_rewards = self.maddpg.compute_shaped_rewards(obs, actions, next_obs, base_rewards)
        
        # 验证奖励塑形的效果
        for agent_id in base_rewards.keys():
            self.assertIn(agent_id, shaped_rewards)
            # 塑形后的奖励应该不等于原始奖励
            self.assertNotEqual(base_rewards[agent_id], shaped_rewards[agent_id])
            
        print("奖励塑形测试完成")

    def test_05_training_step(self):
        """测试训练步骤"""
        print("\n测试训练步骤...")
        
        # 填充足够的经验
        for _ in range(self.maddpg.batch_size * 2):
            obs = self.env.reset()
            actions, log_probs = self.maddpg.select_actions(obs)
            next_obs, rewards, done, _ = self.env.step(actions)
            shaped_rewards = self.maddpg.compute_shaped_rewards(obs, actions, next_obs, rewards)
            self.maddpg.store_transition(obs, actions, shaped_rewards, next_obs,
                                       {k: float(done) for k in obs.keys()}, log_probs)
        
        # 测试训练步骤
        self.maddpg.train()
        
        print("训练步骤测试完成")

    def test_06_short_training(self):
        """测试短期训练效果"""
        print("\n测试短期训练效果...")
        
        n_episodes = 10
        total_steps = 0
        episode_rewards = []
        episode_lengths = []
        win_rates = []
        
        for episode in range(n_episodes):
            episode_reward = defaultdict(float)
            obs = self.env.reset()
            step = 0
            
            while True:
                actions, log_probs = self.maddpg.select_actions(obs)
                next_obs, rewards, done, info = self.env.step(actions)
                
                # 使用奖励塑形
                shaped_rewards = self.maddpg.compute_shaped_rewards(obs, actions, next_obs, rewards)
                
                # 存储经验
                self.maddpg.store_transition(obs, actions, shaped_rewards, next_obs,
                                           {k: float(done) for k in obs.keys()}, log_probs)
                
                # 训练
                self.maddpg.train()
                
                # 更新统计
                for agent_id, reward in shaped_rewards.items():
                    episode_reward[agent_id] += reward
                
                obs = next_obs
                step += 1
                
                if done:
                    break
            
            total_steps += step
            episode_rewards.append(sum(episode_reward.values()))
            episode_lengths.append(step)
            
            # 计算胜率
            red_win = info.get('blue_alive', 1) == 0
            win_rates.append(float(red_win))
            
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Reward: {episode_rewards[-1]:.2f}, "
                  f"Length: {step}, "
                  f"Win: {red_win}")
        
        # 记录性能指标
        self.metrics['episode_rewards'] = episode_rewards
        self.metrics['episode_lengths'] = episode_lengths
        self.metrics['win_rates'] = win_rates
        
        # 绘制训练曲线
        self._plot_training_curves()
        
        print("\n短期训练效果测试完成")

    def test_07_save_load(self):
        """测试模型保存和加载"""
        print("\n测试模型保存和加载...")
        
        save_path = os.path.join(self.save_dir, 'test_model.pt')
        
        # 保存模型
        self.maddpg.save(save_path)
        
        # 创建新的实例
        new_maddpg = ImprovedMADDPG(
            n_red=self.env_config['num_red'],
            n_blue=self.env_config['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
        )
        
        # 加载模型
        new_maddpg.load(save_path)
        
        # 比较动作
        obs = self.env.reset()
        actions1, _ = self.maddpg.select_actions(obs, deterministic=True)
        actions2, _ = new_maddpg.select_actions(obs, deterministic=True)
        
        # 验证动作一致性
        for agent_id in actions1:
            np.testing.assert_array_almost_equal(actions1[agent_id], actions2[agent_id])
        
        print("模型保存和加载测试完成")

    def test_08_gae_computation(self):
        """测试GAE计算"""
        print("\n测试GAE计算...")
        
        # 创建测试数据
        batch_size = 10
        seq_length = 20  # 假设一个batch包含多个时间步
        rewards = torch.rand(batch_size, seq_length)
        values = torch.rand(batch_size, seq_length)
        next_values = torch.rand(batch_size, seq_length)
        dones = torch.zeros(batch_size, seq_length)
        
        # 计算GAE
        advantages, returns = compute_gae(rewards, values, next_values, dones)
        
        # 验证输出
        self.assertEqual(advantages.shape, (batch_size, seq_length))
        self.assertEqual(returns.shape, (batch_size, seq_length))
        
        print("GAE计算测试完成")

    def test_09_performance_metrics(self):
        """计算和显示性能指标"""
        print("\n计算性能指标...")
        
        if not self.metrics['episode_rewards']:
            print("没有可用的性能数据")
            return
            
        # 计算指标
        avg_reward = np.mean(self.metrics['episode_rewards'])
        avg_length = np.mean(self.metrics['episode_lengths'])
        win_rate = np.mean(self.metrics['win_rates'])
        reward_std = np.std(self.metrics['episode_rewards'])
        
        print(f"\n性能指标统计:")
        print(f"平均奖励: {avg_reward:.2f} ± {reward_std:.2f}")
        print(f"平均回合长度: {avg_length:.2f}")
        print(f"胜率: {win_rate*100:.1f}%")
        
        print("性能指标计算完成")

    def _plot_training_curves(self):
        """绘制训练曲线"""
        if not self.metrics['episode_rewards']:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 奖励曲线
        axes[0].plot(self.metrics['episode_rewards'])
        axes[0].set_title('Episode Rewards')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].grid(True)
        
        # 回合长度曲线
        axes[1].plot(self.metrics['episode_lengths'])
        axes[1].set_title('Episode Lengths')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].grid(True)
        
        # 胜率曲线
        axes[2].plot(np.array(self.metrics['win_rates']) * 100)
        axes[2].set_title('Win Rate')
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Win Rate (%)')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()
        logging.info("训练曲线已保存到 training_curves.png")

    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        cls.env.close()
        print("\n改进版MADDPG测试完成!")

def run_tests_in_parallel(test_case, test_method):
    """在单独的进程中运行测试方法"""
    suite = unittest.TestSuite()
    suite.addTest(test_case(test_method))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

def main():
    """运行所有测试并支持多进程"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置matplotlib样式
    plt.style.use('seaborn')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    
    # 获取所有测试方法
    test_loader = unittest.TestLoader()
    test_names = test_loader.getTestCaseNames(TestImprovedMADDPG)
    
    # 创建测试进程池
    pool = mp.Pool(processes=min(len(test_names), mp.cpu_count()))
    
    # 启动每个测试方法在单独的进程中运行
    for test_method in test_names:
        pool.apply_async(run_tests_in_parallel, args=(TestImprovedMADDPG, test_method))
    
    pool.close()
    pool.join()
    
    print("\n性能比较:")
    print("1. 检查 ./test_results/training_curves.png 查看训练曲线")
    print("2. 运行基准测试进行对比")

if __name__ == '__main__':
    main()
