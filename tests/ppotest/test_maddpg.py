# test_improved_maddpg.py

import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import time
import os
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
from datetime import datetime
import seaborn as sns
from multiprocessing import Pool
from agents.MADDPGPPO.improved_maddpg import ImprovedMADDPG
from combat_sim.combat_env import CombatEnv

# 设置matplotlib样式
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class TestImprovedMADDPG(unittest.TestCase):
    """专门测试改进版MADDPG的性能"""

    @classmethod
    def setUpClass(cls):
        """设置日志记录"""
        # 创建目录
        for dir_name in ['logs', 'results', 'checkpoints']:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
        
        # 设置详细日志
        log_filename = f'logs/improved_maddpg_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        cls.logger = logging.getLogger(__name__)
    
    def setUp(self):
        """测试初始化"""
        self.logger.info("="*50)
        self.logger.info("开始新的测试回合")
        self.logger.info("="*50)
        
        # 训练参数
        self.num_episodes = 200  # 增加训练轮次
        self.eval_interval = 25   # 更频繁的评估
        self.num_eval_episodes = 30  # 更多的评估轮次
        self.save_interval = 50   # 模型保存间隔
        
        # 环境参数
        self.num_red = 2
        self.num_blue = 3
        
        # 初始化环境
        self.env = CombatEnv(num_red=self.num_red, num_blue=self.num_blue, num_threads=8)  # 多线程支持
        obs = self.env.reset()
        self.obs_dim = len(obs['red_0'])
        self.act_dim = 3
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 初始化智能体
        self.agent = ImprovedMADDPG(
            self.num_red,
            self.num_blue,
            self.obs_dim,
            self.act_dim
        )
        
        # 性能指标存储
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
        self.logger.info(f"环境初始化完成:")
        self.logger.info(f"- 观察空间维度: {self.obs_dim}")
        self.logger.info(f"- 动作空间维度: {self.act_dim}")
        self.logger.info(f"- 红方数量: {self.num_red}")
        self.logger.info(f"- 蓝方数量: {self.num_blue}")
    
    def evaluate_agent(self, episode_num: int = 0):
        """详细的智能体评估"""
        self.logger.info(f"\n开始第 {episode_num} 轮评估...")
        
        eval_metrics = {
            "rewards": [],
            "episode_lengths": [],
            "wins": 0,
            "red_hits": [],
            "blue_hits": [],
            "red_survival": [],
            "blue_survival": [],
            "engagement_distances": [],
            "action_distributions": defaultdict(list)
        }
        
        for ep in tqdm(range(self.num_eval_episodes), desc="评估进度"):
            obs = self.env.reset()
            episode_reward = 0
            step_count = 0
            episode_hits = {"red": 0, "blue": 0}
            min_distances = []
            
            while True:
                # 记录动作分布
                actions, _ = self.agent.select_actions(obs, deterministic=True)
                for agent_id, action in actions.items():
                    eval_metrics["action_distributions"][agent_id].append(action)
                
                # 环境步进
                next_obs, rewards, done, info = self.env.step(actions)
                
                # 记录最小交战距离
                red_pos = np.array([obs[f'red_{i}'][:2] for i in range(self.num_red)])
                blue_pos = np.array([obs[f'blue_{i}'][:2] for i in range(self.num_blue)])
                distances = []
                for r_pos in red_pos:
                    for b_pos in blue_pos:
                        distances.append(np.linalg.norm(r_pos - b_pos))
                if distances:
                    min_distances.append(min(distances))
                else:
                    min_distances.append(0.0)
                
                # 更新统计
                episode_reward += sum(rewards.values())
                episode_hits["red"] += info.get("red_hits", 0)
                episode_hits["blue"] += info.get("blue_hits", 0)
                
                obs = next_obs
                step_count += 1
                
                if done:
                    break
            
            # 记录本轮评估指标
            eval_metrics["rewards"].append(episode_reward)
            eval_metrics["episode_lengths"].append(step_count)
            eval_metrics["wins"] += 1 if info.get("red_alive", 1) > info.get("blue_alive", 0) else 0
            eval_metrics["red_hits"].append(episode_hits["red"])
            eval_metrics["blue_hits"].append(episode_hits["blue"])
            eval_metrics["red_survival"].append(info.get("red_alive", self.num_red) / self.num_red)
            eval_metrics["blue_survival"].append(info.get("blue_alive", self.num_blue) / self.num_blue)
            eval_metrics["engagement_distances"].append(np.mean(min_distances))
        
        # 计算统计结果
        results = {
            "avg_reward": np.mean(eval_metrics["rewards"]),
            "std_reward": np.std(eval_metrics["rewards"]),
            "win_rate": eval_metrics["wins"] / self.num_eval_episodes,
            "avg_episode_length": np.mean(eval_metrics["episode_lengths"]),
            "red_hit_rate": np.mean(eval_metrics["red_hits"]) / np.mean(eval_metrics["episode_lengths"]),
            "blue_hit_rate": np.mean(eval_metrics["blue_hits"]) / np.mean(eval_metrics["episode_lengths"]),
            "avg_red_survival": np.mean(eval_metrics["red_survival"]),
            "avg_blue_survival": np.mean(eval_metrics["blue_survival"]),
            "avg_engagement_distance": np.mean(eval_metrics["engagement_distances"])
        }
        
        # 详细日志输出
        self.logger.info("\n评估结果:")
        self.logger.info(f"平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        self.logger.info(f"胜率: {results['win_rate']:.2%}")
        self.logger.info(f"平均回合长度: {results['avg_episode_length']:.1f}")
        self.logger.info(f"红方命中率: {results['red_hit_rate']:.3f}")
        self.logger.info(f"蓝方命中率: {results['blue_hit_rate']:.3f}")
        self.logger.info(f"红方存活率: {results['avg_red_survival']:.2%}")
        self.logger.info(f"蓝方存活率: {results['avg_blue_survival']:.2%}")
        self.logger.info(f"平均交战距离: {results['avg_engagement_distance']:.1f}")
        
        return results, eval_metrics
    
    def test_improved_agent(self):
        """完整的训练和测试流程"""
        self.logger.info("\n开始训练改进版MADDPG...")
        training_start = time.time()
        
        for episode in tqdm(range(self.num_episodes), desc="训练进度"):
            episode_start = time.time()
            obs = self.env.reset()
            episode_reward = 0
            step = 0
            episode_hits = {"red": 0, "blue": 0}
            
            while True:
                # 选择动作
                actions, log_probs = self.agent.select_actions(obs)
                next_obs, rewards, done, info = self.env.step(actions)
                
                # 计算塑形奖励
                shaped_rewards = self.agent.compute_shaped_rewards(obs, actions, next_obs, rewards)
                
                # 存储经验
                # Ensure that 'done' is a dict with float values per agent
                dones = {k: float(done) for k in obs.keys()}
                self.agent.store_transition(obs, actions, shaped_rewards, next_obs, dones, log_probs)
                
                # 更新统计
                episode_reward += sum(rewards.values())
                episode_hits["red"] += info.get("red_hits", 0)
                episode_hits["blue"] += info.get("blue_hits", 0)
                
                obs = next_obs
                step += 1
                
                # 训练智能体
                self.agent.train()
                
                if done:
                    break
            
            # 记录每轮训练指标
            self.episode_metrics["rewards"].append(episode_reward)
            self.episode_metrics["lengths"].append(step)
            self.episode_metrics["red_hits"].append(episode_hits["red"])
            self.episode_metrics["blue_hits"].append(episode_hits["blue"])
            
            # 定期评估
            if (episode + 1) % self.eval_interval == 0:
                results, detailed_metrics = self.evaluate_agent(episode + 1)
                
                # 记录评估指标
                for key, value in results.items():
                    self.metrics[key].append(value)
                
                # 保存检查点
                if (episode + 1) % self.save_interval == 0:
                    checkpoint_path = f'checkpoints/improved_maddpg_ep{episode+1}.pt'
                    self.agent.save(checkpoint_path)
                    self.logger.info(f"模型已保存至: {checkpoint_path}")
                
                # 绘制阶段性分析图
                self._plot_training_progress(episode + 1)
        
        training_time = time.time() - training_start
        self.logger.info(f"\n训练完成! 总用时: {training_time:.2f}秒")
        
        # 最终性能评估
        final_results, _ = self.evaluate_agent(self.num_episodes)
        
        # 验证性能指标
        self.assertGreater(final_results["win_rate"], 0.4, "最终胜率过低")
        self.assertGreater(final_results["red_hit_rate"], 0.1, "红方命中率过低")
        self.assertLess(final_results["blue_hit_rate"], 0.2, "蓝方命中率过高")
        
        # 保存完整测试报告
        self._save_test_report(training_time)
    
    def _plot_training_progress(self, episode: int):
        """绘制训练进度分析图"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # 奖励曲线
        sns.lineplot(data=self.episode_metrics["rewards"], ax=axes[0,0])
        axes[0,0].set_title("Episode Rewards")
        axes[0,0].set_xlabel("Episode")
        axes[0,0].set_ylabel("Total Reward")
        
        # 胜率曲线
        sns.lineplot(data=self.metrics["win_rate"], ax=axes[0,1])
        axes[0,1].set_title("Win Rate")
        axes[0,1].set_xlabel("Episode")
        axes[0,1].set_ylabel("Win Rate")
        
        # 命中率对比
        sns.lineplot(data=self.metrics["red_hit_rate"], ax=axes[1,0], label="Red")
        sns.lineplot(data=self.metrics["blue_hit_rate"], ax=axes[1,0], label="Blue")
        axes[1,0].set_title("Hit Rates")
        axes[1,0].set_xlabel("Episode")
        axes[1,0].set_ylabel("Hit Rate")
        axes[1,0].legend()
        
        # 存活率对比
        sns.lineplot(data=self.metrics["avg_red_survival"], ax=axes[1,1], label="Red")
        sns.lineplot(data=self.metrics["avg_blue_survival"], ax=axes[1,1], label="Blue")
        axes[1,1].set_title("Survival Rates")
        axes[1,1].set_xlabel("Episode")
        axes[1,1].set_ylabel("Survival Rate")
        axes[1,1].legend()
        
        # 回合长度
        sns.lineplot(data=self.episode_metrics["lengths"], ax=axes[2,0])
        axes[2,0].set_title("Episode Lengths")
        axes[2,0].set_xlabel("Episode")
        axes[2,0].set_ylabel("Steps")
        
        # 交战距离
        sns.lineplot(data=self.metrics["avg_engagement_distance"], ax=axes[2,1])
        axes[2,1].set_title("Average Engagement Distance")
        axes[2,1].set_xlabel("Episode")
        axes[2,1].set_ylabel("Average Distance")
        
        plt.tight_layout()
        plt.savefig(f"results/training_progress_ep{episode}.png")
        plt.close()
        self.logger.info(f"训练进度分析图已保存至: results/training_progress_ep{episode}.png")
    
    def _save_test_report(self, training_time: float):
        """保存完整的测试报告"""
        report = {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "training_time": training_time,
            "configuration": {
                "num_episodes": self.num_episodes,
                "eval_interval": self.eval_interval,
                "num_eval_episodes": self.num_eval_episodes,
                "num_red": self.num_red,
                "num_blue": self.num_blue,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim
            },
            "final_metrics": self.metrics,
            "training_metrics": dict(self.episode_metrics)
        }
        
        report_path = f'results/test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"测试报告已保存至: {report_path}")
    
    @classmethod
    def tearDownClass(cls):
        """清理测试环境"""
        cls.env.close()
        cls.logger.info("\n改进版MADDPG测试完成!")
    
    if __name__ == '__main__':
        unittest.main(verbosity=2)
