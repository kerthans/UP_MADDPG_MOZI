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

from agents.baseline import MADDPG as BaselineMADDPG
# from agents.improved_maddpg import ImprovedMADDPG
from agents.up import MADDPG as ImprovedMADDPG
from combat_sim.combat_env import CombatEnv

class TestMADDPGComparison(unittest.TestCase):
    """比较基准MADDPG和改进版MADDPG的性能测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置日志记录"""
        # 创建logs目录
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # 设置日志
        log_filename = f'logs/maddpg_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
        self.logger.info("初始化测试环境...")
        
        self.num_episodes = 150
        self.eval_interval = 50
        self.num_eval_episodes = 20
        
        # 环境参数
        self.num_red = 2
        self.num_blue = 3
        
        # 初始化环境
        self.env = CombatEnv(
            num_red=self.num_red,
            num_blue=self.num_blue
        )
        
        # 从环境中获取真实的观察和动作空间维度
        obs = self.env.reset()
        self.obs_dim = len(obs['red_0'])
        self.act_dim = 3
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 初始化智能体
        self.baseline_agent = BaselineMADDPG(
            self.num_red,
            self.num_blue,
            self.obs_dim,
            self.act_dim
        )
        
        self.improved_agent = ImprovedMADDPG(
            self.num_red,
            self.num_blue,
            self.obs_dim,
            self.act_dim
        )
        
        # 性能指标存储
        self.metrics = defaultdict(list)
        
        self.logger.info(f"初始化完成. 观察空间维度: {self.obs_dim}, 动作空间维度: {self.act_dim}")
    
    def evaluate_agent(self, agent, model_type="baseline"):
        """评估智能体性能"""
        self.logger.info(f"开始评估 {model_type} 智能体...")
        total_rewards = []
        win_rate = 0
        avg_episode_length = 0
        survival_rates = {"red": 0.0, "blue": 0.0}
        hit_rates = {"red": 0.0, "blue": 0.0}
        
        for ep in tqdm(range(self.num_eval_episodes), desc=f"评估 {model_type}"):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            hits = {"red": 0, "blue": 0}
            
            while not done:
                if model_type == "baseline":
                    actions = agent.select_actions(obs, noise_scale=0.0)
                else:
                    actions, _ = agent.select_actions(obs, deterministic=True)
                
                next_obs, rewards, done, info = self.env.step(actions)
                episode_reward += sum(rewards.values())
                hits["red"] += info["red_hits"]
                hits["blue"] += info["blue_hits"]
                
                obs = next_obs
                step_count += 1
            
            total_rewards.append(episode_reward)
            avg_episode_length += step_count
            
            if info["red_alive"] > info["blue_alive"]:
                win_rate += 1
                
            survival_rates["red"] += info["red_alive"] / self.num_red
            survival_rates["blue"] += info["blue_alive"] / self.num_blue
            
            total_shots = hits["red"] + hits["blue"]
            if total_shots > 0:
                hit_rates["red"] += hits["red"] / total_shots
                hit_rates["blue"] += hits["blue"] / total_shots
        
        results = {
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "win_rate": win_rate / self.num_eval_episodes,
            "avg_episode_length": avg_episode_length / self.num_eval_episodes,
            "survival_rates": {k: v/self.num_eval_episodes for k, v in survival_rates.items()},
            "hit_rates": {k: v/self.num_eval_episodes for k, v in hit_rates.items()}
        }
        
        self.logger.info(f"{model_type} 评估结果: {json.dumps(results, indent=2)}")
        return results
    
    def train_and_evaluate(self, agent, model_type="baseline"):
        """训练并评估模型"""
        self.logger.info(f"开始训练 {model_type} 模型...")
        training_start = time.time()
        
        for episode in tqdm(range(self.num_episodes), desc=f"训练 {model_type}"):
            obs = self.env.reset()
            episode_reward = 0
            step = 0
            
            while True:
                if model_type == "baseline":
                    actions = agent.select_actions(obs)
                    next_obs, rewards, done, info = self.env.step(actions)
                    agent.store_transition(obs, actions, rewards, next_obs, done)
                else:
                    actions, log_probs = agent.select_actions(obs)
                    next_obs, rewards, done, info = self.env.step(actions)
                    shaped_rewards = agent.compute_shaped_rewards(obs, actions, next_obs, rewards)
                    agent.store_transition(obs, actions, shaped_rewards, next_obs, done, log_probs)
                
                obs = next_obs
                episode_reward += sum(rewards.values())
                step += 1
                
                agent.train()
                
                if done:
                    break
            
            if (episode + 1) % self.eval_interval == 0:
                eval_metrics = self.evaluate_agent(agent, model_type)
                
                # 记录性能指标
                for key, value in eval_metrics.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            self.metrics[f"{model_type}_{key}_{sub_key}"].append(sub_value)
                    else:
                        self.metrics[f"{model_type}_{key}"].append(value)
                
                self.logger.info(f"\n{model_type.capitalize()} Episode {episode + 1}")
                self.logger.info(f"Average Reward: {eval_metrics['avg_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                self.logger.info(f"Win Rate: {eval_metrics['win_rate']:.2%}")
                self.logger.info(f"Average Episode Length: {eval_metrics['avg_episode_length']:.1f}")
        
        training_time = time.time() - training_start
        self.metrics[f"{model_type}_training_time"] = training_time
        self.logger.info(f"{model_type} 训练完成，总用时: {training_time:.2f}秒")
        
        # 保存训练指标
        metrics_file = f'logs/{model_type}_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        return agent
    
    def test_baseline_performance(self):
        """测试基准MADDPG性能"""
        self.logger.info("\n开始测试基准MADDPG性能...")
        self.train_and_evaluate(self.baseline_agent, "baseline")
        
        # 验证关键指标
        self.assertGreater(
            np.mean(self.metrics["baseline_avg_reward"]),
            -5000,
            "基准版本平均奖励过低"
        )
        self.assertGreater(
            np.mean(self.metrics["baseline_win_rate"]),
            0.2,
            "基准版本胜率过低"
        )
    
    def test_improved_performance(self):
        """测试改进版MADDPG性能"""
        self.logger.info("\n开始测试改进版MADDPG性能...")
        self.train_and_evaluate(self.improved_agent, "improved")
        
        # 验证关键指标
        self.assertGreater(
            np.mean(self.metrics["improved_avg_reward"]),
            np.mean(self.metrics["baseline_avg_reward"]),
            "改进版本未能提升平均奖励"
        )
        self.assertGreater(
            np.mean(self.metrics["improved_win_rate"]),
            np.mean(self.metrics["baseline_win_rate"]),
            "改进版本未能提升胜率"
        )
    
    def test_performance_comparison(self):
        """比较两个版本的综合性能"""
        self.logger.info("\n开始性能对比分析...")
        
        # 创建结果目录
        if not os.path.exists('results'):
            os.makedirs('results')
        
        # 绘制性能对比图
        self._plot_performance_comparison()
        
        # 保存详细的性能报告
        self._save_performance_report()
    
    def _plot_performance_comparison(self):
        """绘制性能对比图表"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        metrics_to_plot = [
            ('avg_reward', 'Average Reward'),
            ('win_rate', 'Win Rate'),
            ('avg_episode_length', 'Episode Length'),
            ('survival_rates_red', 'Red Team Survival Rate'),
            ('survival_rates_blue', 'Blue Team Survival Rate'),
            ('hit_rates_red', 'Red Team Hit Rate'),
            ('hit_rates_blue', 'Blue Team Hit Rate')
        ]
        
        for i, (metric, title) in enumerate(metrics_to_plot):
            for model in ['baseline', 'improved']:
                if f"{model}_{metric}" in self.metrics:
                    axes[i].plot(self.metrics[f"{model}_{metric}"], 
                               label=f"{model.capitalize()}")
            axes[i].set_title(title)
            axes[i].legend()
        
        # 训练时间对比
        times = [
            self.metrics.get("baseline_training_time", 0),
            self.metrics.get("improved_training_time", 0)
        ]
        axes[-1].bar(["Baseline MADDPG", "Improved MADDPG"], times)
        axes[-1].set_title("Total Training Time")
        axes[-1].set_ylabel("Time (seconds)")
        
        plt.tight_layout()
        plt.savefig("results/maddpg_comparison_plots.png")
        plt.close()
        
    def _save_performance_report(self):
        """保存详细的性能报告"""
        report = {
            "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "num_episodes": self.num_episodes,
                "eval_interval": self.eval_interval,
                "num_eval_episodes": self.num_eval_episodes,
                "num_red": self.num_red,
                "num_blue": self.num_blue
            },
            "metrics": dict(self.metrics)
        }
        
        with open('results/performance_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info("性能报告已保存到 results/performance_report.json")

if __name__ == '__main__':
    unittest.main(verbosity=2)
