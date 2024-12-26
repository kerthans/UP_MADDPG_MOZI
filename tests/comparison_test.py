# tests/comparison_test.py

import random
import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import time
import os
import signal
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm
import logging
from datetime import datetime

from agents.baseline import MADDPG as BaselineMADDPG
from agents.up import MADDPG as ImprovedMADDPG
from combat_sim.combat_env import CombatEnv


class TestMADDPGComparison(unittest.TestCase):
    """比较基准MADDPG和改进版MADDPG的性能测试"""

    @classmethod
    def setUpClass(cls):
        """设置日志记录和初始化"""
        # 创建logs和results目录
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results', exist_ok=True)

        # 设置日志文件名
        log_filename = f'logs/maddpg_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        # 配置日志记录
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        cls.logger = logging.getLogger('ComparisonTest')
        cls.logger.info("初始化 ComparisonTest 类")

        # 初始化性能指标存储
        cls.metrics = defaultdict(list)

        # 标记是否中断
        cls.interrupted = False

        # 设置中断信号处理
        def signal_handler(sig, frame):
            cls.logger.info("接收到中断信号，保存当前进度...")
            cls.interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @classmethod
    def tearDownClass(cls):
        """保存最终报告"""
        if cls.interrupted:
            cls.logger.info("测试被中断，生成部分报告...")

        cls.logger.info("生成最终对比报告")
        report = {
            'baseline': {
                'rewards': cls.metrics.get('baseline_avg_reward', []),
                'steps': cls.metrics.get('baseline_avg_steps', []),
                'win_rate': cls.metrics.get('baseline_win_rate', [])
            },
            'improved': {
                'rewards': cls.metrics.get('improved_avg_reward', []),
                'steps': cls.metrics.get('improved_avg_steps', []),
                'win_rate': cls.metrics.get('improved_win_rate', [])
            }
        }

        # 保存报告为 npz 文件
        report_path = os.path.join('results', 'comparison_report.npz')
        np.savez(report_path, **report)
        cls.logger.info(f'Comparison report saved to {report_path}')

        # 绘制对比图
        cls.logger.info("绘制对比图")
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        if 'baseline_avg_reward' in cls.metrics and 'improved_avg_reward' in cls.metrics:
            plt.plot(cls.metrics['baseline_avg_reward'], label='Baseline')
            plt.plot(cls.metrics['improved_avg_reward'], label='Improved')
            plt.xlabel('Evaluation Interval')
            plt.ylabel('Average Reward')
            plt.legend()
            plt.title('Average Reward per Evaluation Interval')

        plt.subplot(2, 1, 2)
        if 'baseline_avg_steps' in cls.metrics and 'improved_avg_steps' in cls.metrics:
            plt.plot(cls.metrics['baseline_avg_steps'], label='Baseline')
            plt.plot(cls.metrics['improved_avg_steps'], label='Improved')
            plt.xlabel('Evaluation Interval')
            plt.ylabel('Average Steps')
            plt.legend()
            plt.title('Average Steps per Evaluation Interval')

        plt.tight_layout()
        plot_path = os.path.join('results', 'comparison_plot.png')
        plt.savefig(plot_path)
        plt.close()
        cls.logger.info(f'Comparison plot saved to {plot_path}')

        # 保存详细的性能报告
        report_json_path = os.path.join('results', 'performance_report.json')
        with open(report_json_path, 'w') as f:
            json.dump(report, f, indent=2)
        cls.logger.info(f"性能报告已保存到 {report_json_path}")

    def setUp(self):
        """测试初始化"""
        self.logger = self.__class__.logger
        self.metrics = self.__class__.metrics
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
            num_blue=self.num_blue,
            max_steps=200,
            field_size=1000.0,
            attack_range=100.0,
            min_speed=10.0,
            max_speed=30.0,
            max_turn_rate=np.pi / 6,
            hit_probability=0.8,
            num_threads=8
        )

        # 从环境中获取真实的观察和动作空间维度
        obs = self.env.reset()
        # 假设每个智能体的观察是字典格式，取第一个智能体的观察维度
        first_obs_key = next(iter(obs))
        self.obs_dim = len(obs[first_obs_key])
        self.act_dim = self.env.action_space.shape[0]

        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # 初始化智能体
        self.baseline_agent = BaselineMADDPG(
            n_red=self.num_red,
            n_blue=self.num_blue,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
        )

        self.improved_agent = ImprovedMADDPG(
            n_red=self.num_red,
            n_blue=self.num_blue,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
        )

        self.logger.info(f"初始化完成. 观察空间维度: {self.obs_dim}, 动作空间维度: {self.act_dim}")

    def train_agent(self, maddpg, env, num_episodes, max_steps, model_type="baseline"):
        """训练单个智能体"""
        episode_rewards = []
        episode_steps = []
        episode_wins = 0
        metrics = defaultdict(list)
        
        # 初始化进度条
        progress = tqdm(range(num_episodes), desc=f'Training {model_type} MADDPG')
        
        for episode in progress:
            obs = env.reset()
            episode_reward = 0
            episode_td_errors = []  # 用于记录TD误差
            
            # 重置所有智能体的噪声
            if hasattr(maddpg, 'agents'):
                for agent in maddpg.agents:
                    if hasattr(agent, 'reset_noise'):
                        agent.reset_noise()
            
            for step in range(max_steps):
                # 根据不同模型类型选择动作
                if model_type == "baseline":
                    actions = maddpg.select_actions(obs)
                else:
                    actions = maddpg.select_actions(obs, noise_scale=max(0.1, 1.0 - episode/num_episodes))
                
                next_obs, rewards, done, info = env.step(actions)
                
                # 存储经验
                maddpg.store_transition(obs, actions, rewards, next_obs, done)
                
                # 训练并获取额外指标
                if model_type == "improved":
                    maddpg.train()
                    if hasattr(maddpg, 'memory') and hasattr(maddpg.memory, 'priorities'):
                        metrics['priorities'].append(np.mean(maddpg.memory.priorities))
                else:
                    maddpg.train()
                
                episode_reward += np.mean(list(rewards.values()))
                obs = next_obs
                
                if done:
                    if info.get('blue_alive', 0) == 0:
                        episode_wins += 1
                    break
            
            # 记录每个episode的指标
            episode_rewards.append(episode_reward)
            episode_steps.append(step + 1)
            
            # 更新进度条信息
            progress.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'win_rate': f'{episode_wins/(episode+1):.2%}'
            })
            
            # 记录详细指标
            metrics['red_alive'].append(info.get('red_alive', 0))
            metrics['blue_alive'].append(info.get('blue_alive', 0))
            metrics['red_hits'].append(info.get('red_hits', 0))
            metrics['blue_hits'].append(info.get('blue_hits', 0))
            
            # 检查是否被中断
            if self.__class__.interrupted:
                self.logger.info(f"{model_type} training interrupted at episode {episode}")
                break
        
        # 计算总体统计
        stats = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_steps': np.mean(episode_steps),
            'win_rate': episode_wins / len(episode_rewards),
            'avg_red_alive': np.mean(metrics['red_alive']),
            'avg_blue_alive': np.mean(metrics['blue_alive']),
            'avg_red_hits': np.mean(metrics['red_hits']),
            'avg_blue_hits': np.mean(metrics['blue_hits'])
        }
        
        if model_type == "improved":
            stats['avg_priority'] = np.mean(metrics['priorities']) if metrics['priorities'] else 0
        
        self.logger.info(f"{model_type.capitalize()} training completed: {json.dumps(stats, indent=2)}")
        return stats

    def evaluate_agent(self, agent, env, num_eval_episodes, model_type="baseline"):
        """评估智能体性能"""
        self.logger.info(f"开始评估 {model_type} 智能体...")
        total_rewards = []
        win_rate = 0
        avg_episode_length = 0
        survival_rates = {"red": 0.0, "blue": 0.0}
        hit_rates = {"red": 0.0, "blue": 0.0}

        for ep in tqdm(range(num_eval_episodes), desc=f"评估 {model_type} MADDPG", unit='episode'):
            obs = env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            hits = {"red": 0, "blue": 0}

            while not done:
                if model_type == "baseline":
                    actions = agent.select_actions(obs, noise_scale=0.0)
                else:
                    actions = agent.select_actions(obs, noise_scale=0.0)  # 假设改进版也使用确定性策略

                next_obs, rewards, done, info = env.step(actions)
                episode_reward += sum(rewards.values())
                hits["red"] += info.get("red_hits", 0)
                hits["blue"] += info.get("blue_hits", 0)

                obs = next_obs
                step_count += 1

            total_rewards.append(episode_reward)
            avg_episode_length += step_count

            if info.get("blue_alive", 0) == 0:
                win_rate += 1

            survival_rates["red"] += info.get("red_alive", 0) / self.num_red
            survival_rates["blue"] += info.get("blue_alive", 0) / self.num_blue

            total_shots = hits["red"] + hits["blue"]
            if total_shots > 0:
                hit_rates["red"] += hits["red"] / total_shots
                hit_rates["blue"] += hits["blue"] / total_shots

            # 检查是否被中断
            if self.__class__.interrupted:
                self.logger.info("检测到中断信号，停止评估...")
                break

        results = {
            "avg_reward": np.mean(total_rewards) if total_rewards else 0,
            "std_reward": np.std(total_rewards) if total_rewards else 0,
            "win_rate": win_rate / len(total_rewards) if total_rewards else 0,
            "avg_episode_length": avg_episode_length / len(total_rewards) if total_rewards else 0,
            "survival_rates": {k: v / len(total_rewards) if total_rewards else 0 for k, v in survival_rates.items()},
            "hit_rates": {k: v / len(total_rewards) if total_rewards else 0 for k, v in hit_rates.items()}
        }

        self.logger.info(f"{model_type.capitalize()} MADDPG 评估结果: {json.dumps(results, indent=2)}")
        return results

    def test_maddpg_comparison(self):
        """比较两个版本的MADDPG性能"""
        num_phases = 5
        episodes_per_phase = 100
        results = defaultdict(list)
        
        self.logger.info("\n========== 开始MADDPG性能对比测试 ==========")
        
        try:
            for phase in range(num_phases):
                self.logger.info(f"\n---------- 阶段 {phase + 1}/{num_phases} ----------")
                
                # Baseline训练和评估
                baseline_train = self.train_agent(
                    self.baseline_agent, self.env, episodes_per_phase, 
                    self.env.max_steps, "baseline"
                )
                baseline_eval = self.evaluate_agent(
                    self.baseline_agent, self.env, self.num_eval_episodes, "baseline"
                )
                
                # 改进版训练和评估
                improved_train = self.train_agent(
                    self.improved_agent, self.env, episodes_per_phase, 
                    self.env.max_steps, "improved"
                )
                improved_eval = self.evaluate_agent(
                    self.improved_agent, self.env, self.num_eval_episodes, "improved"
                )
                
                # 记录结果
                results['baseline_train'].append(baseline_train)
                results['baseline_eval'].append(baseline_eval)
                results['improved_train'].append(improved_train)
                results['improved_eval'].append(improved_eval)
                
                # 生成阶段性报告
                phase_report = {
                    'phase': phase + 1,
                    'baseline': {
                        'train': baseline_train,
                        'eval': baseline_eval
                    },
                    'improved': {
                        'train': improved_train,
                        'eval': improved_eval
                    }
                }
                
                # 保存阶段性报告
                report_path = os.path.join('results', f'phase_{phase+1}_report.json')
                with open(report_path, 'w') as f:
                    json.dump(phase_report, f, indent=2)
                
                # 检查是否需要提前停止
                if self.__class__.interrupted:
                    break
            
            # 生成最终对比报告
            final_report = self._generate_final_report(results)
            
            # 保存最终报告
            final_report_path = os.path.join(
                'results', 
                f'final_comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            with open(final_report_path, 'w') as f:
                json.dump(final_report, f, indent=2)
            
            self.logger.info(f"最终报告已保存至: {final_report_path}")
            
        except Exception as e:
            self.logger.error(f"测试过程中发生错误: {str(e)}")
            raise


if __name__ == '__main__':
    unittest.main(verbosity=2)
