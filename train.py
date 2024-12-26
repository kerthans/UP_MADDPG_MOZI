# trainer.py

import os
import time
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import seaborn as sns
from matplotlib.gridspec import GridSpec
from combat_sim.combat_env import CombatEnv
from agents.up import MADDPG
import logging
from typing import Dict, List, Optional, Tuple, Union
import multiprocessing as mp

class CombatTrainer:
    def __init__(
        self,
        exp_name: str,
        num_red: int = 2,
        num_blue: int = 3,
        num_envs: int = 8,
        total_episodes: int = 1000,
        save_interval: int = 100,
        eval_interval: int = 50,
        early_stop_patience: int = 20,
        early_stop_threshold: float = 0.01,
        base_path: str = "./results"
    ):
        """
        初始化训练器
        
        Args:
            exp_name: 实验名称
            num_red: 红方智能体数量
            num_blue: 蓝方智能体数量
            num_envs: 并行环境数量
            total_episodes: 总训练回合数
            save_interval: 模型保存间隔
            eval_interval: 评估间隔
            early_stop_patience: 早停耐心值
            early_stop_threshold: 早停阈值
            base_path: 基础保存路径
        """
        self.exp_name = exp_name
        self.num_red = num_red
        self.num_blue = num_blue
        self.num_envs = num_envs
        self.total_episodes = total_episodes
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.early_stop_patience = early_stop_patience
        self.early_stop_threshold = early_stop_threshold
        
        # 创建保存路径
        self.exp_path = Path(base_path) / exp_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化日志
        self._setup_logging()
        
        # 初始化训练数据记录器
        self.train_records = {
            'episode': [],
            'red_reward': [],
            'blue_reward': [],
            'red_hits': [],
            'blue_hits': [],
            'red_alive': [],
            'blue_alive': [],
            'episode_length': []
        }
        
        # 初始化评估数据记录器
        self.eval_records = {
            'episode': [],
            'red_reward': [],
            'blue_reward': [],
            'red_hits': [],
            'blue_hits': [],
            'red_alive': [],
            'blue_alive': [],
            'episode_length': []
        }
        
        # 初始化环境和智能体
        self._setup_env_and_agents()
        
        # 早停相关变量
        self.best_eval_reward = -np.inf
        self.patience_counter = 0
        
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.exp_path / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_env_and_agents(self):
        """设置环境和智能体"""
        # 创建多个训练环境
        self.train_envs = [
            CombatEnv(num_red=self.num_red, num_blue=self.num_blue)
            for _ in range(self.num_envs)
        ]
        
        # 创建评估环境
        self.eval_env = CombatEnv(num_red=self.num_red, num_blue=self.num_blue)
        
        # 获取环境信息
        obs_dim = self.train_envs[0].observation_space.shape[0]
        act_dim = self.train_envs[0].action_space.shape[0]
        
        # 创建MADDPG智能体
        self.maddpg = MADDPG(
            n_red=self.num_red,
            n_blue=self.num_blue,
            obs_dim=obs_dim,
            act_dim=act_dim
        )
        
    def _save_checkpoint(self, episode: int):
        """
        保存检查点
        
        Args:
            episode: 当前回合数
        """
        checkpoint_path = self.exp_path / f'checkpoint_ep_{episode}.pt'
        self.maddpg.save(checkpoint_path)
        self.logger.info(f'Saved checkpoint at episode {episode}')
        
        # 保存训练记录
        self._save_records()
        
    def _save_records(self):
        """保存训练和评估记录"""
        # 保存为CSV
        train_df = pd.DataFrame(self.train_records)
        eval_df = pd.DataFrame(self.eval_records)
        
        train_df.to_csv(self.exp_path / 'train_records.csv', index=False)
        eval_df.to_csv(self.exp_path / 'eval_records.csv', index=False)
        
        # 保存为JSON以保留更多信息
        with open(self.exp_path / 'train_records.json', 'w') as f:
            json.dump(self.train_records, f, indent=4)
        with open(self.exp_path / 'eval_records.json', 'w') as f:
            json.dump(self.eval_records, f, indent=4)
            
    def _update_records(self, records: dict, info: dict, rewards: dict, episode: int):
        """
        更新记录
        
        Args:
            records: 记录字典
            info: 环境信息
            rewards: 奖励信息
            episode: 当前回合数
        """
        records['episode'].append(episode)
        records['red_reward'].append(np.mean([r for aid, r in rewards.items() if 'red' in aid]))
        records['blue_reward'].append(np.mean([r for aid, r in rewards.items() if 'blue' in aid]))
        records['red_hits'].append(info['red_hits'])
        records['blue_hits'].append(info['blue_hits'])
        records['red_alive'].append(info['red_alive'])
        records['blue_alive'].append(info['blue_alive'])
        records['episode_length'].append(info.get('episode_length', 0))
        
    def evaluate(self, num_episodes: int = 5) -> float:
        """
        评估当前策略
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            平均红方奖励
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            obs = self.eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                actions = self.maddpg.select_actions(obs, noise_scale=0.0)  # 评估时不使用噪声
                next_obs, rewards, done, info = self.eval_env.step(actions)
                episode_reward += np.mean([r for aid, r in rewards.items() if 'red' in aid])
                obs = next_obs
                
                if done:
                    self._update_records(self.eval_records, info, rewards, len(self.eval_records['episode']))
                    
            eval_rewards.append(episode_reward)
            
        mean_reward = np.mean(eval_rewards)
        self.logger.info(f'Evaluation: Mean reward = {mean_reward:.4f}')
        return mean_reward
        
    def _check_early_stop(self, eval_reward: float) -> bool:
        """
        检查是否需要早停
        
        Args:
            eval_reward: 评估奖励
            
        Returns:
            是否应该早停
        """
        if eval_reward > self.best_eval_reward + self.early_stop_threshold:
            self.best_eval_reward = eval_reward
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stop_patience:
                self.logger.info(f'Early stopping triggered after {self.patience_counter} evaluations without improvement')
                return True
        return False
        
    def train(self):
        """训练过程"""
        try:
            self.logger.info(f'Starting training experiment: {self.exp_name}')
            
            # 训练进度条
            pbar = tqdm(range(self.total_episodes), desc='Training')
            
            for episode in pbar:
                # 并行收集数据
                episode_data = []
                for env in self.train_envs:
                    obs = env.reset()
                    done = False
                    episode_reward = 0
                    
                    while not done:
                        actions = self.maddpg.select_actions(obs)
                        next_obs, rewards, done, info = env.step(actions)
                        
                        # 存储转移
                        self.maddpg.store_transition(obs, actions, rewards, next_obs, done)
                        
                        episode_reward += np.mean([r for aid, r in rewards.items() if 'red' in aid])
                        obs = next_obs
                        
                        if done:
                            episode_data.append((info, rewards))
                            
                # 更新网络
                self.maddpg.train()
                
                # 更新记录
                for info, rewards in episode_data:
                    self._update_records(self.train_records, info, rewards, episode)
                
                # 更新进度条信息
                pbar.set_postfix({
                    'red_reward': self.train_records['red_reward'][-1],
                    'blue_reward': self.train_records['blue_reward'][-1]
                })
                
                # 定期评估
                if episode % self.eval_interval == 0:
                    eval_reward = self.evaluate()
                    
                    # 检查早停
                    if self._check_early_stop(eval_reward):
                        break
                        
                # 定期保存
                if episode % self.save_interval == 0:
                    self._save_checkpoint(episode)
                    
            # 训练结束，保存最终模型和记录
            self._save_checkpoint(episode)
            self._generate_report()
            self.logger.info('Training completed successfully')
            
        except Exception as e:
            self.logger.error(f'Training interrupted: {str(e)}')
            self._save_checkpoint('interrupted')
            self._generate_report()
            raise
            
    def _generate_report(self):
        """生成综合评估报告"""
        report_path = self.exp_path / 'evaluation_report.md'
        fig_path = self.exp_path / 'training_curves.png'
        
        # 生成训练曲线
        self._plot_training_curves(fig_path)
        
        # 计算统计数据
        train_stats = {
            'mean_red_reward': np.mean(self.train_records['red_reward']),
            'mean_blue_reward': np.mean(self.train_records['blue_reward']),
            'mean_red_hits': np.mean(self.train_records['red_hits']),
            'mean_blue_hits': np.mean(self.train_records['blue_hits']),
            'mean_red_alive': np.mean(self.train_records['red_alive']),
            'mean_blue_alive': np.mean(self.train_records['blue_alive']),
            'mean_episode_length': np.mean(self.train_records['episode_length'])
        }
        
        eval_stats = {
            'mean_red_reward': np.mean(self.eval_records['red_reward']),
            'mean_blue_reward': np.mean(self.eval_records['blue_reward']),
            'mean_red_hits': np.mean(self.eval_records['red_hits']),
            'mean_blue_hits': np.mean(self.eval_records['blue_hits']),
            'mean_red_alive': np.mean(self.eval_records['red_alive']),
            'mean_blue_alive': np.mean(self.eval_records['blue_alive']),
            'mean_episode_length': np.mean(self.eval_records['episode_length'])
        }
        
        # 生成报告内容
        report_content = f"""# 训练评估报告

## 实验信息
- 实验名称: {self.exp_name}
- 训练时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- 红方智能体数量: {self.num_red}
- 蓝方智能体数量: {self.num_blue}
- 并行环境数量: {self.num_envs}
- 总训练回合数: {len(self.train_records['episode'])}

## 训练统计
- 平均红方奖励: {train_stats['mean_red_reward']:.4f}
- 平均蓝方奖励: {train_stats['mean_blue_reward']:.4f}
- 平均红方击中数: {train_stats['mean_red_hits']:.2f}
- 平均蓝方击中数: {train_stats['mean_blue_hits']:.2f}
- 平均红方存活数: {train_stats['mean_red_alive']:.2f}
- 平均蓝方存活数: {train_stats['mean_blue_alive']:.2f}
- 平均回合长度: {train_stats['mean_episode_length']:.2f}

## 评估统计
- 平均红方奖励: {eval_stats['mean_red_reward']:.4f}
- 平均蓝方奖励: {eval_stats['mean_blue_reward']:.4f}
- 平均红方击中数: {eval_stats['mean_red_hits']:.2f}
- 平均蓝方击中数: {eval_stats['mean_blue_hits']:.2f}
- 平均红方存活数: {eval_stats['mean_red_alive']:.2f}
- 平均蓝方存活数: {eval_stats['mean_blue_alive']:.2f}
- 平均回合长度: {eval_stats['mean_episode_length']:.2f}

## 训练过程分析
1. 收敛性分析:
   - 训练过程{' ' if self.patience_counter >= self.early_stop_patience else '未'}触发早停机制
   - 最佳评估奖励: {self.best_eval_reward:.4f}
   - 耐心计数器: {self.patience_counter}

2. 对抗效果分析:
   - 红蓝双方的平均奖励差: {abs(train_stats['mean_red_reward'] - train_stats['mean_blue_reward']):.4f}
   - 红蓝双方的平均击中差: {abs(train_stats['mean_red_hits'] - train_stats['mean_blue_hits']):.2f}
   - 红蓝双方的平均存活差: {abs(train_stats['mean_red_alive'] - train_stats['mean_blue_alive']):.2f}

## 结论与建议
1. 训练效果总结:
   {'- 训练提前在第' + str(len(self.train_records['episode'])) + '回合停止' if self.patience_counter >= self.early_stop_patience else '- 完成了预定的训练回合数'}
   - 模型表现{'稳定' if abs(train_stats['mean_red_reward'] - train_stats['mean_blue_reward']) < 1.0 else '不稳定'}
   - 红蓝双方{'达到了较好的平衡' if abs(train_stats['mean_red_hits'] - train_stats['mean_blue_hits']) < 0.5 else '存在明显差距'}

2. 改进建议:
   - {'建议增加训练回合数' if len(self.train_records['episode']) < self.total_episodes * 0.8 else '训练回合数适中'}
   - {'建议调整奖励权重以平衡红蓝双方' if abs(train_stats['mean_red_reward'] - train_stats['mean_blue_reward']) > 1.0 else '奖励设计合理'}
   - {'可以考虑增加并行环境数量以加速训练' if self.num_envs < 8 else '并行环境数量充足'}
"""
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
    def _plot_training_curves(self, save_path: str):
        """生成训练曲线可视化图表"""
        # 使用默认样式而不是seaborn，提高兼容性
        plt.rcParams.update(plt.rcParamsDefault)
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 2, figure=fig)
        
        # 设置全局字体和样式
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7
        
        # 1. 奖励曲线
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.train_records['episode'], self.train_records['red_reward'], 
                'r-', label='Red Team (Train)', alpha=0.6)
        ax1.plot(self.train_records['episode'], self.train_records['blue_reward'], 
                'b-', label='Blue Team (Train)', alpha=0.6)
        
        if self.eval_records['episode']:  # 只在有评估数据时绘制
            ax1.plot(self.eval_records['episode'], self.eval_records['red_reward'], 
                    'r--', label='Red Team (Eval)', alpha=0.8)
            ax1.plot(self.eval_records['episode'], self.eval_records['blue_reward'], 
                    'b--', label='Blue Team (Eval)', alpha=0.8)
        
        ax1.set_title('Average Rewards per Episode', fontsize=12)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        
        # 2. 击中数曲线
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(self.train_records['episode'], self.train_records['red_hits'], 
                'r-', label='Red Hits (Train)', alpha=0.6)
        ax2.plot(self.train_records['episode'], self.train_records['blue_hits'], 
                'b-', label='Blue Hits (Train)', alpha=0.6)
        
        if self.eval_records['episode']:
            ax2.plot(self.eval_records['episode'], self.eval_records['red_hits'], 
                    'r--', label='Red Hits (Eval)', alpha=0.8)
            ax2.plot(self.eval_records['episode'], self.eval_records['blue_hits'], 
                    'b--', label='Blue Hits (Eval)', alpha=0.8)
        
        ax2.set_title('Hits per Episode', fontsize=12)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Number of Hits')
        ax2.legend()
        
        # 3. 存活数曲线
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(self.train_records['episode'], self.train_records['red_alive'], 
                'r-', label='Red Alive (Train)', alpha=0.6)
        ax3.plot(self.train_records['episode'], self.train_records['blue_alive'], 
                'b-', label='Blue Alive (Train)', alpha=0.6)
        
        if self.eval_records['episode']:
            ax3.plot(self.eval_records['episode'], self.eval_records['red_alive'], 
                    'r--', label='Red Alive (Eval)', alpha=0.8)
            ax3.plot(self.eval_records['episode'], self.eval_records['blue_alive'], 
                    'b--', label='Blue Alive (Eval)', alpha=0.8)
        
        ax3.set_title('Survival Rate per Episode', fontsize=12)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Number of Survivors')
        ax3.legend()
        
        # 4. 回合长度分布 - 使用更简单的方式绘制，避免seaborn依赖
        ax4 = fig.add_subplot(gs[2, :])
        train_data = self.train_records['episode_length']
        eval_data = self.eval_records['episode_length']
        
        # 使用普通的箱型图替代seaborn
        if train_data:  # 确保有训练数据
            bp1 = ax4.boxplot([train_data], positions=[0], labels=['Training'],
                            patch_artist=True)
            plt.setp(bp1['boxes'], facecolor='lightblue')
            
        if eval_data:  # 确保有评估数据
            bp2 = ax4.boxplot([eval_data], positions=[1], labels=['Evaluation'],
                            patch_artist=True)
            plt.setp(bp2['boxes'], facecolor='lightgreen')
        
        ax4.set_title('Episode Length Distribution', fontsize=12)
        ax4.set_ylabel('Steps')
        
        # 调整布局并保存
        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except Exception as e:
            self.logger.error(f"Error saving plot: {str(e)}")
        finally:
            plt.close()
def train_and_compare(
    base_config: dict,
    variant_configs: List[dict],
    base_name: str = "baseline",
    num_runs: int = 3,
    base_path: str = "./results/comparison"
):
    """
    训练并比较多个配置
    
    Args:
        base_config: 基准配置
        variant_configs: 变体配置列表
        base_name: 基准名称
        num_runs: 每个配置的运行次数
        base_path: 基础保存路径
    """
    all_results = {}
    
    # 训练基准配置
    base_results = []
    for run in range(num_runs):
        trainer = CombatTrainer(
            exp_name=f"{base_name}_run_{run}",
            base_path=base_path,
            **base_config
        )
        trainer.train()
        base_results.append(trainer.eval_records)
    all_results[base_name] = base_results
    
    # 训练变体配置
    for idx, config in enumerate(variant_configs):
        variant_name = f"variant_{idx}"
        variant_results = []
        for run in range(num_runs):
            trainer = CombatTrainer(
                exp_name=f"{variant_name}_run_{run}",
                base_path=base_path,
                **config
            )
            trainer.train()
            variant_results.append(trainer.eval_records)
        all_results[variant_name] = variant_results
        
    # 生成对比报告
    _generate_comparison_report(all_results, base_path)

def _generate_comparison_report(results: Dict[str, List[dict]], base_path: str):
    """
    生成配置对比报告
    
    Args:
        results: 所有配置的训练结果
        base_path: 保存路径
    """
    report_path = Path(base_path) / 'comparison_report'
    report_path.mkdir(parents=True, exist_ok=True)
    
    # 生成对比图表
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. 奖励对比
    ax1 = fig.add_subplot(gs[0, 0])
    for name, runs in results.items():
        rewards = [np.mean(run['red_reward']) for run in runs]
        ax1.boxplot(rewards, positions=[list(results.keys()).index(name)], 
                   labels=[name], widths=0.7)
    ax1.set_title('Red Team Reward Distribution', fontsize=12)
    ax1.set_ylabel('Average Reward')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 击中数对比
    ax2 = fig.add_subplot(gs[0, 1])
    for name, runs in results.items():
        hits = [np.mean(run['red_hits']) for run in runs]
        ax2.boxplot(hits, positions=[list(results.keys()).index(name)], 
                   labels=[name], widths=0.7)
    ax2.set_title('Red Team Hits Distribution', fontsize=12)
    ax2.set_ylabel('Average Hits')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 存活率对比
    ax3 = fig.add_subplot(gs[1, 0])
    for name, runs in results.items():
        alive = [np.mean(run['red_alive']) for run in runs]
        ax3.boxplot(alive, positions=[list(results.keys()).index(name)], 
                   labels=[name], widths=0.7)
    ax3.set_title('Red Team Survival Rate Distribution', fontsize=12)
    ax3.set_ylabel('Average Survivors')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # 4. 回合长度对比
    ax4 = fig.add_subplot(gs[1, 1])
    for name, runs in results.items():
        lengths = [np.mean(run['episode_length']) for run in runs]
        ax4.boxplot(lengths, positions=[list(results.keys()).index(name)], 
                   labels=[name], widths=0.7)
    ax4.set_title('Episode Length Distribution', fontsize=12)
    ax4.set_ylabel('Average Steps')
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(report_path / 'comparison_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成统计报告
    stats = {}
    for name, runs in results.items():
        stats[name] = {
            'red_reward': {
                'mean': np.mean([np.mean(run['red_reward']) for run in runs]),
                'std': np.std([np.mean(run['red_reward']) for run in runs])
            },
            'red_hits': {
                'mean': np.mean([np.mean(run['red_hits']) for run in runs]),
                'std': np.std([np.mean(run['red_hits']) for run in runs])
            },
            'red_alive': {
                'mean': np.mean([np.mean(run['red_alive']) for run in runs]),
                'std': np.std([np.mean(run['red_alive']) for run in runs])
            },
            'episode_length': {
                'mean': np.mean([np.mean(run['episode_length']) for run in runs]),
                'std': np.std([np.mean(run['episode_length']) for run in runs])
            }
        }
    
    # 生成报告内容
    report_content = "# 配置对比报告\n\n"
    
    for name, stat in stats.items():
        report_content += f"## {name}\n"
        report_content += f"- 红方平均奖励: {stat['red_reward']['mean']:.4f} ± {stat['red_reward']['std']:.4f}\n"
        report_content += f"- 红方平均击中数: {stat['red_hits']['mean']:.2f} ± {stat['red_hits']['std']:.2f}\n"
        report_content += f"- 红方平均存活数: {stat['red_alive']['mean']:.2f} ± {stat['red_alive']['std']:.2f}\n"
        report_content += f"- 平均回合长度: {stat['episode_length']['mean']:.2f} ± {stat['episode_length']['std']:.2f}\n\n"
    
    # 保存报告
    with open(report_path / 'comparison_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)

if __name__ == "__main__":
    # 示例用法
    trainer = CombatTrainer(
        exp_name="test_experiment",
        num_red=2,
        num_blue=3,
        num_envs=8,
        total_episodes=1000
    )
    trainer.train()