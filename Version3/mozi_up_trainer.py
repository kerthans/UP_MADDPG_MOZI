# mozi_up_trainer.py

import os
import sys
import numpy as np
import torch
import time
import json
from datetime import datetime
from typing import Dict, List, Union, Any
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt
from envs.mozi_adapter import MoziAdapter
from envs.env_config import EnvConfig
from agents.up import MADDPG

class MoziEnhancedTrainer:
    """增强版墨子MADDPG训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 训练配置参数
        """
        self.config = config
        
        # 环境初始化
        if not os.environ.get("MOZIPATH"):
            os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
        print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")
        
        self.env = MoziAdapter(
            num_red=config['env_config']['num_red'],
            num_blue=config['env_config']['num_blue'],
            max_steps=config['env_config']['max_steps'],
            env_config=EnvConfig
        )
        
        # 获取观察和动作空间维度
        obs = self.env.reset()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
        # 初始化优化版MADDPG
        self.maddpg = MADDPG(
            n_red=config['env_config']['num_red'],
            n_blue=config['env_config']['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            **config['model_config']
        )
        
        # 创建保存目录
        self.save_dir = os.path.join(
            config['save_dir'],
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(self._convert_config_to_serializable(config), f, indent=4)
        
        # 初始化训练指标
        self.metrics = {
            'episode_rewards': deque(maxlen=10000),
            'red_rewards': deque(maxlen=10000),
            'blue_rewards': deque(maxlen=10000),
            'red_win_rates': deque(maxlen=10000),
            'red_survival_rates': deque(maxlen=10000),
            'blue_survival_rates': deque(maxlen=10000),
            'episode_lengths': deque(maxlen=10000),
            'noise_values': deque(maxlen=10000),
            'actor_losses': deque(maxlen=10000),
            'critic_losses': deque(maxlen=10000),
            'td_errors': deque(maxlen=10000),
            'hit_rates': deque(maxlen=10000),
            'red_casualties': deque(maxlen=10000),
            'blue_casualties': deque(maxlen=10000)
        }
        
        # 训练控制参数
        self.initial_noise = config['training']['initial_noise']
        self.noise = self.initial_noise
        self.noise_decay = config['training']['noise_decay']
        self.min_noise = config['training']['min_noise']
        self.best_reward = float('-inf')
        self.patience = config['training'].get('patience', 20)
        self.patience_counter = 0
        self.min_episodes = config['training'].get('min_episodes', 1000)
        
        # 训练日志配置
        self.log_interval = config['training'].get('log_interval', 1)
        self.save_interval = config['training'].get('save_interval', 100)
        self.eval_interval = config['training'].get('eval_interval', 100)
        self.plot_interval = config['training'].get('plot_interval', 100)
        
        # 环境重置控制
        self.env_reset_counter = 0
        self.env_reset_frequency = config['training'].get('env_reset_frequency', 1000)

    def train(self):
        """执行训练主循环"""
        print("\n开始增强版MADDPG训练...")
        try:
            progress_bar = tqdm(range(self.config['training']['n_episodes']))
            
            for episode in progress_bar:
                # 定期重置环境
                if self.env_reset_counter >= self.env_reset_frequency:
                    self.env = MoziAdapter(
                        num_red=self.config['env_config']['num_red'],
                        num_blue=self.config['env_config']['num_blue'],
                        max_steps=self.config['env_config']['max_steps'],
                        env_config=EnvConfig
                    )
                    self.env_reset_counter = 0
                
                # 训练单个回合
                episode_stats = self._train_episode()
                self.env_reset_counter += 1
                
                # 更新指标
                self._update_metrics(episode_stats)
                
                # 更新进度条
                progress_bar.set_postfix({
                    'reward': f"{episode_stats['total_reward']:.1f}",
                    'red_wins': f"{np.mean(list(self.metrics['red_win_rates'])[-100:]):.2%}",
                    'steps': episode_stats['episode_length']
                })
                
                # 定期评估
                if (episode + 1) % self.eval_interval == 0:
                    eval_metrics = self._evaluate()
                    self._log_eval_metrics(eval_metrics, episode + 1)
                    
                    # 检查早停
                    if self._check_early_stopping(eval_metrics['mean_reward']):
                        print("\n触发早停条件！")
                        break
                
                # 定期保存和绘图
                if (episode + 1) % self.save_interval == 0:
                    self._save_checkpoint(episode + 1)
                
                if (episode + 1) % self.plot_interval == 0:
                    self._plot_metrics()
                    self._print_current_stats()
                
                # 动态调整噪声
                self._adjust_noise(episode)
                
        except KeyboardInterrupt:
            print("\n训练被中断，正在保存...")
        
        finally:
            # 保存最终结果
            self._save_checkpoint('final')
            self._plot_metrics()
            self._save_final_report()
            print(f"\n训练结束! 结果已保存至: {self.save_dir}")

    def _train_episode(self) -> Dict[str, float]:
        """训练单个回合
        
        Returns:
            Dict: 回合统计信息
        """
        obs = self.env.reset()
        episode_stats = {
            'total_reward': 0,
            'red_reward': 0,
            'blue_reward': 0,
            'step_rewards': [],
            'red_alive': [],
            'blue_alive': [],
            'actions_taken': [],
            'training_steps': 0,
            'episode_length': 0,
            'hits': 0,
            'shots': 0
        }
        
        red_alive_start = self.config['env_config']['num_red']
        blue_alive_start = self.config['env_config']['num_blue']
        
        for step in range(self.config['env_config']['max_steps']):
            # 选择动作
            actions = self.maddpg.select_actions(obs, self.noise)
            episode_stats['actions_taken'].append(actions)
            
            # 执行动作
            next_obs, rewards, done, info = self.env.step(actions)
            
            # 记录射击统计
            for agent_id, action in actions.items():
                if agent_id.startswith('red_') and isinstance(action, np.ndarray) and action[-1] > 0.5:
                    episode_stats['shots'] += 1
                    if info.get('hit_' + agent_id, False):
                        episode_stats['hits'] += 1
            
            # 更新奖励统计
            step_reward = sum(rewards.values() if rewards else [0])
            episode_stats['total_reward'] += step_reward
            episode_stats['red_reward'] += sum(v for k, v in rewards.items() if k.startswith('red_'))
            episode_stats['blue_reward'] += sum(v for k, v in rewards.items() if k.startswith('blue_'))
            episode_stats['step_rewards'].append(step_reward)
            
            # 更新存活统计
            episode_stats['red_alive'].append(info['red_alive'])
            episode_stats['blue_alive'].append(info['blue_alive'])
            
            # 存储经验
            self.maddpg.store_transition(obs, actions, rewards, next_obs, done)
            
            # 训练网络
            if len(self.maddpg.memory) >= self.maddpg.batch_size:
                self.maddpg.train()
                episode_stats['training_steps'] += 1
            
            # 更新观察
            obs = next_obs
            
            if done:
                break
        
        # 计算最终统计
        episode_stats['episode_length'] = step + 1
        episode_stats['hit_rate'] = episode_stats['hits'] / max(1, episode_stats['shots'])
        episode_stats['red_casualties'] = red_alive_start - info['red_alive']
        episode_stats['blue_casualties'] = blue_alive_start - info['blue_alive']
        episode_stats['red_win'] = info['blue_alive'] == 0
        episode_stats['red_survival_rate'] = info['red_alive'] / red_alive_start
        episode_stats['blue_survival_rate'] = info['blue_alive'] / blue_alive_start
        
        return episode_stats

    def _evaluate(self, num_episodes=10) -> Dict[str, float]:
        """评估当前策略
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            Dict: 评估指标
        """
        eval_stats = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            red_wins = 0
            step = 0
            
            while True:
                actions = self.maddpg.select_actions(obs, 0)  # 评估时不使用噪声
                next_obs, rewards, done, info = self.env.step(actions)
                
                episode_reward += sum(rewards.values())
                obs = next_obs
                step += 1
                
                if done:
                    red_wins += int(info['blue_alive'] == 0)
                    eval_stats.append({
                        'reward': episode_reward,
                        'steps': step,
                        'red_win': red_wins,
                        'red_alive': info['red_alive'],
                        'blue_alive': info['blue_alive']
                    })
                    break
        
        return {
            'mean_reward': np.mean([s['reward'] for s in eval_stats]),
            'std_reward': np.std([s['reward'] for s in eval_stats]),
            'mean_steps': np.mean([s['steps'] for s in eval_stats]),
            'mean_red_win_rate': np.mean([s['red_win'] for s in eval_stats]),
            'mean_red_alive': np.mean([s['red_alive'] for s in eval_stats]),
            'mean_blue_alive': np.mean([s['blue_alive'] for s in eval_stats])
        }

    def _update_metrics(self, episode_stats: Dict[str, Any]):
        """更新训练指标
        
        Args:
            episode_stats: 回合统计信息
        """
        self.metrics['episode_rewards'].append(episode_stats['total_reward'])
        self.metrics['red_rewards'].append(episode_stats['red_reward'])
        self.metrics['blue_rewards'].append(episode_stats['blue_reward'])
        self.metrics['red_win_rates'].append(float(episode_stats['red_win']))
        self.metrics['red_survival_rates'].append(episode_stats['red_survival_rate'])
        self.metrics['blue_survival_rates'].append(episode_stats['blue_survival_rate'])
        self.metrics['episode_lengths'].append(episode_stats['episode_length'])
        self.metrics['noise_values'].append(self.noise)
        self.metrics['hit_rates'].append(episode_stats['hit_rate'])
        self.metrics['red_casualties'].append(episode_stats['red_casualties'])
        self.metrics['blue_casualties'].append(episode_stats['blue_casualties'])

    def _plot_metrics(self):
        """绘制训练指标图表"""
        plt.figure(figsize=(20, 15))
        
        # 奖励曲线
        plt.subplot(3, 2, 1)
        self._plot_smoothed_curve(self.metrics['episode_rewards'], 'Total Reward', color='blue')
        self._plot_smoothed_curve(self.metrics['red_rewards'], 'Red Reward', color='red')
        self._plot_smoothed_curve(self.metrics['blue_rewards'], 'Blue Reward', color='green')
        plt.title('Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # 胜率和存活率
        plt.subplot(3, 2, 2)
        self._plot_smoothed_curve(self.metrics['red_win_rates'], 'Red Win Rate', color='purple')
        self._plot_smoothed_curve(self.metrics['red_survival_rates'], 'Red Survival Rate', color='red')
        self._plot_smoothed_curve(self.metrics['blue_survival_rates'], 'Blue Survival Rate', color='blue')
        plt.title('Performance Metrics')
        plt.xlabel('Episode')
        plt.ylabel('Rate')
        plt.legend()
        
        # 回合长度
        plt.subplot(3, 2, 3)
        self._plot_smoothed_curve(self.metrics['episode_lengths'], 'Episode Length', color='orange')
        plt.title('Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # 探索噪声
        plt.subplot(3, 2, 4)
        plt.plot(list(self.metrics['noise_values']), color='brown', label='Noise Value')
        plt.title('Exploration Noise')
        plt.xlabel('Episode')
        plt.ylabel('Noise Value')
        plt.legend()
        
        # Hit率和伤亡统计
        plt.subplot(3, 2, 5)
        self._plot_smoothed_curve(self.metrics['hit_rates'], 'Hit Rate', color='green')
        plt.title('Combat Effectiveness')
        plt.xlabel('Episode')
        plt.ylabel('Rate')
        plt.legend()
        
        # 伤亡对比
        plt.subplot(3, 2, 6)
        self._plot_smoothed_curve(self.metrics['red_casualties'], 'Red Casualties', color='red')
        self._plot_smoothed_curve(self.metrics['blue_casualties'], 'Blue Casualties', color='blue')
        plt.title('Casualties')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'), dpi=300)
        plt.close()

    def _plot_smoothed_curve(self, data, label, color='blue', window=50):
        """绘制平滑曲线
        
        Args:
            data: 数据列表或deque
            label: 图例标签
            color: 线条颜色
            window: 平滑窗口大小
        """
        if isinstance(data, deque):
            data = list(data)
        if len(data) > window:
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            plt.plot(smoothed, label=label, color=color)
        else:
            plt.plot(data, label=label, color=color)

    def _check_early_stopping(self, current_reward: float) -> bool:
        """检查是否触发早停
        
        Args:
            current_reward: 当前评估奖励
            
        Returns:
            bool: 是否应该停止训练
        """
        if len(self.metrics['episode_rewards']) < self.min_episodes:
            return False
            
        if current_reward > self.best_reward:
            self.best_reward = current_reward
            self.patience_counter = 0
            self._save_checkpoint('best')
        else:
            self.patience_counter += 1
            
        return self.patience_counter >= self.patience

    def _adjust_noise(self, episode: int):
        """动态调整探索噪声
        
        Args:
            episode: 当前回合数
        """
        # 基础衰减
        self.noise = max(self.min_noise, self.noise * self.noise_decay)
        
        # 基于性能的动态调整
        if len(self.metrics['episode_rewards']) >= 100:
            recent_rewards = list(self.metrics['episode_rewards'])[-100:]
            if np.mean(recent_rewards) < np.mean(list(self.metrics['episode_rewards'])[-200:-100]):
                self.noise = min(self.initial_noise, self.noise * 1.1)

    def _save_checkpoint(self, episode):
        """保存检查点
        
        Args:
            episode: 回合数或标识符
        """
        # 保存模型
        model_path = os.path.join(self.save_dir, f'model_episode_{episode}.pt')
        self.maddpg.save(model_path)
        
        # 保存训练状态
        checkpoint = {
            'metrics': {k: list(v) for k, v in self.metrics.items()},
            'config': self._convert_config_to_serializable(self.config),
            'episode': episode,
            'noise': self.noise,
            'best_reward': self.best_reward,
            'patience_counter': self.patience_counter,
            'env_reset_counter': self.env_reset_counter
        }
        
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pt')
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str, model_path: str) -> int:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            model_path: 模型文件路径
            
        Returns:
            int: 加载的回合数
        """
        # 加载训练状态
        checkpoint = torch.load(checkpoint_path)
        self.metrics = {k: deque(v, maxlen=10000) for k, v in checkpoint['metrics'].items()}
        self.noise = checkpoint['noise']
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.env_reset_counter = checkpoint.get('env_reset_counter', 0)
        
        # 加载模型
        self.maddpg.load(model_path)
        
        return checkpoint['episode']

    def _log_eval_metrics(self, metrics: Dict[str, float], episode: int):
        """记录评估指标
        
        Args:
            metrics: 评估指标字典
            episode: 当前回合数
        """
        log_path = os.path.join(self.save_dir, 'eval_metrics.txt')
        with open(log_path, 'a') as f:
            f.write(f"\nEpisode {episode}:\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")

    def _print_current_stats(self):
        """打印当前训练统计信息"""
        recent = lambda x: list(x)[-100:]  # 获取最近100回合的数据
        
        print("\n=== 当前训练状态 ===")
        print(f"最近100回合平均奖励: {np.mean(recent(self.metrics['episode_rewards'])):.1f}")
        print(f"红方平均奖励: {np.mean(recent(self.metrics['red_rewards'])):.1f}")
        print(f"蓝方平均奖励: {np.mean(recent(self.metrics['blue_rewards'])):.1f}")
        print(f"红方胜率: {np.mean(recent(self.metrics['red_win_rates'])):.1%}")
        print(f"红方存活率: {np.mean(recent(self.metrics['red_survival_rates'])):.1%}")
        print(f"蓝方存活率: {np.mean(recent(self.metrics['blue_survival_rates'])):.1%}")
        print(f"平均回合长度: {np.mean(recent(self.metrics['episode_lengths'])):.1f}")
        print(f"当前探索噪声: {self.noise:.3f}")
        print(f"命中率: {np.mean(recent(self.metrics['hit_rates'])):.1%}")
        print(f"红方平均伤亡: {np.mean(recent(self.metrics['red_casualties'])):.1f}")
        print(f"蓝方平均伤亡: {np.mean(recent(self.metrics['blue_casualties'])):.1f}")

    def _save_final_report(self):
        """保存最终训练报告"""
        report = {
            'final_metrics': {
                'total_episodes': len(self.metrics['episode_rewards']),
                'final_100_episodes': {
                    'avg_reward': np.mean(list(self.metrics['episode_rewards'])[-100:]),
                    'avg_red_reward': np.mean(list(self.metrics['red_rewards'])[-100:]),
                    'avg_blue_reward': np.mean(list(self.metrics['blue_rewards'])[-100:]),
                    'red_win_rate': np.mean(list(self.metrics['red_win_rates'])[-100:]),
                    'red_survival_rate': np.mean(list(self.metrics['red_survival_rates'])[-100:]),
                    'blue_survival_rate': np.mean(list(self.metrics['blue_survival_rates'])[-100:]),
                    'avg_episode_length': np.mean(list(self.metrics['episode_lengths'])[-100:]),
                    'avg_hit_rate': np.mean(list(self.metrics['hit_rates'])[-100:]),
                    'avg_red_casualties': np.mean(list(self.metrics['red_casualties'])[-100:]),
                    'avg_blue_casualties': np.mean(list(self.metrics['blue_casualties'])[-100:])
                },
                'best_performance': {
                    'best_reward': self.best_reward,
                    'best_win_rate': max(np.convolve(
                        list(self.metrics['red_win_rates']),
                        np.ones(100)/100,
                        mode='valid'
                    ))
                },
                'training_summary': {
                    'total_episodes': len(self.metrics['episode_rewards']),
                    'early_stopped': self.patience_counter >= self.patience,
                    'final_noise_level': self.noise
                }
            },
            'config': self._convert_config_to_serializable(self.config)
        }
        
        with open(os.path.join(self.save_dir, 'final_report.json'), 'w') as f:
            json.dump(report, f, indent=4)

    def _convert_config_to_serializable(self, config: Dict) -> Dict:
        """将配置转换为可序列化的格式
        
        Args:
            config: 原始配置字典
            
        Returns:
            Dict: 可序列化的配置字典
        """
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
            
        return convert(config)

if __name__ == "__main__":
    # 默认训练配置
    config = {
        'env_config': {
            'num_red': 2,
            'num_blue': 3,
            'max_steps': 200,
            'field_size': 100000.0,
            'attack_range': 25000.0,
            'min_speed': 150.0,
            'max_speed': 400.0
        },
        'model_config': {
            'n_step': 3,
            'gamma': 0.99,
            'capacity': 1000000,
            'alpha': 0.6,
            'beta_start': 0.4,
            'beta_frames': 100000,
            'batch_size': 128,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'weight_decay': 1e-5,
            'dropout': 0.3,
            'hidden_dim': 256,
            'tau': 0.01,
        },
        'training': {
            'n_episodes': 5,
            'save_interval': 100,
            'eval_interval': 100,
            'plot_interval': 100,
            'log_interval': 1,
            'initial_noise': 0.3,
            'noise_decay': 0.9995,
            'min_noise': 0.01,
            'patience': 20,
            'min_episodes': 1000,
            'env_reset_frequency': 1000
        },
        'save_dir': 'results/mozi_training'
    }
    
    # 创建训练器并开始训练
    trainer = MoziEnhancedTrainer(config)
    trainer.train()