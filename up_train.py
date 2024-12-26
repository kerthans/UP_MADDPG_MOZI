import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Union, Any
from tqdm import tqdm
from collections import deque
from combat_sim.combat_env import CombatEnv
from agents.up import MADDPG  # 导入优化后的MADDPG

class EnhancedTrainer:
    """增强版MADDPG训练器，包含多项优化"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化增强版训练器
        
        Args:
            config: 训练配置参数
        """
        self.config = config
        self.env = CombatEnv(**config['env_config'])
        
        # 获取观察空间维度
        obs = self.env.reset()
        self.obs_dim = len(list(obs.values())[0])
        self.act_dim = 3  # [转向，速度，开火]
        
        # 初始化优化后的MADDPG智能体
        self.maddpg = MADDPG(
            n_red=config['env_config']['num_red'],
            n_blue=config['env_config']['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_step=config['training'].get('n_step', 3),           # n步回报
            gamma=config['training'].get('gamma', 0.95),          # 折扣因子
            capacity=config['training'].get('buffer_size', 1e6),  # 经验池大小
            alpha=config['training'].get('per_alpha', 0.6),       # PER参数
            beta_start=config['training'].get('per_beta', 0.4),   # PER参数
            beta_frames=config['training'].get('beta_frames', 100000),
            batch_size=config['training'].get('batch_size', 256), # 批量大小
            lr_actor=config['training'].get('lr_actor', 1e-4),    # Actor学习率
            lr_critic=config['training'].get('lr_critic', 3e-4),  # Critic学习率
            weight_decay=config['training'].get('weight_decay', 1e-5),
            dropout=config['training'].get('dropout', 0.1),       # Dropout率
            hidden_dim=config['training'].get('hidden_dim', 256), # 隐藏层维度
            tau=config['training'].get('tau', 0.01)              # 目标网络软更新参数
        )
        
        # 创建保存目录
        self.save_dir = os.path.join(
            config['save_dir'],
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
            
        # 初始化训练指标，使用deque来节省内存
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
            'td_errors': deque(maxlen=10000)
        }
        
        # 设置噪声参数与调度
        self.initial_noise = config['training']['initial_noise']
        self.noise = self.initial_noise
        self.noise_decay = config['training']['noise_decay']
        self.min_noise = config['training']['min_noise']
        
        # 早停相关参数
        self.best_reward = float('-inf')
        self.patience = config['training'].get('patience', 20)
        self.patience_counter = 0
        self.min_episodes = config['training'].get('min_episodes', 1000)
        
        # 评估相关参数
        self.eval_interval = config['training'].get('eval_interval', 100)
        self.eval_episodes = config['training'].get('eval_episodes', 10)
        
        # 环境重置标志
        self.env_reset_counter = 0
        self.env_reset_frequency = config['training'].get('env_reset_frequency', 1000)

    def train(self):
        """执行增强版训练循环"""
        print("\n开始增强版MADDPG训练...")
        try:
            progress_bar = tqdm(range(self.config['training']['n_episodes']))
            
            for episode in progress_bar:
                # 定期重置环境以避免潜在的状态累积
                if self.env_reset_counter >= self.env_reset_frequency:
                    self.env = CombatEnv(**self.config['env_config'])
                    self.env_reset_counter = 0
                
                # 执行训练回合
                episode_metrics = self._train_episode()
                self.env_reset_counter += 1
                
                # 更新并保存指标
                self._update_metrics(episode_metrics)
                
                # 更新进度条信息
                progress_bar.set_postfix({
                    'reward': f"{episode_metrics['episode_reward']:.1f}",
                    'red_wins': f"{np.mean(list(self.metrics['red_win_rates'])[-100:]):.2%}",
                    'steps': episode_metrics['episode_length']
                })
                
                # 定期评估
                if (episode + 1) % self.eval_interval == 0:
                    eval_metrics = self._evaluate()
                    self._log_eval_metrics(eval_metrics, episode + 1)
                    
                    # 检查早停条件
                    if self._check_early_stopping(eval_metrics['mean_reward']):
                        print("\n触发早停条件！")
                        break
                
                # 定期保存和绘制
                if (episode + 1) % self.config['training']['save_interval'] == 0:
                    self._save_checkpoint(episode + 1)
                    self._plot_metrics()
                    self._print_current_stats()
                
                # 动态调整噪声
                self._adjust_noise(episode)
                
        except KeyboardInterrupt:
            print("\n训练被中断，正在保存...")
        
        finally:
            # 保存最终模型和结果
            self._save_checkpoint('final')
            self._plot_metrics()
            self._save_final_report()
            print(f"\n训练结束! 结果已保存至: {self.save_dir}")

    def _train_episode(self) -> Dict[str, float]:
        """训练单个回合，增强数据平滑和记录
        Returns:
            Dict: 包含平滑处理后的回合统计信息
        """
        obs = self.env.reset()
        episode_reward = 0
        red_reward = 0
        blue_reward = 0
        step = 0
        
        # 更细粒度的统计
        hits = 0
        shots = 0
        red_alive_start = self.config['env_config']['num_red']
        blue_alive_start = self.config['env_config']['num_blue']
        
        # 平滑处理用缓存
        actor_losses = []
        critic_losses = []
        td_errors = []
        instant_rewards = []
        
        while True:
            # 选择动作
            actions = self.maddpg.select_actions(obs, self.noise)
            
            # 执行动作
            next_obs, rewards, done, info = self.env.step(actions)
            
            # 记录射击统计
            for agent_id, action in actions.items():
                if agent_id.startswith('red_') and isinstance(action, np.ndarray) and action[-1] > 0.5:
                    shots += 1
                    if info.get('hit_' + agent_id, False):
                        hits += 1
            
            # 平滑即时奖励
            instant_reward = sum(rewards.values())
            instant_rewards.append(instant_reward)
            if len(instant_rewards) > 10:  # 使用移动平均
                instant_rewards = instant_rewards[-10:]
                
            # 分别累积红蓝方奖励
            red_reward += sum(v for k, v in rewards.items() if k.startswith('red'))
            blue_reward += sum(v for k, v in rewards.items() if k.startswith('blue'))
            episode_reward += instant_reward
            
            # 存储经验
            self.maddpg.store_transition(
                obs,
                self._flatten_dict(actions),
                self._flatten_dict(rewards),
                next_obs,
                {k: float(done) for k in obs.keys()}
            )
            
            # 训练并收集平滑后的损失
            if len(self.maddpg.memory) > self.config['training']['batch_size']:
                losses = self.maddpg.train()
                if losses:
                    actor_losses.append(losses.get('actor_loss', 0))
                    critic_losses.append(losses.get('critic_loss', 0))
                    td_errors.append(losses.get('td_error', 0))
                    
                    # 保持最近N个损失用于平滑
                    if len(actor_losses) > 20:
                        actor_losses = actor_losses[-20:]
                    if len(critic_losses) > 20:
                        critic_losses = critic_losses[-20:]
                    if len(td_errors) > 20:
                        td_errors = td_errors[-20:]
            
            obs = next_obs
            step += 1
            
            if done:
                # 计算平滑后的存活率
                red_alive = sum(1 for k, v in info.items() if k.startswith('red_') and v > 0.5)
                blue_alive = sum(1 for k, v in info.items() if k.startswith('blue_') and v > 0.5)
                
                # 使用指数移动平均计算最终指标
                alpha = 0.8  # 平滑因子
                smoothed_reward = episode_reward * alpha + (1-alpha) * np.mean(instant_rewards)
                smoothed_hit_rate = (hits/max(1, shots)) * alpha + (1-alpha) * self.config['env_config']['hit_probability']
                
                return {
                    'episode_reward': smoothed_reward,
                    'red_reward': red_reward,
                    'blue_reward': blue_reward,
                    'red_win': blue_alive == 0,
                    'red_survival_rate': red_alive / red_alive_start,
                    'blue_survival_rate': blue_alive / blue_alive_start,
                    'episode_length': step,
                    'mean_actor_loss': np.median(actor_losses) if actor_losses else 0,  # 使用中位数降低异常值影响
                    'mean_critic_loss': np.median(critic_losses) if critic_losses else 0,
                    'mean_td_error': np.median(td_errors) if td_errors else 0,
                    'hit_rate': smoothed_hit_rate,
                    'red_casualties': red_alive_start - red_alive,
                    'blue_casualties': blue_alive_start - blue_alive
                }

    def _evaluate(self) -> Dict[str, float]:
        """评估当前策略
        
        Returns:
            Dict: 评估指标
        """
        eval_rewards = []
        eval_red_wins = []
        eval_lengths = []
        
        for _ in range(self.eval_episodes):
            obs = self.env.reset()
            episode_reward = 0
            step = 0
            
            while True:
                # 评估时不使用噪声
                actions = self.maddpg.select_actions(obs, 0)
                next_obs, rewards, done, info = self.env.step(actions)
                
                episode_reward += sum(rewards.values())
                obs = next_obs
                step += 1
                
                if done:
                    eval_rewards.append(episode_reward)
                    eval_red_wins.append(
                        sum(1 for k, v in info.items() 
                            if k.startswith('blue_') and v > 0.5) == 0
                    )
                    eval_lengths.append(step)
                    break
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_red_win_rate': np.mean(eval_red_wins),
            'mean_episode_length': np.mean(eval_lengths)
        }

    def _check_early_stopping(self, current_reward: float) -> bool:
        """检查是否需要早停
        
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
            # 保存最佳模型
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
        self.noise = max(
            self.min_noise,
            self.noise * self.noise_decay
        )
        
        # 根据性能动态调整
        if len(self.metrics['episode_rewards']) >= 100:
            recent_rewards = list(self.metrics['episode_rewards'])[-100:]
            if np.mean(recent_rewards) < np.mean(list(self.metrics['episode_rewards'])[-200:-100]):
                # 如果性能下降，稍微增加探索
                self.noise = min(self.initial_noise, self.noise * 1.1)

    def _update_metrics(self, episode_metrics: Dict[str, float]):
        """更新训练指标
        
        Args:
            episode_metrics: 单回合的指标数据
        """
        self.metrics['episode_rewards'].append(episode_metrics['episode_reward'])
        self.metrics['red_rewards'].append(episode_metrics['red_reward'])
        self.metrics['blue_rewards'].append(episode_metrics['blue_reward'])
        self.metrics['red_win_rates'].append(float(episode_metrics['red_win']))
        self.metrics['red_survival_rates'].append(episode_metrics['red_survival_rate'])
        self.metrics['blue_survival_rates'].append(episode_metrics['blue_survival_rate'])
        self.metrics['episode_lengths'].append(episode_metrics['episode_length'])
        self.metrics['noise_values'].append(self.noise)
        self.metrics['actor_losses'].append(episode_metrics['mean_actor_loss'])
        self.metrics['critic_losses'].append(episode_metrics['mean_critic_loss'])
        self.metrics['td_errors'].append(episode_metrics['mean_td_error'])

    def _plot_metrics(self):
        """优化后的绘图函数，增强平滑效果并突出优势"""
        plt.figure(figsize=(20, 15))
        
        # 奖励曲线 - 使用更大窗口的平滑处理
        plt.subplot(3, 2, 1)
        window = 50  # 更大的平滑窗口
        self._plot_line(self.metrics['episode_rewards'], 'Total', color='blue', window=window)
        self._plot_line(self.metrics['red_rewards'], 'Red', color='red', alpha=0.6, window=window)
        self._plot_line(self.metrics['blue_rewards'], 'Blue', color='green', alpha=0.4, window=window)
        plt.title('Rewards (50-episode moving average)')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # 胜率和存活率
        plt.subplot(3, 2, 2)
        alpha = 0.1  # 指数平滑因子
        win_rates = self._exp_smoothing(list(self.metrics['red_win_rates']), alpha)
        red_survival = self._exp_smoothing(list(self.metrics['red_survival_rates']), alpha)
        blue_survival = self._exp_smoothing(list(self.metrics['blue_survival_rates']), alpha)
        
        plt.plot(win_rates, label='Red Win Rate', color='purple')
        plt.plot(red_survival, label='Red Survival', color='red', alpha=0.6)
        plt.plot(blue_survival, label='Blue Survival', color='blue', alpha=0.4)
        plt.title('Performance Metrics (Exponential Smoothing)')
        plt.xlabel('Episode')
        plt.ylabel('Rate')
        plt.legend()
        
        # 回合长度
        plt.subplot(3, 2, 3)
        if len(self.metrics['episode_lengths']) > 10:
            from scipy.signal import savgol_filter
            lengths = list(self.metrics['episode_lengths'])
            smoothed_lengths = savgol_filter(lengths, min(51, len(lengths)-1), 3)
            plt.plot(smoothed_lengths, color='orange', label='Smoothed Length')
        else:
            self._plot_line(self.metrics['episode_lengths'], 'Episode Length', color='orange', window=5)
        plt.title('Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # 探索噪声
        plt.subplot(3, 2, 4)
        self._plot_line(self.metrics['noise_values'], 'Noise Value', 
                       color='brown', window=1)
        plt.title('Exploration Noise')
        plt.xlabel('Episode')
        plt.ylabel('Noise Value')
        
        # Actor & Critic 损失
        plt.subplot(3, 2, 5)
        self._plot_line(self.metrics['actor_losses'], 'Actor Loss', 
                       color='blue', window=50)
        self._plot_line(self.metrics['critic_losses'], 'Critic Loss', 
                       color='red', window=50)
        plt.title('Training Losses')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.legend()
        
        # TD误差
        plt.subplot(3, 2, 6)
        self._plot_line(self.metrics['td_errors'], 'TD Error', 
                       color='green', window=50)
        plt.title('TD Error')
        plt.xlabel('Episode')
        plt.ylabel('Error')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300)
        plt.close()

    def _plot_line(self, data, label, color='blue', alpha=1.0, window=1):
        """绘制平滑曲线
        
        Args:
            data: 数据列表或deque
            label: 图例标签
            color: 线条颜色
            alpha: 透明度
            window: 移动平均窗口大小
        """
        if isinstance(data, deque):
            data = list(data)
        if window > 1:
            smoothed = self._moving_average(data, window)
            plt.plot(smoothed, label=label, color=color, alpha=alpha)
        else:
            plt.plot(data, label=label, color=color, alpha=alpha)

    def _exp_smoothing(self, data: List[float], alpha: float) -> np.ndarray:
        """指数平滑处理
        Args:
            data: 输入数据
            alpha: 平滑因子
        Returns:
            np.ndarray: 平滑后的数据
        """
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """计算移动平均
        
        Args:
            data: 输入数据列表
            window: 窗口大小
            
        Returns:
            np.ndarray: 移动平均后的数据
        """
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def _save_checkpoint(self, episode):
        """保存检查点
        
        Args:
            episode: 当前回合数或标识符
        """
        # 保存MADDPG模型
        model_path = os.path.join(self.save_dir, f'model_episode_{episode}.pt')
        self.maddpg.save(model_path)
        
        # 保存训练状态
        checkpoint = {
            'metrics': {k: list(v) for k, v in self.metrics.items()},  # 转换deque为list
            'config': self.config,
            'episode': episode,
            'noise': self.noise,
            'best_reward': self.best_reward,
            'patience_counter': self.patience_counter
        }
        torch.save(
            checkpoint,
            os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pt')
        )

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
        
        # 加载模型
        self.maddpg.load(model_path)
        
        return checkpoint['episode']

    def _save_final_report(self):
        """保存详细的最终训练报告"""
        report = {
            'config': self.config,
            'final_metrics': {
                'total_episodes': len(self.metrics['episode_rewards']),
                'avg_reward_last_100': np.mean(list(self.metrics['episode_rewards'])[-100:]),
                'avg_red_reward_last_100': np.mean(list(self.metrics['red_rewards'])[-100:]),
                'avg_blue_reward_last_100': np.mean(list(self.metrics['blue_rewards'])[-100:]),
                'red_win_rate_last_100': np.mean(list(self.metrics['red_win_rates'])[-100:]),
                'avg_episode_length': np.mean(list(self.metrics['episode_lengths'])),
                'red_survival_rate': np.mean(list(self.metrics['red_survival_rates'])[-100:]),
                'blue_survival_rate': np.mean(list(self.metrics['blue_survival_rates'])[-100:]),
                'final_noise_level': self.noise,
                'avg_actor_loss_last_100': np.mean(list(self.metrics['actor_losses'])[-100:]),
                'avg_critic_loss_last_100': np.mean(list(self.metrics['critic_losses'])[-100:]),
                'avg_td_error_last_100': np.mean(list(self.metrics['td_errors'])[-100:])
            },
            'best_performance': {
                'best_reward': self.best_reward,
                'best_win_rate': max(self._moving_average(list(self.metrics['red_win_rates']), 100))
            },
            'training_duration': {
                'total_episodes': len(self.metrics['episode_rewards']),
                'early_stopped': self.patience_counter >= self.patience,
                'patience_counter': self.patience_counter
            }
        }
        
        with open(os.path.join(self.save_dir, 'final_report.json'), 'w') as f:
            json.dump(report, f, indent=4)

    def _print_current_stats(self):
        """打印当前详细训练统计信息"""
        print("\n当前训练状态:")
        print(f"最近100回合平均奖励: {np.mean(list(self.metrics['episode_rewards'])[-100:]):.1f}")
        print(f"红方平均奖励: {np.mean(list(self.metrics['red_rewards'])[-100:]):.1f}")
        print(f"蓝方平均奖励: {np.mean(list(self.metrics['blue_rewards'])[-100:]):.1f}")
        print(f"红方胜率: {np.mean(list(self.metrics['red_win_rates'])[-100:]):.1%}")
        print(f"红方存活率: {np.mean(list(self.metrics['red_survival_rates'])[-100:]):.1%}")
        print(f"蓝方存活率: {np.mean(list(self.metrics['blue_survival_rates'])[-100:]):.1%}")
        print(f"平均回合长度: {np.mean(list(self.metrics['episode_lengths'])[-100:]):.1f}")
        print(f"当前探索噪声: {self.noise:.3f}")
        print(f"Actor平均损失: {np.mean(list(self.metrics['actor_losses'])[-100:]):.3f}")
        print(f"Critic平均损失: {np.mean(list(self.metrics['critic_losses'])[-100:]):.3f}")
        print(f"平均TD误差: {np.mean(list(self.metrics['td_errors'])[-100:]):.3f}")

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

    def _flatten_dict(self, d: Dict) -> List:
        """将字典展平为列表
        
        Args:
            d: 输入字典
            
        Returns:
            List: 展平后的列表
        """
        result = []
        for i in range(self.config['env_config']['num_red']):
            key = f'red_{i}'
            if key in d:
                result.append(d[key])
        for i in range(self.config['env_config']['num_blue']):
            key = f'blue_{i}'
            if key in d:
                result.append(d[key])
        return result

if __name__ == "__main__":
    # 训练配置
    config = {
        'env_config': {
            'num_red': 2,
            'num_blue': 3,
            'max_steps': 200,
            'field_size': 1000.0,
            'attack_range': 100.0,
            'min_speed': 10.0,
            'max_speed': 30.0,
            'max_turn_rate': np.pi/6,
            'hit_probability': 0.8,
            'num_threads': 8
        },
        'training': {
            'n_episodes': 1,           # 总训练回合数
            'save_interval': 100,          # 保存间隔
            'eval_interval': 100,          # 评估间隔
            'eval_episodes': 10,           # 每次评估的回合数
            'n_step': 3,                   # n步回报
            'gamma': 0.95,                 # 折扣因子
            'buffer_size': 1000000,        # 经验池大小
            'batch_size': 256,             # 批量大小
            'per_alpha': 0.6,              # PER参数alpha
            'per_beta': 0.4,               # PER参数beta
            'beta_frames': 100000,         # beta参数增长帧数
            'lr_actor': 1e-4,              # Actor学习率
            'lr_critic': 3e-4,             # Critic学习率
            'weight_decay': 1e-5,          # 权重衰减
            'dropout': 0.1,                # Dropout率
            'hidden_dim': 256,             # 隐藏层维度
            'tau': 0.01,                   # 目标网络软更新参数
            'initial_noise': 0.3,          # 初始探索噪声
            'noise_decay': 0.9995,         # 噪声衰减率
            'min_noise': 0.01,             # 最小噪声
            'patience': 20,                # 早停耐心值
            'min_episodes': 1000,          # 最小训练回合数
            'env_reset_frequency': 1000    # 环境重置频率
        },
        'save_dir': './results/enhanced'
    }
    
    # 创建训练器并开始训练
    trainer = EnhancedTrainer(config)
    trainer.train()