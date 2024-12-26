import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
from envs.mozi_adapter import MoziAdapter
from envs.env_config import EnvConfig
from agents.baseline import MADDPG

class MoziBaselineTrainer:
    """墨子平台MADDPG基线版本训练器"""
    
    def __init__(self, config: Dict):
        """
        初始化训练器
        
        Args:
            config: 训练配置参数
        """
        self.config = config
        
        # 确保墨子路径设置
        if not os.environ.get("MOZIPATH"):
            os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
        print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")
        
        # 初始化墨子环境
        self.env = MoziAdapter(
            num_red=config['env_config']['num_red'],
            num_blue=config['env_config']['num_blue'],
            max_steps=config['env_config']['max_steps'],
            env_config=EnvConfig
        )
        
        # 获取观察和动作空间维度
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        
        # 初始化MADDPG智能体
        self.maddpg = MADDPG(
            n_red=config['env_config']['num_red'],
            n_blue=config['env_config']['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim
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
            
        # 初始化训练指标
        self.metrics = {
            'episode_rewards': [],
            'red_rewards': [],
            'blue_rewards': [],
            'red_win_rates': [],
            'red_survival_rates': [],
            'blue_survival_rates': [],
            'episode_lengths': [],
            'noise_values': []
        }
        
        # 设置噪声参数
        self.noise = config['training']['initial_noise']
        self.noise_decay = config['training']['noise_decay']
        self.min_noise = config['training']['min_noise']

    def train(self):
        """执行训练循环"""
        print("\n开始墨子平台MADDPG基线版本训练...")
        try:
            progress_bar = tqdm(range(self.config['training']['n_episodes']))
            
            for episode in progress_bar:
                # 执行一个训练回合
                episode_metrics = self._train_episode()
                
                # 更新并保存指标
                self._update_metrics(episode_metrics)
                
                # 更新进度条信息
                progress_bar.set_postfix({
                    'reward': f"{episode_metrics['episode_reward']:.1f}",
                    'red_wins': f"{self.metrics['red_win_rates'][-1]:.2%}",
                    'steps': episode_metrics['episode_length']
                })
                
                # 定期保存和绘制
                if (episode + 1) % self.config['training']['save_interval'] == 0:
                    self._save_checkpoint(episode + 1)
                    self._plot_metrics()
                    self._print_current_stats()
                
                # 衰减探索噪声
                self.noise = max(
                    self.min_noise,
                    self.noise * self.noise_decay
                )
                
        except KeyboardInterrupt:
            print("\n训练被中断，正在保存...")
        
        finally:
            # 保存最终模型和结果
            self._save_checkpoint('final')
            self._plot_metrics()
            self._save_final_report()
            print(f"\n训练结束! 结果已保存至: {self.save_dir}")

    def _train_episode(self) -> Dict:
        """训练单个回合
        
        Returns:
            Dict: 包含回合统计信息的字典
        """
        obs = self.env.reset()
        episode_reward = 0
        red_reward = 0
        blue_reward = 0
        step = 0
        
        episode_metrics = {
            'episode_reward': 0,
            'red_reward': 0,
            'blue_reward': 0,
            'red_win': False,
            'red_survival_rate': 0,
            'blue_survival_rate': 0,
            'episode_length': 0
        }
        
        while True:
            # 选择动作
            actions = self.maddpg.select_actions(obs, self.noise)
            
            # 执行动作
            next_obs, rewards, done, info = self.env.step(actions)
            
            # 存储经验
            self.maddpg.store_transition(obs, actions, rewards, next_obs, done)
            
            # 如果经验池足够大，进行训练
            if len(self.maddpg.memory) >= self.maddpg.batch_size:
                self.maddpg.train()
            
            # 累积奖励
            episode_reward += sum(rewards.values() if rewards else [0])
            red_reward += sum(v for k, v in rewards.items() if k.startswith('red'))
            blue_reward += sum(v for k, v in rewards.items() if k.startswith('blue'))
            
            # 更新观察
            obs = next_obs
            step += 1
            
            # 每20步打印进度
            if step % 20 == 0:
                print(f"\nStep {step}/{self.config['env_config']['max_steps']}")
                print(f"Red alive: {info['red_alive']}, Blue alive: {info['blue_alive']}")
            
            if done:
                # 计算存活率和胜负
                episode_metrics.update({
                    'episode_reward': episode_reward,
                    'red_reward': red_reward,
                    'blue_reward': blue_reward,
                    'red_win': info['blue_alive'] == 0,
                    'red_survival_rate': info['red_alive'] / self.config['env_config']['num_red'],
                    'blue_survival_rate': info['blue_alive'] / self.config['env_config']['num_blue'],
                    'episode_length': step
                })
                break
                
        return episode_metrics

    def evaluate(self, num_episodes: int = 5):
        """评估模型性能
        
        Args:
            num_episodes: 评估回合数
        """
        print("\n开始模型评估...")
        eval_stats = []
        
        for episode in range(num_episodes):
            print(f"\n评估回合 {episode + 1}/{num_episodes}")
            obs = self.env.reset()
            episode_reward = 0
            
            for step in range(self.config['env_config']['max_steps']):
                # 无噪声动作选择
                actions = self.maddpg.select_actions(obs, noise_scale=0.0)
                next_obs, rewards, done, info = self.env.step(actions)
                
                episode_reward += sum(rewards.values() if rewards else [0])
                obs = next_obs
                
                if done:
                    break
            
            eval_stats.append(episode_reward)
            print(f"回合奖励: {episode_reward:.2f}")
        
        print(f"\n评估结果:")
        print(f"平均奖励: {np.mean(eval_stats):.2f}")
        print(f"标准差: {np.std(eval_stats):.2f}")
        
        return eval_stats

    def _update_metrics(self, episode_metrics: Dict):
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

    def _plot_metrics(self):
        """绘制训练指标图表"""
        plt.figure(figsize=(15, 10))
        
        # 奖励曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics['episode_rewards'], label='Total')
        plt.plot(self.metrics['red_rewards'], label='Red', alpha=0.6)
        plt.plot(self.metrics['blue_rewards'], label='Blue', alpha=0.6)
        plt.title('Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        
        # 胜率和存活率
        plt.subplot(2, 2, 2)
        window = 100
        win_rates = np.convolve(
            self.metrics['red_win_rates'],
            np.ones(window)/window,
            mode='valid'
        )
        plt.plot(win_rates, label='Red Win Rate')
        plt.plot(self.metrics['red_survival_rates'], label='Red Survival', alpha=0.6)
        plt.plot(self.metrics['blue_survival_rates'], label='Blue Survival', alpha=0.6)
        plt.title('Performance Metrics')
        plt.xlabel('Episode')
        plt.ylabel('Rate')
        plt.legend()
        
        # 回合长度
        plt.subplot(2, 2, 3)
        plt.plot(self.metrics['episode_lengths'])
        plt.title('Episode Length')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        # 探索噪声
        plt.subplot(2, 2, 4)
        plt.plot(self.metrics['noise_values'])
        plt.title('Exploration Noise')
        plt.xlabel('Episode')
        plt.ylabel('Noise Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'))
        plt.close()

    def _save_checkpoint(self, episode):
        """保存检查点
        
        Args:
            episode: 当前回合数或标识符
        """
        # 保存 MADDPG 模型
        model_path = os.path.join(self.save_dir, f'model_episode_{episode}.pt')
        self.maddpg.save(model_path)
        
        # 保存训练状态
        checkpoint = {
            'metrics': self.metrics,
            'config': self.config,
            'episode': episode,
            'noise': self.noise
        }
        torch.save(
            checkpoint,
            os.path.join(self.save_dir, f'checkpoint_episode_{episode}.pt')
        )

    def load_checkpoint(self, checkpoint_path, model_path):
        """加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            model_path: 模型文件路径
        """
        # 加载训练状态
        checkpoint = torch.load(checkpoint_path)
        self.metrics = checkpoint['metrics']
        self.noise = checkpoint['noise']
        
        # 加载模型
        self.maddpg.load(model_path)
        
        return checkpoint['episode']

    def _save_final_report(self):
        """保存最终训练报告"""
        report = {
            'config': self.config,
            'final_metrics': {
                'total_episodes': len(self.metrics['episode_rewards']),
                'avg_reward_last_100': np.mean(self.metrics['episode_rewards'][-100:]),
                'red_win_rate_last_100': np.mean(self.metrics['red_win_rates'][-100:]),
                'avg_episode_length': np.mean(self.metrics['episode_lengths']),
                'red_survival_rate': np.mean(self.metrics['red_survival_rates'][-100:]),
                'blue_survival_rate': np.mean(self.metrics['blue_survival_rates'][-100:])
            }
        }
        
        with open(os.path.join(self.save_dir, 'final_report.json'), 'w') as f:
            json.dump(report, f, indent=4)

    def _print_current_stats(self):
        """打印当前训练统计信息"""
        print("\n当前训练状态:")
        print(f"最近100回合平均奖励: {np.mean(self.metrics['episode_rewards'][-100:]):.1f}")
        print(f"红方胜率: {np.mean(self.metrics['red_win_rates'][-100:]):.1%}")
        print(f"红方存活率: {np.mean(self.metrics['red_survival_rates'][-100:]):.1%}")
        print(f"蓝方存活率: {np.mean(self.metrics['blue_survival_rates'][-100:]):.1%}")
        print(f"平均回合长度: {np.mean(self.metrics['episode_lengths'][-100:]):.1f}")
        print(f"当前探索噪声: {self.noise:.3f}")

if __name__ == "__main__":
    # 训练配置
    config = {
        'env_config': {
            'num_red': 2,
            'num_blue': 3,
            'max_steps': 200,
            'field_size': 100000.0,  # 战场大小
            'attack_range': 25000.0,  # 攻击范围
            'min_speed': 150.0,      # 最小速度
            'max_speed': 400.0,      # 最大速度
            'max_turn_rate': np.pi/6, # 最大转向率
            'hit_probability': 0.8,   # 命中概率
            'num_threads': 8         # 并行线程数
        },
        'training': {
            'n_episodes': 4,       # 训练回合数
            'save_interval': 2,     # 保存间隔
            'initial_noise': 0.3,     # 初始探索噪声
            'noise_decay': 0.9995,    # 噪声衰减率
            'min_noise': 0.01        # 最小噪声
        },
        'save_dir': './results/mozi_baseline'  # 结果保存路径
    }
    
    try:
        # 创建训练器
        trainer = MoziBaselineTrainer(config)
        
        # 开始训练
        trainer.train()
        
        # 训练完成后进行评估
        print("\n训练完成，开始评估模型...")
        eval_results = trainer.evaluate(num_episodes=5)
        
        # 保存评估结果
        eval_stats = {
            'mean_reward': float(np.mean(eval_results)),
            'std_reward': float(np.std(eval_results)),
            'min_reward': float(np.min(eval_results)),
            'max_reward': float(np.max(eval_results))
        }
        
        with open(os.path.join(trainer.save_dir, 'eval_results.json'), 'w') as f:
            json.dump(eval_stats, f, indent=4)
            
        print("\n评估结果已保存!")
        
    except KeyboardInterrupt:
        print("\n训练被手动中断")
    except Exception as e:
        print(f"\n训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n程序结束")