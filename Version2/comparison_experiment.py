# comparison_experiment.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from env.combat_env import CombatEnv
from agents.maddpg import MADDPG
import json
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class BaselineMADDPG(MADDPG):
    """基础版MADDPG，移除了改进特性"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 移除ICM网络相关组件
        self.icm = None
        self.icm_optimizer = None
    
    def _compute_intrinsic_rewards(self, states, actions, next_states):
        """基础版本不使用内在奖励"""
        return torch.zeros_like(states[:, :self.num_agents])
    
    def _compute_curiosity_rewards(self, states, actions, next_states):
        """基础版本不使用好奇心奖励"""
        return torch.zeros_like(states[:, :self.num_agents])
    
    def update(self):
        """基础版本的更新方法"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # 采样经验
        samples, indices, weights = self.replay_buffer.sample(self.batch_size)
        states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*samples)
        
        # 准备数据
        states = torch.FloatTensor(np.array(states_list)).to(self.device).view(self.batch_size, -1)
        next_states = torch.FloatTensor(np.array(next_states_list)).to(self.device).view(self.batch_size, -1)
        actions = torch.FloatTensor(np.array(actions_list)).to(self.device).view(self.batch_size, -1)
        rewards = torch.FloatTensor(np.array(rewards_list)).to(self.device).view(self.batch_size, -1)
        dones = torch.FloatTensor(np.array(dones_list)).to(self.device).view(self.batch_size, 1)
        
        # 基础版本更新
        for agent_idx in range(self.num_agents):
            # 更新Critic
            self.critic_optimizers[agent_idx].zero_grad()
            next_actions = self._get_target_actions(next_states)
            target_q = self.target_critics[agent_idx](next_states, next_actions)
            target_q = rewards[:, agent_idx].unsqueeze(1) + (1 - dones) * self.gamma * target_q
            current_q = self.critics[agent_idx](states, actions)
            critic_loss = F.mse_loss(current_q, target_q.detach())
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent_idx].parameters(), 0.5)
            self.critic_optimizers[agent_idx].step()
            
            # 更新Actor
            self.actor_optimizers[agent_idx].zero_grad()
            current_actions = self._get_current_actions(states, agent_idx)
            actor_loss = -self.critics[agent_idx](states, current_actions).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 0.5)
            self.actor_optimizers[agent_idx].step()
        
        # 软更新目标网络
        self.soft_update_targets()

def run_experiment(config, agent_type='improved', seed=None):
    """运行单次实验
    Args:
        config: 实验配置字典
        agent_type: 'improved' 或 'baseline'
        seed: 随机种子
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # 创建环境
    env = CombatEnv(
        num_red=config['num_red'],
        num_blue=config['num_blue'],
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        max_steps=config['max_steps'],
        heterogeneous=config['heterogeneous']
    )
    
    # 选择算法类型
    agent_class = MADDPG if agent_type == 'improved' else BaselineMADDPG
    
    # 创建智能体
    agent = agent_class(
        num_agents=config['num_red'] + config['num_blue'],
        state_dim=config['state_dim'],
        action_dim=config['action_dim'],
        lr=config['lr'],
        gamma=config['gamma'],
        tau=config['tau'],
        buffer_size=config['buffer_size'],
        batch_size=config['batch_size'],
        hidden_dim=config['hidden_dim'],
        device=config['device']
    )
    
    # 初始化指标记录
    metrics = {
        'episode_rewards': [],
        'red_rewards': [],
        'blue_rewards': [],
        'red_survival': [],
        'blue_survival': [],
        'steps': [],
        'win_rate': []  # 新增胜率指标
    }
    
    # 训练循环
    progress_bar = tqdm(range(config['num_episodes']), 
                       desc=f'Training {agent_type} MADDPG',
                       ncols=100)
    
    window_size = 100  # 计算移动平均的窗口大小
    wins = 0  # 记录胜利次数
    
    for episode in progress_bar:
        obs = env.reset()
        episode_reward = 0
        red_reward = blue_reward = 0
        step_count = 0
        
        # 记录初始存活状态
        initial_red_alive = env.red_alive.copy()
        initial_blue_alive = env.blue_alive.copy()
        
        # 回合进行
        while step_count < config['max_steps']:
            # 获取每个智能体的观察
            per_agent_obs = []
            for i in range(agent.num_agents):
                agent_obs = obs[i * config['state_dim']:(i + 1) * config['state_dim']]
                per_agent_obs.append(agent_obs)
            
            # 选择动作
            noise = max(0.1, 1.0 - episode/config['num_episodes'])  # 衰减噪声
            actions = agent.select_actions(per_agent_obs, noise=noise)
            
            # 执行动作
            next_obs, rewards, done, info = env.step(actions)
            
            # 计算奖励
            red_rewards = rewards[:config['num_red']]
            blue_rewards = rewards[config['num_red']:]
            red_reward += sum(red_rewards)
            blue_reward += sum(blue_rewards)
            
            # 存储经验
            agent.replay_buffer.add(
                state=obs,
                action=np.array(actions),
                reward=np.array(rewards),
                next_state=next_obs,
                done=done
            )
            
            # 更新网络
            if agent.replay_buffer.size() >= config['batch_size']:
                agent.update()
            
            episode_reward += sum(rewards)
            obs = next_obs
            step_count += 1
            
            if done:
                break
        
        # 计算存活率
        red_survival = np.mean(env.red_alive) / np.mean(initial_red_alive) if np.mean(initial_red_alive) > 0 else 0
        blue_survival = np.mean(env.blue_alive) / np.mean(initial_blue_alive) if np.mean(initial_blue_alive) > 0 else 0
        
        # 判断胜负
        red_win = np.mean(env.red_alive) > np.mean(env.blue_alive)
        if red_win:
            wins += 1
        
        # 更新指标
        metrics['episode_rewards'].append(episode_reward)
        metrics['red_rewards'].append(red_reward)
        metrics['blue_rewards'].append(blue_reward)
        metrics['red_survival'].append(red_survival)
        metrics['blue_survival'].append(blue_survival)
        metrics['steps'].append(step_count)
        metrics['win_rate'].append(wins / (episode + 1))
        
        # 计算移动平均
        window_start = max(0, episode - window_size + 1)
        avg_reward = np.mean(metrics['episode_rewards'][window_start:])
        avg_survival = np.mean(metrics['red_survival'][window_start:])
        
        # 更新进度条
        progress_bar.set_postfix({
            'Reward': f"{avg_reward:.2f}",
            'RedSurv': f"{avg_survival:.2%}",
            'WinRate': f"{wins/(episode+1):.2%}"
        })
    
    return metrics

def plot_comparison(improved_metrics, baseline_metrics, save_dir):
    """绘制详细的对比图
    Args:
        improved_metrics: 改进版MADDPG的指标数据
        baseline_metrics: 基础版MADDPG的指标数据
        save_dir: 结果保存目录
    """
    # 设置全局绘图样式
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.figsize'] = (15, 18)
    
    # 创建子图
    fig, axes = plt.subplots(3, 2)
    fig.suptitle('MADDPG vs Baseline Performance Comparison', fontsize=16, y=0.95)
    
    # 1. 总奖励对比
    ax1 = axes[0, 0]
    window = 50  # 平滑窗口大小
    improved_rewards = pd.Series(improved_metrics['episode_rewards']).rolling(window).mean()
    baseline_rewards = pd.Series(baseline_metrics['episode_rewards']).rolling(window).mean()
    
    ax1.plot(improved_rewards, label='Improved', color='blue', alpha=0.8)
    ax1.plot(baseline_rewards, label='Baseline', color='red', alpha=0.8)
    ax1.fill_between(range(len(improved_rewards)), 
                    improved_rewards - improved_rewards.std(),
                    improved_rewards + improved_rewards.std(),
                    color='blue', alpha=0.2)
    ax1.fill_between(range(len(baseline_rewards)),
                    baseline_rewards - baseline_rewards.std(),
                    baseline_rewards + baseline_rewards.std(),
                    color='red', alpha=0.2)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards (Moving Average)')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # 2. 胜率对比
    ax2 = axes[0, 1]
    ax2.plot(improved_metrics['win_rate'], label='Improved', color='blue', alpha=0.8)
    ax2.plot(baseline_metrics['win_rate'], label='Baseline', color='red', alpha=0.8)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Win Rate Comparison')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    # 3. 存活率对比
    ax3 = axes[1, 0]
    improved_survival = pd.Series(improved_metrics['red_survival']).rolling(window).mean()
    baseline_survival = pd.Series(baseline_metrics['red_survival']).rolling(window).mean()
    
    ax3.plot(improved_survival, label='Improved', color='blue', alpha=0.8)
    ax3.plot(baseline_survival, label='Baseline', color='red', alpha=0.8)
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Red Team Survival Rate')
    ax3.set_title('Survival Rate Comparison')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()
    
    # 4. 每回合步数对比
    ax4 = axes[1, 1]
    improved_steps = pd.Series(improved_metrics['steps']).rolling(window).mean()
    baseline_steps = pd.Series(baseline_metrics['steps']).rolling(window).mean()
    
    ax4.plot(improved_steps, label='Improved', color='blue', alpha=0.8)
    ax4.plot(baseline_steps, label='Baseline', color='red', alpha=0.8)
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Steps per Episode')
    ax4.set_title('Episode Length Comparison')
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.legend()
    
    # 5. 步数分布直方图
    ax5 = axes[2, 0]
    ax5.hist(improved_metrics['steps'], bins=30, alpha=0.5, color='blue', 
             density=True, label='Improved')
    ax5.hist(baseline_metrics['steps'], bins=30, alpha=0.5, color='red', 
             density=True, label='Baseline')
    ax5.set_xlabel('Steps per Episode')
    ax5.set_ylabel('Density')
    ax5.set_title('Steps Distribution')
    ax5.grid(True, linestyle='--', alpha=0.7)
    ax5.legend()
    
    # 6. 性能指标汇总
# 6. 性能指标汇总
    ax6 = axes[2, 1]
    ax6.axis('off')
    final_window = 100  # 最后100回合的统计
    summary_text = (
        'Performance Summary (Last 100 Episodes):\n\n'
        f"Improved MADDPG:\n"
        f"Avg Reward: {np.mean(improved_metrics['episode_rewards'][-final_window:]):.2f}\n"
        f"Avg Survival: {np.mean(improved_metrics['red_survival'][-final_window:])*100:.2f}%\n"
        f"Final Win Rate: {improved_metrics['win_rate'][-1]*100:.2f}%\n"
        f"Avg Steps: {np.mean(improved_metrics['steps'][-final_window:]):.1f}\n\n"
        f"Baseline MADDPG:\n"
        f"Avg Reward: {np.mean(baseline_metrics['episode_rewards'][-final_window:]):.2f}\n"
        f"Avg Survival: {np.mean(baseline_metrics['red_survival'][-final_window:])*100:.2f}%\n"
        f"Final Win Rate: {baseline_metrics['win_rate'][-1]*100:.2f}%\n"
        f"Avg Steps: {np.mean(baseline_metrics['steps'][-final_window:]):.1f}\n\n"
        f"Improvement Ratios:\n"
        f"Reward: {np.mean(improved_metrics['episode_rewards'][-final_window:])/np.mean(baseline_metrics['episode_rewards'][-final_window:]):.2f}x\n"
        f"Survival: {np.mean(improved_metrics['red_survival'][-final_window:])/np.mean(baseline_metrics['red_survival'][-final_window:]):.2f}x\n"
        f"Win Rate: {improved_metrics['win_rate'][-1]/max(baseline_metrics['win_rate'][-1], 1e-6):.2f}x"
    )
    ax6.text(0.1, 0.7, summary_text, fontsize=10, va='top', linespacing=1.5)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细的统计数据到CSV
    stats_df = pd.DataFrame({
        'Metric': ['Avg Reward', 'Avg Survival', 'Win Rate', 'Avg Steps'],
        'Improved': [
            np.mean(improved_metrics['episode_rewards'][-final_window:]),
            np.mean(improved_metrics['red_survival'][-final_window:]),
            improved_metrics['win_rate'][-1],
            np.mean(improved_metrics['steps'][-final_window:])
        ],
        'Baseline': [
            np.mean(baseline_metrics['episode_rewards'][-final_window:]),
            np.mean(baseline_metrics['red_survival'][-final_window:]),
            baseline_metrics['win_rate'][-1],
            np.mean(baseline_metrics['steps'][-final_window:])
        ]
    })
    stats_df.to_csv(os.path.join(save_dir, 'comparison_stats.csv'), index=False)

def main():
    """主函数"""
    # 实验配置
    config = {
        'num_episodes': 2,
        'max_steps': 200,
        'num_red': 3,
        'num_blue': 3,
        'state_dim': 9,
        'action_dim': 2,
        'hidden_dim': 256,
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 512,
        'buffer_size': int(1e6),
        'heterogeneous': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("comparison_results", timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存实验配置
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\n{'='*50}")
    print("Starting comparison experiment...")
    print(f"Results will be saved to: {results_dir}")
    print(f"{'='*50}\n")
    
    # 设置随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    try:
        # 运行改进版MADDPG
        print("\nTraining Improved MADDPG...")
        improved_metrics = run_experiment(config, agent_type='improved', seed=seed)
        
        # 运行基础版MADDPG
        print("\nTraining Baseline MADDPG...")
        baseline_metrics = run_experiment(config, agent_type='baseline', seed=seed)
        
        # 保存原始数据
        improved_df = pd.DataFrame(improved_metrics)
        baseline_df = pd.DataFrame(baseline_metrics)
        improved_df.to_csv(os.path.join(results_dir, 'improved_metrics.csv'), index=False)
        baseline_df.to_csv(os.path.join(results_dir, 'baseline_metrics.csv'), index=False)
        
        # 绘制对比图
        print("\nGenerating comparison plots...")
        plot_comparison(improved_metrics, baseline_metrics, results_dir)
        
        print(f"\nExperiment completed successfully!")
        print(f"All results have been saved to: {results_dir}")
        print(f"{'='*50}\n")
        
    except Exception as e:
        print(f"\nError occurred during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 确保所有图表都被关闭
        plt.close('all')

if __name__ == "__main__":
    main()