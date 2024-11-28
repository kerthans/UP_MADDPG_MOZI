# main.py

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from agents.maddpg import MADDPG
from env.combat_env import CombatEnv
import os
import json
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init(autoreset=True)

def plot_training_curves(results_dir, metrics):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Metrics Dashboard', fontsize=16)
    
    # 奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(metrics['rewards_per_episode'], label='Rewards', alpha=0.6, color='blue')
    ax1.plot(metrics['moving_avg_rewards'], label='Moving Average', linewidth=2, color='red')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Rewards per Episode')
    ax1.grid(True)
    ax1.legend()

    # 成功率曲线
    ax2 = axes[0, 1]
    window = 100
    success_rate = [sum(metrics['success_per_episode'][max(0, i-window):i])/min(i, window) 
                   for i in range(1, len(metrics['success_per_episode'])+1)]
    ax2.plot(success_rate, label='Success Rate', color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title(f'Success Rate (Window Size: {window})')
    ax2.grid(True)
    ax2.legend()

    # 步数分布
    ax3 = axes[1, 0]
    sns.histplot(data=metrics['steps_per_episode'], bins=30, ax=ax3, color='purple')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Steps Distribution')

    # 噪声衰减曲线
    ax4 = axes[1, 1]
    ax4.plot(metrics['noise_values'], label='Exploration Noise', color='orange')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Noise Value')
    ax4.set_title('Exploration Noise Decay')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_evaluation_report(results_dir, metrics, params):
    """保存评估报告"""
    report = {
        'training_time': str(datetime.now()),
        'parameters': params,
        'metrics': {
            'total_episodes': len(metrics['rewards_per_episode']),
            'average_reward': float(np.mean(metrics['rewards_per_episode'])),
            'max_reward': float(max(metrics['rewards_per_episode'])),
            'min_reward': float(min(metrics['rewards_per_episode'])),
            'success_rate': float(np.mean(metrics['success_per_episode']) * 100),
            'average_steps': float(np.mean(metrics['steps_per_episode'])),
            'final_100_episodes': {
                'avg_reward': float(np.mean(metrics['rewards_per_episode'][-100:])),
                'success_rate': float(np.mean(metrics['success_per_episode'][-100:]) * 100),
                'avg_steps': float(np.mean(metrics['steps_per_episode'][-100:]))
            }
        }
    }

    # 保存详细指标
    metrics_df = pd.DataFrame({
        'Episode': range(len(metrics['rewards_per_episode'])),
        'Reward': metrics['rewards_per_episode'],
        'Steps': metrics['steps_per_episode'],
        'Success': metrics['success_per_episode'],
        'Noise': metrics['noise_values']
    })
    metrics_df.to_csv(os.path.join(results_dir, 'detailed_metrics.csv'), index=False)

    # 保存报告
    with open(os.path.join(results_dir, 'evaluation_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

    return report

def print_evaluation_report(report):
    """打印评估报告"""
    print("\n" + "="*50)
    print(Fore.GREEN + "Training Evaluation Report")
    print("="*50)
    print(f"Training completed at: {report['training_time']}")
    
    print("\nOverall Performance:")
    metrics = report['metrics']
    print(f"- Total Episodes: {metrics['total_episodes']}")
    print(f"- Average Reward: {metrics['average_reward']:.2f}")
    print(f"- Max Reward: {metrics['max_reward']:.2f}")
    print(f"- Min Reward: {metrics['min_reward']:.2f}")
    print(f"- Overall Success Rate: {metrics['success_rate']:.2f}%")
    print(f"- Average Steps per Episode: {metrics['average_steps']:.2f}")
    
    print("\nFinal 100 Episodes Performance:")
    final_metrics = metrics['final_100_episodes']
    print(f"- Average Reward: {final_metrics['avg_reward']:.2f}")
    print(f"- Success Rate: {final_metrics['success_rate']:.2f}%")
    print(f"- Average Steps: {final_metrics['avg_steps']:.2f}")
    
    print("\nTraining Parameters:")
    for key, value in report['parameters'].items():
        print(f"- {key}: {value}")
    print("="*50)

def set_random_seeds(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # 训练参数
    params = {
        'num_episodes': 500,     # 增加训练轮数
        'max_steps': 200,
        'num_red': 3,
        'num_blue': 3,
        'state_dim': 5,
        'action_dim': 2,
        'lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'initial_noise': 0.3,     # 初始探索噪声
        'noise_decay': 0.9995,    # 噪声衰减率
        'min_noise': 0.01,        # 最小噪声
        'hidden_dim': 256,
        'batch_size': 256,        # 增加批次大小
        'buffer_size': int(1e6),
        'max_velocity': 5.0,
        'seed': 42                # 随机种子
    }
    
    # 设置随机种子
    set_random_seeds(params['seed'])
    
    # 创建结果目录
    results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化环境
    env = CombatEnv(
        num_red=params['num_red'],
        num_blue=params['num_blue'],
        state_dim=params['state_dim'],
        action_dim=params['action_dim'],
        max_velocity=params['max_velocity'],
        max_steps=params['max_steps']
    )
    
    # 初始化MADDPG
    maddpg = MADDPG(
        num_agents=params['num_red'] + params['num_blue'],
        state_dim=params['state_dim'],
        action_dim=params['action_dim'],
        lr=params['lr'],
        gamma=params['gamma'],
        tau=params['tau'],
        buffer_size=params['buffer_size'],
        batch_size=params['batch_size'],
        hidden_dim=params['hidden_dim']
    )

    # 训练指标记录
    metrics = {
        'rewards_per_episode': [],
        'steps_per_episode': [],
        'success_per_episode': [],
        'noise_values': [],
        'moving_avg_rewards': []
    }

    # 训练循环
    print(Fore.CYAN + "\nStarting training...\n")
    progress_bar = tqdm(range(params['num_episodes']), desc="Training Progress")
    current_noise = params['initial_noise']
    
    try:
        for episode in progress_bar:
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            
            while step_count < params['max_steps']:
                # 获取每个智能体的观察
                per_agent_obs = []
                for i in range(maddpg.num_agents):
                    agent_obs = obs[i * params['state_dim']:(i + 1) * params['state_dim']]
                    per_agent_obs.append(agent_obs)
                
                # 选择动作
                actions = maddpg.select_actions(per_agent_obs, noise=current_noise)
                
                # 执行动作
                next_obs, rewards, done, _ = env.step(actions)
                
                # 存储经验
                maddpg.replay_buffer.add((per_agent_obs, actions, rewards, next_obs, done))
                
                # 更新神经网络
                if maddpg.replay_buffer.size() > params['batch_size']:
                    maddpg.update()
                
                episode_reward += sum(rewards)
                obs = next_obs
                step_count += 1
                
                if done:
                    break
            
            # 更新噪声
            current_noise = max(
                params['min_noise'],
                current_noise * params['noise_decay']
            )
            
            # 记录指标
            metrics['rewards_per_episode'].append(episode_reward)
            metrics['steps_per_episode'].append(step_count)
            metrics['success_per_episode'].append(done)
            metrics['noise_values'].append(current_noise)
            
            # 计算移动平均
            window_size = 100
            metrics['moving_avg_rewards'] = pd.Series(metrics['rewards_per_episode']).rolling(
                window=window_size, min_periods=1).mean().tolist()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Reward': f"{episode_reward:.2f}",
                'AvgReward': f"{metrics['moving_avg_rewards'][-1]:.2f}",
                'Steps': step_count,
                'Success': done,
                'Noise': f"{current_noise:.3f}"
            })
            
            # 定期保存模型和绘制图表
            if (episode + 1) % 100 == 0:
                maddpg.save(os.path.join(results_dir, f'model_episode_{episode+1}.pt'))
                plot_training_curves(results_dir, metrics)
                
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\nTraining interrupted by user. Saving current progress...")
    
    finally:
        # 保存最终模型
        maddpg.save(os.path.join(results_dir, 'final_model.pt'))
        
        # 生成并保存评估报告
        report = save_evaluation_report(results_dir, metrics, params)
        
        # 绘制最终的训练曲线
        plot_training_curves(results_dir, metrics)
        
        # 打印评估报告
        print_evaluation_report(report)
        
        print(Fore.GREEN + f"\nTraining completed. Results saved in: {results_dir}")

if __name__ == "__main__":
    main()