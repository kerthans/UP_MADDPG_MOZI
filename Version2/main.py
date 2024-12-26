# main.py

from concurrent.futures import ThreadPoolExecutor
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
import torch.cuda as cuda
from torch.cuda.amp import autocast, GradScaler
import warnings
from typing import Dict, List, Optional

# 初始化colorama用于彩色终端输出
init(autoreset=True)

def plot_training_curves(results_dir: str, metrics: Dict):
    """绘制训练过程的各项指标曲线"""
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Training Metrics Dashboard', fontsize=16)  # 使用英文标题
    
    # 总奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(metrics['rewards_per_episode'], label='Reward per Episode', alpha=0.6, color='blue')
    ax1.plot(metrics['moving_avg_rewards'], label='Moving Average', linewidth=2, color='red')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True)
    ax1.legend()

    # 红蓝方奖励曲线
    ax2 = axes[0, 1]
    ax2.plot(metrics['red_rewards'], label='Red Team', color='red', alpha=0.6)
    ax2.plot(metrics['blue_rewards'], label='Blue Team', color='blue', alpha=0.6)
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Team Rewards')
    ax2.set_title('Red vs Blue Team Rewards')
    ax2.grid(True)
    ax2.legend()

    # 存活率曲线
    ax3 = axes[1, 0]
    window = 100
    red_survival = pd.Series(metrics['red_survival']).rolling(window).mean()
    blue_survival = pd.Series(metrics['blue_survival']).rolling(window).mean()
    ax3.plot(red_survival, label='Red Survival', color='red')
    ax3.plot(blue_survival, label='Blue Survival', color='blue')
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Survival Rate')
    ax3.set_title(f'Team Survival Rate (Window: {window})')
    ax3.grid(True)
    ax3.legend()

    # 能量分布
    ax4 = axes[1, 1]
    ax4.plot(metrics['red_energy'], label='Red Energy', color='red', alpha=0.6)
    ax4.plot(metrics['blue_energy'], label='Blue Energy', color='blue', alpha=0.6)
    ax4.set_xlabel('Episodes')
    ax4.set_ylabel('Average Energy')
    ax4.set_title('Team Energy Changes')
    ax4.grid(True)
    ax4.legend()

    # 步数分布
    ax5 = axes[2, 0]
    sns.histplot(data=metrics['steps_per_episode'], bins=30, ax=ax5, color='purple')
    ax5.set_xlabel('Steps')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Episode Steps Distribution')

    # 噪声衰减曲线
    ax6 = axes[2, 1]
    ax6.plot(metrics['noise_values'], label='Exploration Noise', color='orange')
    ax6.set_xlabel('Episodes')
    ax6.set_ylabel('Noise Value')
    ax6.set_title('Exploration Noise Decay')
    ax6.grid(True)
    ax6.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_evaluation_report(results_dir: str, metrics: Dict, params: Dict):
    """
    保存评估报告
    
    Args:
        results_dir: 结果保存目录
        metrics: 训练指标数据
        params: 训练参数配置
    """
    # 处理空数据情况
    def safe_mean(data):
        if not data:
            return 0.0
        return float(np.mean(data))
    
    def safe_max(data):
        if not data:
            return 0.0
        return float(max(data))
    
    def safe_min(data):
        if not data:
            return 0.0
        return float(min(data))
    
    def safe_mean_last_n(data, n=100):
        if not data:
            return 0.0
        return float(np.mean(data[-n:]))

    # 创建参数的深拷贝以避免修改原始数据
    params_copy = params.copy()
    
    # 处理不可JSON序列化的类型
    if 'device' in params_copy:
        params_copy['device'] = str(params_copy['device'])

    report = {
        'training_time': str(datetime.now()),
        'parameters': params_copy,
        'metrics': {
            'total_episodes': len(metrics['rewards_per_episode']),
            'average_reward': safe_mean(metrics['rewards_per_episode']),
            'max_reward': safe_max(metrics['rewards_per_episode']),
            'min_reward': safe_min(metrics['rewards_per_episode']),
            'red_performance': {
                'avg_reward': safe_mean(metrics['red_rewards']),
                'avg_survival': safe_mean(metrics['red_survival']) * 100,
                'avg_energy': safe_mean(metrics['red_energy'])
            },
            'blue_performance': {
                'avg_reward': safe_mean(metrics['blue_rewards']),
                'avg_survival': safe_mean(metrics['blue_survival']) * 100,
                'avg_energy': safe_mean(metrics['blue_energy'])
            },
            'average_steps': safe_mean(metrics['steps_per_episode']),
            'final_100_episodes': {
                'avg_reward': safe_mean_last_n(metrics['rewards_per_episode']),
                'red_survival': safe_mean_last_n(metrics['red_survival']) * 100,
                'blue_survival': safe_mean_last_n(metrics['blue_survival']) * 100,
                'avg_steps': safe_mean_last_n(metrics['steps_per_episode'])
            }
        }
    }

    # 保存详细指标数据
    metrics_df = pd.DataFrame({
        'Episode': range(len(metrics['rewards_per_episode'])),
        'Total_Reward': metrics['rewards_per_episode'],
        'Red_Reward': metrics['red_rewards'],
        'Blue_Reward': metrics['blue_rewards'],
        'Red_Survival': metrics['red_survival'],
        'Blue_Survival': metrics['blue_survival'],
        'Red_Energy': metrics['red_energy'],
        'Blue_Energy': metrics['blue_energy'],
        'Steps': metrics['steps_per_episode'],
        'Noise': metrics['noise_values']
    })
    metrics_df.to_csv(os.path.join(results_dir, 'detailed_metrics.csv'), index=False)

    # 保存评估报告
    with open(os.path.join(results_dir, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return report

def print_evaluation_report(report: Dict):
    """
    打印评估报告
    
    Args:
        report: 评估报告字典
    """
    print("\n" + "="*50)
    print(Fore.GREEN + "训练评估报告")
    print("="*50)
    print(f"训练完成时间: {report['training_time']}")
    
    print("\n整体表现:")
    metrics = report['metrics']
    print(f"- 总回合数: {metrics['total_episodes']}")
    print(f"- 平均奖励: {metrics['average_reward']:.2f}")
    print(f"- 最大奖励: {metrics['max_reward']:.2f}")
    print(f"- 最小奖励: {metrics['min_reward']:.2f}")
    print(f"- 平均步数: {metrics['average_steps']:.2f}")
    
    print("\n红方表现:")
    red_perf = metrics['red_performance']
    print(f"- 平均奖励: {red_perf['avg_reward']:.2f}")
    print(f"- 平均存活率: {red_perf['avg_survival']:.2f}%")
    print(f"- 平均能量: {red_perf['avg_energy']:.2f}")
    
    print("\n蓝方表现:")
    blue_perf = metrics['blue_performance']
    print(f"- 平均奖励: {blue_perf['avg_reward']:.2f}")
    print(f"- 平均存活率: {blue_perf['avg_survival']:.2f}%")
    print(f"- 平均能量: {blue_perf['avg_energy']:.2f}")
    
    print("\n最后100回合表现:")
    final = metrics['final_100_episodes']
    print(f"- 平均奖励: {final['avg_reward']:.2f}")
    print(f"- 红方存活率: {final['red_survival']:.2f}%")
    print(f"- 蓝方存活率: {final['blue_survival']:.2f}%")
    print(f"- 平均步数: {final['avg_steps']:.2f}")
    
    print("\n训练参数配置:")
    for key, value in report['parameters'].items():
        print(f"- {key}: {value}")
    print("="*50)

def set_random_seeds(seed: int):
    """
    设置随机种子以确保实验可重复性
    
    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    """主函数"""
    # 训练参数配置
    params = {
        'num_episodes': 2000,          # 训练回合数
        'max_steps': 200,              # 每回合最大步数
        'num_red': 3,                  # 红方智能体数量
        'num_blue': 3,                 # 蓝方智能体数量
        'state_dim': 9,               # 状态空间维度
        'action_dim': 2,              # 动作空间维度
        'hidden_dim': 256,            # 隐藏层维度
        'lr': 3e-4,                   # 学习率
        'gamma': 0.99,                # 折扣因子
        'tau': 0.005,                 # 目标网络软更新系数
        'initial_noise': 0.3,         # 初始探索噪声
        'noise_decay': 0.9995,        # 噪声衰减率
        'min_noise': 0.01,            # 最小噪声值
        'batch_size': 512,            # 批次大小
        'buffer_size': int(1e6),      # 经验回放缓冲区大小
        'save_freq': 100,             # 模型保存频率
        'grad_clip': 0.5,             # 梯度裁剪阈值
        'update_freq': 2,             # 网络更新频率
        'seed': 42,                   # 随机种子
        'heterogeneous': True,        # 是否启用异构智能体
        'use_curiosity': True,        # 是否使用好奇心机制
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # 训练设备
    }
    
    # 设置随机种子
    set_random_seeds(params['seed'])
    
    # 创建结果保存目录
    results_dir = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化智能体类型配置
    red_agent_types = ['scout', 'fighter', 'bomber'] if params['heterogeneous'] else ['fighter'] * params['num_red']
    blue_agent_types = ['fighter', 'fighter', 'fighter']  # 蓝方保持同质
    
    # 创建环境
    env = CombatEnv(
        num_red=params['num_red'],
        num_blue=params['num_blue'],
        state_dim=params['state_dim'],
        action_dim=params['action_dim'],
        max_steps=params['max_steps'],
        red_agent_types=red_agent_types,
        blue_agent_types=blue_agent_types,
        heterogeneous=params['heterogeneous']
    )
    
    # 创建MADDPG智能体
    maddpg = MADDPG(
        num_agents=params['num_red'] + params['num_blue'],
        state_dim=params['state_dim'],
        action_dim=params['action_dim'],
        lr=params['lr'],
        gamma=params['gamma'],
        tau=params['tau'],
        buffer_size=params['buffer_size'],
        batch_size=params['batch_size'],
        hidden_dim=params['hidden_dim'],
        agent_types=red_agent_types + blue_agent_types,
        device=params['device']  # 直接传递device参数
    )
    
    # 初始化训练指标记录
    metrics = {
        'rewards_per_episode': [],    # 总奖励
        'red_rewards': [],            # 红方奖励
        'blue_rewards': [],           # 蓝方奖励
        'red_survival': [],           # 红方存活率
        'blue_survival': [],          # 蓝方存活率
        'red_energy': [],             # 红方能量
        'blue_energy': [],            # 蓝方能量
        'steps_per_episode': [],      # 步数
        'noise_values': [],           # 噪声值
        'moving_avg_rewards': [0]     # 移动平均奖励
    }

    print(Fore.CYAN + "\n开始训练...\n")
    progress_bar = tqdm(range(params['num_episodes']))
    current_noise = params['initial_noise']
    update_count = 0
    scaler = GradScaler()  # 用于混合精度训练
    
    try:
        for episode in progress_bar:
            obs = env.reset()
            episode_reward = 0
            red_reward = 0
            blue_reward = 0
            step_count = 0
            
            # 记录每回合初始状态
            initial_red_alive = env.red_alive.copy()
            initial_blue_alive = env.blue_alive.copy()
            
            while step_count < params['max_steps']:
                # 将观察值分割为每个智能体的观察
                per_agent_obs = []
                for i in range(maddpg.num_agents):
                    agent_obs = obs[i * params['state_dim']:(i + 1) * params['state_dim']]
                    per_agent_obs.append(torch.FloatTensor(agent_obs).to(params['device']))

                # 使用自动混合精度加速训练
                with autocast():
                    # 选择动作
                    actions = maddpg.select_actions(per_agent_obs, noise=current_noise)
                    
                    # 执行动作
                    next_obs, rewards, done, info = env.step(actions)
                    
                    # 分别计算红蓝双方奖励
                    red_rewards = rewards[:params['num_red']]
                    blue_rewards = rewards[params['num_red']:]
                    red_reward += sum(red_rewards)
                    blue_reward += sum(blue_rewards)
                    
                    # 预处理数据
                    states = np.concatenate([o.cpu().numpy() for o in per_agent_obs])
                    actions_np = np.array(actions)
                    rewards_np = np.array(rewards)
                    next_states_np = np.array(next_obs)
                    
                    # 正确的方式添加经验到回放缓冲区
                    maddpg.replay_buffer.add(
                        state=states,
                        action=actions_np,
                        reward=rewards_np,
                        next_state=next_states_np,
                        done=done
                    )
                    
                    # 网络更新
                    if maddpg.replay_buffer.size() >= params['batch_size']:
                        update_count += 1
                        if update_count % params['update_freq'] == 0:
                            maddpg.update()
                
                episode_reward += sum(rewards)
                obs = next_obs
                step_count += 1
                
                if done:
                    break
            
            # 计算存活率和平均能量
            red_survival_rate = np.mean(env.red_alive) / np.mean(initial_red_alive) if np.mean(initial_red_alive) > 0 else 0
            blue_survival_rate = np.mean(env.blue_alive) / np.mean(initial_blue_alive) if np.mean(initial_blue_alive) > 0 else 0
            avg_red_energy = np.mean(env.red_energy[env.red_alive]) if np.any(env.red_alive) else 0
            avg_blue_energy = np.mean(env.blue_energy[env.blue_alive]) if np.any(env.blue_alive) else 0
            
            # 更新探索噪声
            current_noise = max(params['min_noise'], current_noise * params['noise_decay'])
            
            # 更新指标
            metrics['rewards_per_episode'].append(episode_reward)
            metrics['red_rewards'].append(red_reward)
            metrics['blue_rewards'].append(blue_reward)
            metrics['red_survival'].append(red_survival_rate)
            metrics['blue_survival'].append(blue_survival_rate)
            metrics['red_energy'].append(avg_red_energy)
            metrics['blue_energy'].append(avg_blue_energy)
            metrics['steps_per_episode'].append(step_count)
            metrics['noise_values'].append(current_noise)
            
            # 计算移动平均
            window_size = min(100, episode + 1)
            metrics['moving_avg_rewards'].append(
                np.mean(metrics['rewards_per_episode'][-window_size:])
            )
            
            # 更新进度条信息
            progress_bar.set_postfix({
                'Total': f"{episode_reward:.2f}",
                'Red': f"{red_reward:.2f}",
                'Blue': f"{blue_reward:.2f}",
                'RedSurv': f"{red_survival_rate:.2%}",
                'BlueSurv': f"{blue_survival_rate:.2%}",
                'Noise': f"{current_noise:.3f}"
            })
            
            # 定期保存模型和绘制曲线
            if (episode + 1) % params['save_freq'] == 0:
                save_path = os.path.join(results_dir, f'model_episode_{episode+1}.pt')
                maddpg.save(save_path)
                plot_training_curves(results_dir, metrics)
                
                # 打印阶段性评估信息
                print(f"\n{Fore.YELLOW}====== 训练进度: {episode + 1}/{params['num_episodes']} ======")
                print(f"最近100回合平均奖励: {metrics['moving_avg_rewards'][-1]:.2f}")
                print(f"红方存活率: {np.mean(metrics['red_survival'][-100:]):.2%}")
                print(f"蓝方存活率: {np.mean(metrics['blue_survival'][-100:]):.2%}")
                print(f"当前探索噪声: {current_noise:.3f}")
                print(f"模型已保存至: {save_path}\n")
                
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n训练被用户中断。正在保存当前进度...")
    
    except Exception as e:
        print(Fore.RED + f"\n训练过程出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 保存最终模型和训练报告
        try:
            final_model_path = os.path.join(results_dir, 'final_model.pt')
            maddpg.save(final_model_path)
            report = save_evaluation_report(results_dir, metrics, params)
            plot_training_curves(results_dir, metrics)
            print_evaluation_report(report)
            print(Fore.GREEN + f"\n训练完成！所有结果已保存至: {results_dir}")
            
        except Exception as e:
            print(Fore.RED + f"保存最终结果时出错: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    # 设置警告过滤
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # 设置torch多线程
    if torch.cuda.is_available():
        torch.set_num_threads(4)
    
    # 启动训练
    main()