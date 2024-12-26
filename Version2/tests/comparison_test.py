# tests/comparison_test.py

import json
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import pandas as pd
from env.combat_env import CombatEnv
from agents.maddpg import MADDPG
from agents.baseline_maddpg import BaselineMADDPG

class ComparisonTest:
    """MADDPG对比测试类"""
    def __init__(self, test_episodes=1000, eval_interval=50):
        self.test_episodes = test_episodes
        self.eval_interval = eval_interval
        self.results_dir = os.path.join("results", "comparison_tests", 
                                      datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 环境参数
        self.params = {
            'num_episodes': test_episodes,
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
            'initial_noise': 0.3,
            'noise_decay': 0.9995,
            'min_noise': 0.01
        }
        
        # 创建环境和智能体
        self._setup_environment()
        self._setup_agents()
        
        # 初始化指标记录
        self.metrics = {
            'improved': self._init_metrics(),
            'baseline': self._init_metrics()
        }

    def _init_metrics(self):
        """初始化性能指标"""
        return {
            'episode_rewards': [],
            'red_rewards': [],
            'blue_rewards': [],
            'red_survival': [],
            'blue_survival': [],
            'steps_per_episode': [],
            'win_rates': [],
            'team_coordination': [],
            'energy_efficiency': [],
            'completion_time': [],
            'tactical_advantage': []
        }

    def _setup_environment(self):
        """设置环境"""
        self.env_improved = CombatEnv(
            num_red=self.params['num_red'],
            num_blue=self.params['num_blue'],
            state_dim=self.params['state_dim'],
            action_dim=self.params['action_dim'],
            max_steps=self.params['max_steps'],
            heterogeneous=True
        )
        
        self.env_baseline = CombatEnv(
            num_red=self.params['num_red'],
            num_blue=self.params['num_blue'],
            state_dim=self.params['state_dim'],
            action_dim=self.params['action_dim'],
            max_steps=self.params['max_steps'],
            heterogeneous=False
        )

    def _setup_agents(self):
        """设置智能体"""
        # 改进版MADDPG
        self.improved_maddpg = MADDPG(
            num_agents=self.params['num_red'] + self.params['num_blue'],
            state_dim=self.params['state_dim'],
            action_dim=self.params['action_dim'],
            lr=self.params['lr'],
            gamma=self.params['gamma'],
            tau=self.params['tau'],
            buffer_size=self.params['buffer_size'],
            batch_size=self.params['batch_size'],
            hidden_dim=self.params['hidden_dim'],
            agent_types=['scout', 'fighter', 'bomber'] + ['fighter'] * self.params['num_blue']
        )
        
        # 基础版MADDPG
        self.baseline_maddpg = BaselineMADDPG(
            num_agents=self.params['num_red'] + self.params['num_blue'],
            state_dim=self.params['state_dim'],
            action_dim=self.params['action_dim'],
            lr=self.params['lr'],
            gamma=self.params['gamma'],
            tau=self.params['tau'],
            buffer_size=self.params['buffer_size'],
            batch_size=self.params['batch_size'],
            hidden_dim=self.params['hidden_dim']
        )

    def _calculate_tactical_metrics(self, env, agent_positions, agent_velocities):
        """计算战术性能指标"""
        # 计算队伍协调性（基于队伍间距）
        red_positions = agent_positions[:self.params['num_red']]
        red_center = np.mean(red_positions, axis=0)
        team_coordination = -np.mean([np.linalg.norm(pos - red_center) for pos in red_positions])
        
        # 计算能量效率
        red_velocities = agent_velocities[:self.params['num_red']]
        energy_efficiency = -np.mean([np.sum(np.square(vel)) for vel in red_velocities])
        
        # 计算战术优势
        tactical_advantage = self._calculate_position_advantage(env, red_positions)
        
        return team_coordination, energy_efficiency, tactical_advantage

    def _calculate_position_advantage(self, env, positions):
        """计算位置优势分数"""
        if not hasattr(env, 'blue_positions') or not hasattr(env, 'blue_alive'):
            return 0.0
            
        blue_positions = env.blue_positions[env.blue_alive]
        if len(blue_positions) == 0:
            return 0.0
            
        advantages = []
        for pos in positions:
            # 计算与敌方的距离
            distances = [np.linalg.norm(pos - blue_pos) for blue_pos in blue_positions]
            min_distance = min(distances)
            
            # 计算攻击角度优势
            if hasattr(env, 'red_velocities'):
                velocity = env.red_velocities[0]  # 使用第一个智能体的速度作为参考
                if np.linalg.norm(velocity) > 0:
                    direction = velocity / np.linalg.norm(velocity)
                    to_enemy = blue_positions[np.argmin(distances)] - pos
                    if np.linalg.norm(to_enemy) > 0:
                        to_enemy = to_enemy / np.linalg.norm(to_enemy)
                        angle = np.arccos(np.clip(np.dot(direction, to_enemy), -1.0, 1.0))
                        angle_advantage = np.cos(2 * angle)  # 优化攻击角度评分
                    else:
                        angle_advantage = 0
                else:
                    angle_advantage = 0
            else:
                angle_advantage = 0
            
            # 计算高地优势（如果环境支持）
            height_advantage = 0
            if hasattr(env, 'get_height'):
                height_advantage = env.get_height(pos)
            
            # 综合优势评分
            position_score = (
                1.0 / (1 + min_distance) +  # 距离优势
                0.5 * angle_advantage +     # 角度优势
                0.3 * height_advantage      # 高地优势
            )
            advantages.append(position_score)
            
        return np.mean(advantages)

    def run_episode(self, env, agent, metrics, noise):
        """运行单个训练回合"""
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        
        # 记录初始状态
        initial_red_alive = env.red_alive.copy()
        initial_blue_alive = env.blue_alive.copy()
        
        while step_count < self.params['max_steps']:
            # 分割观察值
            per_agent_obs = []
            for i in range(agent.num_agents):
                agent_obs = obs[i * self.params['state_dim']:(i + 1) * self.params['state_dim']]
                per_agent_obs.append(agent_obs)
            
            # 选择动作
            actions = agent.select_actions(per_agent_obs, noise=noise)
            next_obs, rewards, done, info = env.step(actions)
            
            # 记录数据
            episode_reward += sum(rewards)
            
            # 存储经验
            if hasattr(agent, 'replay_buffer'):
                state = np.concatenate(per_agent_obs)
                next_state = np.array(next_obs)
                agent.replay_buffer.add(state, np.array(actions), 
                                     np.array(rewards), next_state, done)
            
            # 更新智能体
            if agent.replay_buffer.size() >= agent.batch_size:
                agent.update()
            
            obs = next_obs
            step_count += 1
            
            if done:
                break
        
        # 计算性能指标
        red_survival_rate = np.mean(env.red_alive) / np.mean(initial_red_alive) if np.mean(initial_red_alive) > 0 else 0
        blue_survival_rate = np.mean(env.blue_alive) / np.mean(initial_blue_alive) if np.mean(initial_blue_alive) > 0 else 0
        
        # 计算战术指标
        team_coord, energy_eff, tact_adv = self._calculate_tactical_metrics(
            env, 
            env.red_positions,
            env.red_velocities if hasattr(env, 'red_velocities') else np.zeros_like(env.red_positions)
        )
        
        return {
            'episode_reward': episode_reward,
            'red_reward': sum(rewards[:self.params['num_red']]),
            'blue_reward': sum(rewards[self.params['num_red']:]),
            'red_survival': red_survival_rate,
            'blue_survival': blue_survival_rate,
            'steps': step_count,
            'team_coordination': team_coord,
            'energy_efficiency': energy_eff,
            'tactical_advantage': tact_adv
        }

    def run_comparison(self):
        """运行对比测试"""
        print("\n开始对比测试...")
        progress_bar = tqdm(range(self.test_episodes))
        current_noise = self.params['initial_noise']
        
        for episode in progress_bar:
            # 运行改进版MADDPG
            improved_results = self.run_episode(
                self.env_improved,
                self.improved_maddpg,
                self.metrics['improved'],
                current_noise
            )
            
            # 运行基准版MADDPG
            baseline_results = self.run_episode(
                self.env_baseline,
                self.baseline_maddpg,
                self.metrics['baseline'],
                current_noise
            )
            
            # 更新指标
            for version, results in [('improved', improved_results), 
                                   ('baseline', baseline_results)]:
                metrics = self.metrics[version]
                metrics['episode_rewards'].append(results['episode_reward'])
                metrics['red_rewards'].append(results['red_reward'])
                metrics['blue_rewards'].append(results['blue_reward'])
                metrics['red_survival'].append(results['red_survival'])
                metrics['blue_survival'].append(results['blue_survival'])
                metrics['steps_per_episode'].append(results['steps'])
                metrics['team_coordination'].append(results['team_coordination'])
                metrics['energy_efficiency'].append(results['energy_efficiency'])
                metrics['tactical_advantage'].append(results['tactical_advantage'])
            
            # 更新噪声
            current_noise = max(
                self.params['min_noise'],
                current_noise * self.params['noise_decay']
            )
            
            # 更新进度条信息
            if episode % 10 == 0:
                improved_avg = np.mean(self.metrics['improved']['episode_rewards'][-100:])
                baseline_avg = np.mean(self.metrics['baseline']['episode_rewards'][-100:])
                progress_bar.set_postfix({
                    'Improved': f"{improved_avg:.2f}",
                    'Baseline': f"{baseline_avg:.2f}",
                    'Noise': f"{current_noise:.3f}"
                })
            
            # 定期保存结果
            if episode % self.eval_interval == 0:
                self.save_comparison_results()
                self.plot_comparison_curves()

    def save_comparison_results(self):
        """保存对比结果"""
        results = {
            'parameters': self.params,
            'metrics': {
                'improved': {k: np.array(v).tolist() for k, v in self.metrics['improved'].items()},
                'baseline': {k: np.array(v).tolist() for k, v in self.metrics['baseline'].items()}
            }
        }
        
        # 保存为JSON文件
        import json
        with open(os.path.join(self.results_dir, 'comparison_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存为CSV文件以便进一步分析
        for version in ['improved', 'baseline']:
            df = pd.DataFrame(self.metrics[version])
            df.to_csv(os.path.join(self.results_dir, f'{version}_metrics.csv'), index=False)

    def plot_comparison_curves(self):
        """绘制对比曲线"""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 总奖励对比
        ax1 = plt.subplot(3, 2, 1)
        self._plot_metric_comparison(ax1, 'episode_rewards', 
                                   '回合总奖励', window=50)
        
        # 2. 红蓝方奖励对比
        ax2 = plt.subplot(3, 2, 2)
        for version in ['improved', 'baseline']:
            data = pd.DataFrame({
                'Red': self.metrics[version]['red_rewards'],
                'Blue': self.metrics[version]['blue_rewards']
            })
            data = data.rolling(50).mean()
            ax2.plot(data['Red'], label=f'{version.capitalize()} Red')
            ax2.plot(data['Blue'], label=f'{version.capitalize()} Blue')
        ax2.set_title('红蓝方奖励对比')
        ax2.set_xlabel('回合数')
        ax2.set_ylabel('奖励值')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 存活率对比
        ax3 = plt.subplot(3, 2, 3)
        self._plot_metric_comparison(ax3, 'red_survival', 
                                   '红方存活率', window=50, percentage=True)
        
        # 4. 战术性能对比
        ax4 = plt.subplot(3, 2, 4)
        self._plot_metric_comparison(ax4, 'tactical_advantage',
                                   '战术优势评分', window=50)
        
        # 5. 团队协调性对比
        ax5 = plt.subplot(3, 2, 5)
        self._plot_metric_comparison(ax5, 'team_coordination',
                                   '团队协调性', window=50)
        
        # 6. 能量效率对比
        ax6 = plt.subplot(3, 2, 6)
        self._plot_metric_comparison(ax6, 'energy_efficiency',
                                   '能量效率', window=50)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_metric_comparison(self, ax, metric_name, title, window=50, percentage=False):
        """绘制指标对比曲线"""
        for version in ['improved', 'baseline']:
            data = pd.Series(self.metrics[version][metric_name])
            data = data.rolling(window).mean()
            if percentage:
                data = data * 100
            ax.plot(data, label=version.capitalize())
        ax.set_title(title)
        ax.set_xlabel('回合数')
        ax.set_ylabel('数值' + ('(%)' if percentage else ''))
        ax.legend()
        ax.grid(True)

    def generate_comparison_report(self):
        """生成对比测试报告"""
        report = {
            'test_parameters': self.params,
            'performance_comparison': {
                'average_rewards': {
                    'improved': np.mean(self.metrics['improved']['episode_rewards']),
                    'baseline': np.mean(self.metrics['baseline']['episode_rewards'])
                },
                'red_survival_rate': {
                    'improved': np.mean(self.metrics['improved']['red_survival']) * 100,
                    'baseline': np.mean(self.metrics['baseline']['red_survival']) * 100
                },
                'tactical_performance': {
                    'improved': np.mean(self.metrics['improved']['tactical_advantage']),
                    'baseline': np.mean(self.metrics['baseline']['tactical_advantage'])
                },
                'team_coordination': {
                    'improved': np.mean(self.metrics['improved']['team_coordination']),
                    'baseline': np.mean(self.metrics['baseline']['team_coordination'])
                },
                'energy_efficiency': {
                    'improved': np.mean(self.metrics['improved']['energy_efficiency']),
                    'baseline': np.mean(self.metrics['baseline']['energy_efficiency'])
                }
            }
        }
        
        # 计算改进百分比
        for metric, values in report['performance_comparison'].items():
            if values['baseline'] != 0:
                improvement = ((values['improved'] - values['baseline']) / 
                             abs(values['baseline'])) * 100
                values['improvement_percentage'] = improvement
            else:
                values['improvement_percentage'] = float('inf')
        
        # 保存报告
        with open(os.path.join(self.results_dir, 'comparison_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        return report

def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建并运行对比测试
    test = ComparisonTest(test_episodes=1000, eval_interval=50)
    test.run_comparison()
    
    # 生成报告
    report = test.generate_comparison_report()
    
    # 打印主要结果
    print("\n=== 对比测试报告 ===")
    for metric, values in report['performance_comparison'].items():
        print(f"\n{metric}:")
        print(f"  改进版: {values['improved']:.2f}")
        print(f"  基准版: {values['baseline']:.2f}")
        if 'improvement_percentage' in values:
            print(f"  改进幅度: {values['improvement_percentage']:.2f}%")

if __name__ == "__main__":
    main()
