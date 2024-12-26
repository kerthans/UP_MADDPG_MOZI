# tests/analysis_utils.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats # type: ignore

class PerformanceAnalyzer:
    """性能分析工具类"""
    
    def __init__(self, results_dir):
        self.results_dir = results_dir
        self.improved_data = pd.read_csv(os.path.join(results_dir, 'improved_metrics.csv'))
        self.baseline_data = pd.read_csv(os.path.join(results_dir, 'baseline_metrics.csv'))
        
    def plot_learning_curves(self):
        """绘制学习曲线的详细对比"""
        plt.style.use('seaborn')
        fig, axes = plt.subplots(3, 3, figsize=(20, 20))
        fig.suptitle('详细学习曲线对比', fontsize=16)
        
        # 1. 平滑奖励曲线对比
        self._plot_smoothed_comparison(
            axes[0, 0], 'episode_rewards', '平均回合奖励',
            window=50, ci=95
        )
        
        # 2. 红方vs蓝方胜率
        self._plot_win_rates(axes[0, 1])
        
        # 3. 能量效率分布
        self._plot_efficiency_distribution(axes[0, 2])
        
        # 4. 生存时间分布
        self._plot_survival_time_distribution(axes[1, 0])
        
        # 5. 战术性能雷达图
        self._plot_tactical_radar(axes[1, 1])
        
        # 6. 学习稳定性对比
        self._plot_stability_comparison(axes[1, 2])
        
        # 7. 智能体协作分析
        self._plot_cooperation_analysis(axes[2, 0])
        
        # 8. 任务完成效率
        self._plot_completion_efficiency(axes[2, 1])
        
        # 9. 性能提升趋势
        self._plot_improvement_trends(axes[2, 2])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'detailed_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_smoothed_comparison(self, ax, metric, title, window=50, ci=95):
        """绘制平滑对比曲线"""
        for label, data in [('Improved', self.improved_data), 
                           ('Baseline', self.baseline_data)]:
            sns.lineplot(data=data[metric].rolling(window).mean(),
                        label=label, ci=ci, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Value')
        ax.grid(True)
        
    def _plot_win_rates(self, ax):
        """绘制红蓝方胜率对比"""
        win_rates = {
            'Improved': {
                'Red': (self.improved_data['red_survival'] > 
                       self.improved_data['blue_survival']).mean(),
                'Blue': (self.improved_data['blue_survival'] > 
                        self.improved_data['red_survival']).mean()
            },
            'Baseline': {
                'Red': (self.baseline_data['red_survival'] > 
                       self.baseline_data['blue_survival']).mean(),
                'Blue': (self.baseline_data['blue_survival'] > 
                        self.baseline_data['red_survival']).mean()
            }
        }
        
        x = np.arange(2)
        width = 0.35
        
        ax.bar(x - width/2, [win_rates['Improved']['Red'], 
                            win_rates['Improved']['Blue']], 
               width, label='Improved')
        ax.bar(x + width/2, [win_rates['Baseline']['Red'], 
                            win_rates['Baseline']['Blue']], 
               width, label='Baseline')
        
        ax.set_title('红蓝方胜率对比')
        ax.set_xticks(x)
        ax.set_xticklabels(['Red', 'Blue'])
        ax.set_ylabel('Win Rate')
        ax.legend()
        
    def _plot_efficiency_distribution(self, ax):
        """绘制能量效率分布"""
        sns.kdeplot(data=self.improved_data['energy_efficiency'],
                   label='Improved', ax=ax)
        sns.kdeplot(data=self.baseline_data['energy_efficiency'],
                   label='Baseline', ax=ax)
        ax.set_title('能量效率分布')
        ax.set_xlabel('Energy Efficiency')
        ax.set_ylabel('Density')
        
    def _plot_survival_time_distribution(self, ax):
        """绘制生存时间分布"""
        sns.boxplot(data=[
            self.improved_data['steps_per_episode'],
            self.baseline_data['steps_per_episode']
        ], ax=ax)
        ax.set_xticklabels(['Improved', 'Baseline'])
        ax.set_title('生存时间分布')
        ax.set_ylabel('Steps')
        
    def _plot_tactical_radar(self, ax):
        """绘制战术性能雷达图"""
        metrics = ['tactical_advantage', 'team_coordination', 
                  'energy_efficiency', 'red_survival']
        
        improved_stats = [self.improved_data[m].mean() for m in metrics]
        baseline_stats = [self.baseline_data[m].mean() for m in metrics]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
        
        ax.plot(angles, improved_stats, 'o-', label='Improved')
        ax.plot(angles, baseline_stats, 'o-', label='Baseline')
        ax.set_xticks(angles)
        ax.set_xticklabels(metrics)
        ax.set_title('战术性能雷达图')
        ax.legend()
        
    def _plot_stability_comparison(self, ax):
        """绘制学习稳定性对比"""
        window = 50
        for label, data in [('Improved', self.improved_data), 
                           ('Baseline', self.baseline_data)]:
            rolling_std = data['episode_rewards'].rolling(window).std()
            ax.plot(rolling_std, label=label)
        ax.set_title('学习稳定性对比')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Standard Deviation')
        ax.legend()
        
    def _plot_cooperation_analysis(self, ax):
        """绘制智能体协作分析"""
        for label, data in [('Improved', self.improved_data), 
                           ('Baseline', self.baseline_data)]:
            cooperation_score = data['team_coordination'].rolling(50).mean()
            ax.plot(cooperation_score, label=label)
        ax.set_title('智能体协作分析')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Coordination Score')
        ax.legend()
        
    def _plot_completion_efficiency(self, ax):
        """绘制任务完成效率"""
        improved_efficiency = (self.improved_data['red_survival'] / 
                             self.improved_data['steps_per_episode'])
        baseline_efficiency = (self.baseline_data['red_survival'] / 
                             self.baseline_data['steps_per_episode'])
        
        sns.kdeplot(data=improved_efficiency, label='Improved', ax=ax)
        sns.kdeplot(data=baseline_efficiency, label='Baseline', ax=ax)
        ax.set_title('任务完成效率分布')
        ax.set_xlabel('Efficiency (survival/steps)')
        ax.set_ylabel('Density')
        
    def _plot_improvement_trends(self, ax):
        """绘制性能提升趋势"""
        window = 100
        improved_trend = (self.improved_data['episode_rewards'].rolling(window).mean() -
                         self.baseline_data['episode_rewards'].rolling(window).mean())
        ax.plot(improved_trend, label='Performance Gain')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_title('相对基准的性能提升趋势')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Difference')
        ax.legend()
        
    def generate_statistical_report(self):
        """生成统计分析报告"""
        metrics = ['episode_rewards', 'red_survival', 'tactical_advantage',
                  'team_coordination', 'energy_efficiency']
        
        report = {
            'basic_statistics': {},
            'significance_tests': {},
            'effect_sizes': {}
        }
        
        for metric in metrics:
            # 基础统计
            improved_stats = self.improved_data[metric].describe()
            baseline_stats = self.baseline_data[metric].describe()
            
            report['basic_statistics'][metric] = {
                'improved': improved_stats.to_dict(),
                'baseline': baseline_stats.to_dict()
            }
            
            # 显著性检验
            t_stat, p_value = stats.ttest_ind(
                self.improved_data[metric],
                self.baseline_data[metric]
            )
            
            report['significance_tests'][metric] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            }
            
            # 效应量（Cohen's d）
            d = (self.improved_data[metric].mean() - self.baseline_data[metric].mean()) / \
                np.sqrt((self.improved_data[metric].var() + self.baseline_data[metric].var()) / 2)
            
            report['effect_sizes'][metric] = float(d)
            
        return report

def analyze_experiment_results(results_dir):
    """分析实验结果的主函数"""
    analyzer = PerformanceAnalyzer(results_dir)
    
    # 绘制详细分析图表
    analyzer.plot_learning_curves()
    
    # 生成统计报告
    report = analyzer.generate_statistical_report()
    
    # 保存报告
    import json
    with open(os.path.join(results_dir, 'statistical_analysis.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    return report

if __name__ == "__main__":
    # 示例用法
    results_dir = "path/to/your/results"
    report = analyze_experiment_results(results_dir)
    print("Analysis completed. Results saved to:", results_dir)