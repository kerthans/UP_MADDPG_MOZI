import os
import json
import multiprocessing
from pathlib import Path
from datetime import datetime
from baseline_train import BaselineTrainer
from up_train import EnhancedTrainer
from analysis import MADDPGAnalyzer
import numpy as np

def train_baseline(config, result_queue):
    """运行baseline训练"""
    config['training'].update({
        'initial_noise': 0.4,
        'noise_decay': 0.9998,
        'batch_size': 128,
        'buffer_size': 500000
    })
    
    trainer = BaselineTrainer(config)
    trainer.train()
    result_queue.put(('baseline', trainer.save_dir))

def train_enhanced(config, result_queue):
    """运行增强版训练"""
    config['training'].update({
        'n_step': 3,           # 使用n步回报
        'gamma': 0.99,         # 提高折扣因子
        'buffer_size': 1000000,# 增大buffer size
        'batch_size': 256,     # 增大batch size
        'per_alpha': 0.6,      # PER参数
        'per_beta': 0.4,       # PER参数
        'beta_frames': 100000,
        'lr_actor': 2e-4,      # 提高actor学习率
        'lr_critic': 6e-4,     # 提高critic学习率
        'hidden_dim': 512,     # 增大网络规模
        'patience': 30,        # 增加早停耐心值
        'initial_noise': 0.3,  # 降低初始噪声
        'noise_decay': 0.9995  # 提高噪声衰减速度
    })
    
    trainer = EnhancedTrainer(config)
    trainer.train()
    result_queue.put(('enhanced', trainer.save_dir))

def main():
    # 基础配置
    base_config = {
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
            'num_threads': 8  # 减小线程数以适应并行训练
        },
        'training': {
            'n_episodes': 20,        # 训练回合数
            'save_interval': 100,      # 保存间隔
            'eval_interval': 100,      # 评估间隔
            'initial_noise': 0.3,
            'noise_decay': 0.9995,
            'min_noise': 0.01
        }
    }

    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path(f'comparison_results_{timestamp}')
    result_dir.mkdir(parents=True, exist_ok=True)

    # 设置各自的保存目录
    baseline_config = base_config.copy()
    enhanced_config = base_config.copy()
    baseline_config['save_dir'] = str(result_dir / 'baseline')
    enhanced_config['save_dir'] = str(result_dir / 'enhanced')

    # 使用多进程并行训练
    result_queue = multiprocessing.Queue()
    baseline_process = multiprocessing.Process(
        target=train_baseline, 
        args=(baseline_config, result_queue)
    )
    enhanced_process = multiprocessing.Process(
        target=train_enhanced, 
        args=(enhanced_config, result_queue)
    )

    print("开始并行训练...")
    baseline_process.start()
    enhanced_process.start()

    # 等待训练完成
    baseline_process.join()
    enhanced_process.join()

    # 获取训练结果路径
    results = {}
    while not result_queue.empty():
        model_type, save_dir = result_queue.get()
        results[model_type] = save_dir

    # 进行对比分析
    analyzer = MADDPGAnalyzer(
        baseline_path=Path(results['baseline']) / 'final_report.json',
        improve_path=Path(results['enhanced']) / 'final_report.json'
    )

    # 生成分析报告和可视化
    print("\n生成分析报告和可视化...")
    analysis_dir = result_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analyzer.generate_analysis_report(str(analysis_dir))

    print(f"\n对比训练完成！结果保存在: {result_dir}")
    print(f"基准版本结果: {results['baseline']}")
    print(f"增强版本结果: {results['enhanced']}")
    print(f"分析报告位置: {analysis_dir}")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows支持
    main()