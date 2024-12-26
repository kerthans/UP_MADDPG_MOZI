# test_up_training.py

import os
import sys
import numpy as np
import torch
import time
import pytest
from datetime import datetime
import json
from envs.mozi_adapter import MoziAdapter
from envs.env_config import EnvConfig
from envs.state_handler import StateHandler
from envs.combat_mechanics import CombatMechanics
from agents.up import MADDPG  # 导入优化后的MADDPG

def setup_environment(env_config=None):
    """设置测试环境"""
    if not os.environ.get("MOZIPATH"):
        os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
    print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")

    config = {
        'num_red': 2,
        'num_blue': 3,
        'max_steps': getattr(env_config or EnvConfig, 'MAX_STEPS', 200),
        'num_episodes': 5,
        'field_size': 100000.0,
        'attack_range': getattr(env_config or EnvConfig, 'ATTACK_RANGE', 25000.0),
        'min_speed': getattr(env_config or EnvConfig, 'MIN_SPEED', 150.0),
        'max_speed': getattr(env_config or EnvConfig, 'MAX_SPEED', 400.0),
        'save_interval': 2,
        'model_dir': 'up_models',  # 新的模型保存目录
        'log_dir': 'up_logs',      # 新的日志保存目录
        # MADDPG优化参数
        'n_step': 3,               # n步预测
        'gamma': 0.99,             # 折扣因子
        'capacity': 1000000,       # 经验回放容量
        'alpha': 0.6,              # 优先级采样参数
        'beta_start': 0.4,         # IS权重初始beta值
        'beta_frames': 100000,     # beta递增帧数
        'batch_size': 128,         # 批次大小
        'lr_actor': 1e-4,          # Actor学习率
        'lr_critic': 1e-3,         # Critic学习率
        'weight_decay': 1e-5,      # 权重衰减
        'dropout': 0.3,            # Dropout率
        'hidden_dim': 256,         # 隐藏层维度
        'tau': 0.01,               # 目标网络软更新系数
    }
    return config

@pytest.fixture
def training_env():
    """环境fixture"""
    config = setup_environment(EnvConfig)
    env = MoziAdapter(
        num_red=config['num_red'],
        num_blue=config['num_blue'],
        max_steps=config['max_steps'],
        env_config=EnvConfig
    )
    return env, config


def save_training_stats(stats, episode, config):
    """保存训练统计数据，处理numpy数组的序列化"""
    os.makedirs(config['log_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(config['log_dir'], f'training_stats_{timestamp}.json')

    def convert_to_serializable(obj):
        """递归转换数据为可JSON序列化的格式"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        return obj

    # 转换统计数据
    serializable_stats = convert_to_serializable(stats)

    # 转换配置数据（排除不可序列化的对象）
    serializable_config = {
        k: convert_to_serializable(v)
        for k, v in config.items()
        if not callable(v)
    }

    # 保存JSON
    with open(filename, 'w') as f:
        json.dump({
            'episode': episode,
            'stats': serializable_stats,
            'config': serializable_config
        }, f, indent=4)

def test_single_episode(training_env):
    """测试单回合训练"""
    env, config = training_env
    print("\n=== 测试优化版MADDPG单回合训练 ===")

    try:
        # 初始化优化版MADDPG
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        maddpg = MADDPG(
            n_red=config['num_red'],
            n_blue=config['num_blue'],
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_step=config['n_step'],
            gamma=config['gamma'],
            capacity=config['capacity'],
            alpha=config['alpha'],
            beta_start=config['beta_start'],
            beta_frames=config['beta_frames'],
            batch_size=config['batch_size'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            hidden_dim=config['hidden_dim'],
            tau=config['tau']
        )

        print(f"优化版MADDPG初始化完成 - 观察维度: {obs_dim}, 动作维度: {act_dim}")

        # 运行单回合
        obs = env.reset()
        episode_stats = {
            'total_reward': 0,
            'step_rewards': [],
            'red_alive': [],
            'blue_alive': [],
            'actions_taken': []
        }

        for step in range(config['max_steps']):
            # 选择动作
            actions = maddpg.select_actions(obs)
            episode_stats['actions_taken'].append(actions)

            # 执行动作
            next_obs, rewards, done, info = env.step(actions)

            # 存储经验
            maddpg.store_transition(obs, actions, rewards, next_obs, done)

            # 训练网络
            if len(maddpg.memory) >= maddpg.batch_size:
                maddpg.train()

            # 更新统计信息
            step_reward = sum(rewards.values() if rewards else [0])
            episode_stats['total_reward'] += step_reward
            episode_stats['step_rewards'].append(step_reward)
            episode_stats['red_alive'].append(info['red_alive'])
            episode_stats['blue_alive'].append(info['blue_alive'])

            # 打印进度
            if (step + 1) % 20 == 0:
                print(f"\nStep {step + 1}/{config['max_steps']}")
                print(f"Red存活: {info['red_alive']}, Blue存活: {info['blue_alive']}")
                print(f"当前累计奖励: {episode_stats['total_reward']:.2f}")

            # 更新观察
            obs = next_obs

            if done:
                break

        # 保存统计数据
        save_training_stats(episode_stats, 1, config)

        print("\n=== 单回合训练完成 ===")
        print(f"总步数: {step + 1}")
        print(f"总奖励: {episode_stats['total_reward']:.2f}")
        print(f"经验池大小: {len(maddpg.memory)}")

        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_episode_training(training_env):
    """测试多回合训练"""
    env, config = training_env
    print("\n=== 测试优化版MADDPG多回合训练 ===")

    try:
        # 初始化优化版MADDPG
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        maddpg = MADDPG(
            n_red=config['num_red'],
            n_blue=config['num_blue'],
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_step=config['n_step'],
            gamma=config['gamma'],
            capacity=config['capacity'],
            alpha=config['alpha'],
            beta_start=config['beta_start'],
            beta_frames=config['beta_frames'],
            batch_size=config['batch_size'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            hidden_dim=config['hidden_dim'],
            tau=config['tau']
        )

        training_stats = []

        for episode in range(config['num_episodes']):
            print(f"\n开始回合 {episode + 1}/{config['num_episodes']}")
            
            obs = env.reset()
            episode_stats = {
                'episode': episode + 1,
                'total_reward': 0,
                'step_rewards': [],
                'red_alive': [],
                'blue_alive': [],
                'actions_taken': [],
                'training_steps': 0
            }

            for step in range(config['max_steps']):
                # 选择动作
                actions = maddpg.select_actions(obs, noise_scale=max(0.1, 1.0 - episode/config['num_episodes']))
                episode_stats['actions_taken'].append(actions)

                # 执行动作
                next_obs, rewards, done, info = env.step(actions)

                # 存储经验
                maddpg.store_transition(obs, actions, rewards, next_obs, done)

                # 训练网络
                if len(maddpg.memory) >= maddpg.batch_size:
                    maddpg.train()
                    episode_stats['training_steps'] += 1

                # 更新统计信息
                step_reward = sum(rewards.values() if rewards else [0])
                episode_stats['total_reward'] += step_reward
                episode_stats['step_rewards'].append(step_reward)
                episode_stats['red_alive'].append(info['red_alive'])
                episode_stats['blue_alive'].append(info['blue_alive'])

                # 打印进度
                if (step + 1) % 20 == 0:
                    print(f"Step {step + 1}/{config['max_steps']}")
                    print(f"Red存活: {info['red_alive']}, Blue存活: {info['blue_alive']}")

                # 更新观察
                obs = next_obs

                if done:
                    break

            # 保存回合统计
            training_stats.append(episode_stats)
            save_training_stats(episode_stats, episode + 1, config)

            print(f"\n回合 {episode + 1} 统计:")
            print(f"总步数: {step + 1}")
            print(f"总奖励: {episode_stats['total_reward']:.2f}")
            print(f"训练步数: {episode_stats['training_steps']}")

            # 保存模型检查点
            if (episode + 1) % config['save_interval'] == 0:
                os.makedirs(config['model_dir'], exist_ok=True)
                maddpg.save(os.path.join(config['model_dir'], f'maddpg_up_episode_{episode + 1}.pth'))

        print("\n=== 多回合训练完成 ===")
        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_evaluation(training_env):
    """测试模型评估"""
    env, config = training_env
    print("\n=== 测试优化版MADDPG模型评估 ===")

    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        maddpg = MADDPG(
            n_red=config['num_red'],
            n_blue=config['num_blue'],
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_step=config['n_step'],
            gamma=config['gamma'],
            capacity=config['capacity'],
            alpha=config['alpha'],
            beta_start=config['beta_start'],
            beta_frames=config['beta_frames'],
            batch_size=config['batch_size'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            hidden_dim=config['hidden_dim'],
            tau=config['tau']
        )

        # 运行评估回合
        eval_stats = []
        num_eval_episodes = 2

        for episode in range(num_eval_episodes):
            print(f"\n评估回合 {episode + 1}/{num_eval_episodes}")
            
            obs = env.reset()
            episode_stats = {
                'episode': episode + 1,
                'total_reward': 0,
                'red_alive': [],
                'blue_alive': [],
                'actions_taken': []
            }

            for step in range(config['max_steps']):
                # 无噪声动作选择
                actions = maddpg.select_actions(obs, noise_scale=0.0)
                episode_stats['actions_taken'].append(actions)

                # 执行动作
                next_obs, rewards, done, info = env.step(actions)

                # 更新统计信息
                episode_stats['total_reward'] += sum(rewards.values() if rewards else [0])
                episode_stats['red_alive'].append(info['red_alive'])
                episode_stats['blue_alive'].append(info['blue_alive'])

                obs = next_obs

                if done:
                    break

            eval_stats.append(episode_stats)
            save_training_stats(episode_stats, f'eval_{episode + 1}', config)

        # 计算评估指标
        avg_reward = np.mean([stats['total_reward'] for stats in eval_stats])
        avg_red_survival = np.mean([np.mean(stats['red_alive']) for stats in eval_stats])
        avg_blue_survival = np.mean([np.mean(stats['blue_alive']) for stats in eval_stats])

        print("\n=== 评估结果 ===")
        print(f"平均奖励: {avg_reward:.2f}")
        print(f"平均红方存活率: {avg_red_survival:.2%}")
        print(f"平均蓝方存活率: {avg_blue_survival:.2%}")

        return True

    except Exception as e:
        print(f"评估失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_save_load(training_env):
    """测试模型保存和加载"""
    env, config = training_env
    print("\n=== 测试优化版MADDPG模型保存和加载 ===")

    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        # 创建模型并训练一个回合
        maddpg = MADDPG(
            n_red=config['num_red'],
            n_blue=config['num_blue'],
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_step=config['n_step'],
            gamma=config['gamma'],
            capacity=config['capacity'],
            alpha=config['alpha'],
            beta_start=config['beta_start'],
            beta_frames=config['beta_frames'],
            batch_size=config['batch_size'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            hidden_dim=config['hidden_dim'],
            tau=config['tau']
        )

        # 执行少量训练步骤
        obs = env.reset()
        for _ in range(10):
            actions = maddpg.select_actions(obs)
            next_obs, rewards, done, _ = env.step(actions)
            maddpg.store_transition(obs, actions, rewards, next_obs, done)
            obs = next_obs
            if len(maddpg.memory) >= maddpg.batch_size:
                maddpg.train()

        # 保存模型
        save_path = os.path.join(config['model_dir'], 'test_save_load.pth')
        os.makedirs(config['model_dir'], exist_ok=True)
        maddpg.save(save_path)
        print(f"\n模型已保存到: {save_path}")

        # 创建新的模型实例并加载
        new_maddpg = MADDPG(
            n_red=config['num_red'],
            n_blue=config['num_blue'],
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_step=config['n_step'],
            gamma=config['gamma'],
            capacity=config['capacity'],
            alpha=config['alpha'],
            beta_start=config['beta_start'],
            beta_frames=config['beta_frames'],
            batch_size=config['batch_size'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            hidden_dim=config['hidden_dim'],
            tau=config['tau']
        )
        new_maddpg.load(save_path)
        print("模型加载成功!")

        # 验证加载的模型可以正常运行
        obs = env.reset()
        actions = new_maddpg.select_actions(obs)
        _, _, _, _ = env.step(actions)

        print("加载的模型可以正常运行")
        return True

    except Exception as e:
        print(f"保存/加载测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_hyperparameter_adjustment(training_env):
    """测试超参数动态调整"""
    env, config = training_env
    print("\n=== 测试优化版MADDPG超参数动态调整 ===")

    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        maddpg = MADDPG(
            n_red=config['num_red'],
            n_blue=config['num_blue'],
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_step=config['n_step'],
            gamma=config['gamma'],
            capacity=config['capacity'],
            alpha=config['alpha'],
            beta_start=config['beta_start'],
            beta_frames=config['beta_frames'],
            batch_size=config['batch_size'],
            lr_actor=config['lr_actor'],
            lr_critic=config['lr_critic'],
            weight_decay=config['weight_decay'],
            dropout=config['dropout'],
            hidden_dim=config['hidden_dim'],
            tau=config['tau']
        )

        # 测试调整各种超参数
        new_params = {
            'gamma': 0.98,
            'tau': 0.02,
            'batch_size': 64,
        }
        
        print("\n调整前的参数:")
        print(f"gamma: {maddpg.gamma}")
        print(f"tau: {maddpg.tau}")
        print(f"batch_size: {maddpg.batch_size}")

        maddpg.adjust_hyperparameters(new_params)

        print("\n调整后的参数:")
        print(f"gamma: {maddpg.gamma}")
        print(f"tau: {maddpg.tau}")
        print(f"batch_size: {maddpg.batch_size}")

        # 验证调整后的模型可以正常运行
        obs = env.reset()
        for _ in range(5):
            actions = maddpg.select_actions(obs)
            next_obs, rewards, done, _ = env.step(actions)
            maddpg.store_transition(obs, actions, rewards, next_obs, done)
            if len(maddpg.memory) >= maddpg.batch_size:
                maddpg.train()
            obs = next_obs
            if done:
                break

        print("\n使用新参数训练正常")
        return True

    except Exception as e:
        print(f"超参数调整测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 运行所有测试
    pytest.main(["-v", __file__])