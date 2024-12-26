# test_mozi_training.py

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
from agents.baseline import MADDPG
from envs.state_handler import StateHandler
from envs.combat_mechanics import CombatMechanics


def setup_environment(env_config=None):
    """设置测试环境"""
    if not os.environ.get("MOZIPATH"):
        os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
    print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")

    config = {
        'num_red': 2,
        'num_blue': 3,
        'max_steps': getattr(env_config or EnvConfig, 'MAX_STEPS', 200),
        'num_episodes': 5,  # 减少测试回合数
        'field_size': 100000.0,
        'attack_range': getattr(env_config or EnvConfig, 'ATTACK_RANGE', 25000.0),
        'min_speed': getattr(env_config or EnvConfig, 'MIN_SPEED', 150.0),
        'max_speed': getattr(env_config or EnvConfig, 'MAX_SPEED', 400.0),
        'save_interval': 2,  # 每隔多少回合保存
        'model_dir': 'test_models',
        'log_dir': 'test_logs'
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


def test_single_episode(training_env):
    """测试单回合训练"""
    env, config = training_env
    print("\n=== 测试单回合训练 ===")

    try:
        # 初始化MADDPG
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        maddpg = MADDPG(config['num_red'], config['num_blue'], obs_dim, act_dim)

        print(f"MADDPG初始化完成 - 观察维度: {obs_dim}, 动作维度: {act_dim}")

        # 运行单回合
        obs = env.reset()
        episode_reward = 0

        for step in range(config['max_steps']):
            # 选择动作
            actions = maddpg.select_actions(obs)

            # 执行动作
            next_obs, rewards, done, info = env.step(actions)

            # 存储经验
            maddpg.store_transition(obs, actions, rewards, next_obs, done)

            # 累计奖励
            episode_reward += sum(rewards.values() if rewards else [0])

            # 更新观察
            obs = next_obs

            # 打印进度
            print(f"\nStep {step + 1}/{config['max_steps']}")
            print(f"Red: {info['red_alive']}, Blue: {info['blue_alive']}")
            print(f"当前奖励: {episode_reward:.2f}")

            if done:
                break

        # 验证经验收集
        assert len(maddpg.memory) > 0
        print(f"\n成功收集 {len(maddpg.memory)} 条经验")

        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_episode_training(training_env):
    """测试多回合训练"""
    env, config = training_env
    print("\n=== 测试多回合训练 ===")

    try:
        # 初始化MADDPG
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        maddpg = MADDPG(config['num_red'], config['num_blue'], obs_dim, act_dim)

        training_stats = []

        # 运行多个回合
        for episode in range(config['num_episodes']):
            print(f"\n开始回合 {episode + 1}/{config['num_episodes']}")

            obs = env.reset()
            episode_reward = 0
            episode_steps = 0

            for step in range(config['max_steps']):
                # 选择和执行动作
                actions = maddpg.select_actions(obs)
                next_obs, rewards, done, info = env.step(actions)

                # 存储经验
                maddpg.store_transition(obs, actions, rewards, next_obs, done)

                # 如果经验足够，进行训练
                if len(maddpg.memory) >= maddpg.batch_size:
                    maddpg.train()

                # 更新统计
                episode_reward += sum(rewards.values() if rewards else [0])
                episode_steps += 1
                obs = next_obs

                # 打印进度
                if (step + 1) % 20 == 0:
                    print(f"Step {step + 1}/{config['max_steps']}")
                    print(f"Red: {info['red_alive']}, Blue: {info['blue_alive']}")

                if done:
                    break

            # 记录回合统计
            training_stats.append({
                'episode': episode + 1,
                'reward': episode_reward,
                'steps': episode_steps
            })

            print(f"\n回合 {episode + 1} 统计:")
            print(f"总步数: {episode_steps}")
            print(f"总奖励: {episode_reward:.2f}")

            # 保存检查点
            if (episode + 1) % config['save_interval'] == 0:
                os.makedirs(config['model_dir'], exist_ok=True)
                maddpg.save(os.path.join(config['model_dir'], f'maddpg_episode_{episode + 1}.pth'))

        # 验证训练结果
        assert len(training_stats) == config['num_episodes']
        print("\n训练完成!")
        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model_evaluation(training_env):
    """测试模型评估"""
    env, config = training_env
    print("\n=== 测试模型评估 ===")

    try:
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        maddpg = MADDPG(config['num_red'], config['num_blue'], obs_dim, act_dim)

        # 运行评估回合
        eval_stats = []
        num_eval_episodes = 2

        for episode in range(num_eval_episodes):
            print(f"\n评估回合 {episode + 1}/{num_eval_episodes}")
            obs = env.reset()
            episode_reward = 0

            for step in range(config['max_steps']):
                # 无噪声动作选择
                actions = maddpg.select_actions(obs, noise_scale=0.0)
                next_obs, rewards, done, info = env.step(actions)

                episode_reward += sum(rewards.values() if rewards else [0])
                obs = next_obs

                if done:
                    break

            eval_stats.append(episode_reward)

        # 验证评估结果
        assert len(eval_stats) == num_eval_episodes
        print(f"\n平均评估奖励: {np.mean(eval_stats):.2f}")
        return True

    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    pytest.main(["-v", __file__])