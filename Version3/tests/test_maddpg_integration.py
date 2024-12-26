import os
import sys
import numpy as np
import torch
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
    
    # 基础配置
    config = {
        'num_red': 2,
        'num_blue': 3,
        'max_steps': getattr(env_config or EnvConfig, 'MAX_STEPS', 200),
        'field_size': 100000.0,
        'attack_range': getattr(env_config or EnvConfig, 'ATTACK_RANGE', 25000.0),
        'min_speed': getattr(env_config or EnvConfig, 'MIN_SPEED', 150.0),
        'max_speed': getattr(env_config or EnvConfig, 'MAX_SPEED', 400.0)
    }
    
    return config

#
# def test_maddpg_integration():
#     """测试MADDPG与墨子环境的集成"""
#     print("\n=== 开始MADDPG集成测试 ===")
#     try:
#         # 1. 设置环境
#         config = setup_environment(EnvConfig)
#         env = MoziAdapter(
#             num_red=config['num_red'],
#             num_blue=config['num_blue'],
#             max_steps=config['max_steps'],
#             env_config=EnvConfig
#         )
#
#         print("\n1. 环境初始化完成")
#
#         # 2. 初始化MADDPG
#         obs_dim = env.observation_space.shape[0]
#         act_dim = env.action_space.shape[0]
#         maddpg = MADDPG(config['num_red'], config['num_blue'], obs_dim, act_dim)
#
#         print("\n2. MADDPG初始化完成")
#         print(f"观察空间维度: {obs_dim}")
#         print(f"动作空间维度: {act_dim}")
#
#         # 3. 测试单回合交互
#         print("\n3. 开始测试单回合交互...")
#         obs = env.reset()
#         print(f"初始观察: {obs.keys()}")
#
#         episode_reward = 0
#         step_count = 0
#
#         for step in range(config['max_steps']):
#             step_count += 1
#
#             # 选择动作
#             actions = maddpg.select_actions(obs)
#             print(f"\n选择的动作: {actions}")
#
#             # 执行动作
#             next_obs, rewards, done, info = env.step(actions)
#             print(f"执行结果: \n - 观察: {next_obs.keys()}\n - 奖励: {rewards}\n - 信息: {info}")
#
#             # 存储经验
#             maddpg.store_transition(obs, actions, rewards, next_obs, done)
#
#             # 更新状态
#             obs = next_obs
#             episode_reward += sum(rewards.values() if rewards else [0])
#
#             print(f"\nStep {step + 1}/{config['max_steps']}")
#             print(f"红方单位数: {info['red_alive']}")
#             print(f"蓝方单位数: {info['blue_alive']}")
#             print(f"当前回合奖励: {episode_reward}")
#
#             if done:
#                 print(f"回合在第 {step + 1} 步结束")
#                 break
#
#         print(f"\n完成 {step_count} 步交互")
#
#         print("\n4. 测试训练过程...")
#         if len(maddpg.memory) > maddpg.batch_size:
#             maddpg.train()
#             print("完成一次训练")
#         else:
#             print(f"经验池中样本数量（{len(maddpg.memory)}）不足以进行训练（需要{maddpg.batch_size}）")
#
#         print("\n=== 集成测试完成 ===")
#         return True
#
#     except Exception as e:
#         print(f"\n测试失败: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False
# def test_maddpg_integration():
#     """测试MADDPG与墨子环境的集成"""
#     print("\n=== 开始MADDPG集成测试 ===")
#     try:
#         # 1. 设置环境
#         config = setup_environment(EnvConfig)
#         env = MoziAdapter(
#             num_red=config['num_red'],
#             num_blue=config['num_blue'],
#             max_steps=config['max_steps'],
#             env_config=EnvConfig
#         )
#
#         print("\n1. 环境初始化完成")
#         print(f"环境配置: {config}")
#
#         # 2. 初始化MADDPG
#         obs_dim = env.observation_space.shape[0]
#         act_dim = env.action_space.shape[0]
#         maddpg = MADDPG(config['num_red'], config['num_blue'], obs_dim, act_dim)
#
#         print("\n2. MADDPG初始化完成")
#         print(f"观察空间维度: {obs_dim}")
#         print(f"动作空间维度: {act_dim}")
#
#         # 3. 测试单回合交互
#         print("\n3. 开始测试单回合交互...")
#         obs = env.reset()
#         print(f"初始观察: ")
#         for k, v in obs.items():
#             print(f"  {k}: shape={np.array(v).shape}")
#
#         episode_reward = 0
#         step_count = 0
#
#         for step in range(config['max_steps']):
#             step_count += 1
#
#             # 选择动作
#             actions = maddpg.select_actions(obs)
#             print(f"\n步骤 {step + 1} 选择的动作:")
#             for k, v in actions.items():
#                 print(f"  {k}: {v}")
#
#             # 执行动作
#             next_obs, rewards, done, info = env.step(actions)
#             print(f"\n步骤 {step + 1} 执行结果:")
#             print(f"观察: {list(next_obs.keys())}")
#             print(f"奖励: {rewards}")
#             print(f"信息: {info}")
#
#             # 存储经验
#             maddpg.store_transition(obs, actions, rewards, next_obs, done)
#
#             # 更新状态
#             obs = next_obs
#             episode_reward += sum(rewards.values() if rewards else [0])
#
#             if done:
#                 print(f"\n回合在第 {step + 1} 步结束")
#                 break
#
#         print(f"\n完成 {step_count} 步交互")
#
#         return True
#
#     except Exception as e:
#         print(f"\n测试失败: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False
def test_maddpg_integration():
    """测试MADDPG与墨子环境的集成"""
    print("\n=== 开始MADDPG集成测试 ===")
    try:
        # 1. 设置环境
        config = setup_environment(EnvConfig)
        env = MoziAdapter(
            num_red=config['num_red'],
            num_blue=config['num_blue'],
            max_steps=config['max_steps'],
            env_config=EnvConfig
        )

        print("\n1. 环境初始化完成")

        # 2. 初始化MADDPG
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        maddpg = MADDPG(config['num_red'], config['num_blue'], obs_dim, act_dim)

        print("\n2. MADDPG初始化完成")
        print(f"观察空间维度: {obs_dim}")
        print(f"动作空间维度: {act_dim}")

        # 3. 测试单回合交互
        print("\n3. 开始测试单回合交互...")
        obs = env.reset()
        print(f"初始观察: {obs.keys()}")

        episode_reward = 0
        step_count = 0

        for step in range(config['max_steps']):
            step_count += 1

            # 选择动作
            actions = maddpg.select_actions(obs)
            print(f"\n选择的动作: {actions}")

            # 执行动作
            next_obs, rewards, done, info = env.step(actions)
            print(f"执行结果: \n - 观察: {next_obs.keys()}\n - 奖励: {rewards}\n - 信息: {info}")

            # 存储经验
            maddpg.store_transition(obs, actions, rewards, next_obs, done)

            # 更新状态
            obs = next_obs
            episode_reward += sum(rewards.values() if rewards else [0])

            print(f"\nStep {step + 1}/{config['max_steps']}")
            print(f"红方单位数: {info['red_alive']}")
            print(f"蓝方单位数: {info['blue_alive']}")
            print(f"当前回合奖励: {episode_reward}")

            if done:
                print(f"回合在第 {step + 1} 步结束")
                break

        print(f"\n完成 {step_count} 步交互")

        print("\n4. 测试训练过程...")
        if len(maddpg.memory) > maddpg.batch_size:
            maddpg.train()
            print("完成一次训练")
        else:
            print(f"经验池中样本数量（{len(maddpg.memory)}）不足以进行训练（需要{maddpg.batch_size}）")

        print("\n=== 集成测试完成 ===")
        return True

    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
if __name__ == "__main__":
    test_result = test_maddpg_integration()
    print(f"\nMADDPG集成测试{'成功' if test_result else '失败'}")