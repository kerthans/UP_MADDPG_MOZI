# mozi_upmodel_load.py

import os
import sys
import numpy as np
import torch
from envs.mozi_adapter import MoziAdapter
from envs.env_config import EnvConfig
from agents.up import MADDPG


class SimpleMoziInference:
    def __init__(self, model_path: str):
        """初始化推理器"""
        # 环境配置
        self.env_config = {
            'num_red': 2,
            'num_blue': 3,
            'max_steps': 200
        }

        # 模型配置
        self.model_config = {
            'n_step': 3,
            'gamma': 0.99,
            'capacity': 1000000,
            'alpha': 0.6,
            'beta_start': 0.4,
            'beta_frames': 100000,
            'batch_size': 128,
            'lr_actor': 1e-4,
            'lr_critic': 1e-3,
            'weight_decay': 1e-5,
            'dropout': 0.3,
            'hidden_dim': 256,
            'tau': 0.01,
        }

        # 初始化环境
        if not os.environ.get("MOZIPATH"):
            os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
        print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")

        self.env = MoziAdapter(
            num_red=self.env_config['num_red'],
            num_blue=self.env_config['num_blue'],
            max_steps=self.env_config['max_steps'],
            env_config=EnvConfig
        )

        # 获取观察和动作空间维度
        obs = self.env.reset()
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        # 初始化并加载模型
        self.maddpg = MADDPG(
            n_red=self.env_config['num_red'],
            n_blue=self.env_config['num_blue'],
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            **self.model_config
        )

        print(f"加载模型: {model_path}")
        self.maddpg.load(model_path)
        print("模型加载成功！")

    def run(self, num_episodes: int = 5):
        """运行仿真"""
        print(f"\n开始运行 {num_episodes} 个回合...")

        for episode in range(num_episodes):
            print(f"\n=== 回合 {episode + 1}/{num_episodes} ===")
            obs = self.env.reset()
            total_reward = 0
            step = 0

            while True:
                # 选择动作
                actions = self.maddpg.select_actions(obs)

                # 执行动作
                next_obs, rewards, done, info = self.env.step(actions)

                # 更新统计
                total_reward += sum(rewards.values())
                step += 1

                # 显示实时状态
                red_reward = sum(v for k, v in rewards.items() if k.startswith('red_'))
                print(f"\r步数: {step}, 红方累计奖励: {red_reward:.1f}, "
                      f"红方存活: {info['red_alive']}, 蓝方存活: {info['blue_alive']}",
                      end='', flush=True)

                obs = next_obs

                if done:
                    print(f"\n回合结束! 总步数: {step}")
                    print(f"总奖励: {total_reward:.1f}")
                    print(f"红方存活数量: {info['red_alive']}")
                    print(f"蓝方存活数量: {info['blue_alive']}")
                    print(f"红方胜利: {'是' if info['blue_alive'] == 0 else '否'}")
                    break


if __name__ == "__main__":
    # 直接指定模型路径
    MODEL_PATH = r"G:\work\danzi\mozi\mozi_ai_sdk\testxiangding\results\mozi_training\20241220_023536\model_episode_final.pt"

    # 创建推理器并运行
    inferencer = SimpleMoziInference(MODEL_PATH)
    inferencer.run(num_episodes=1)  # 运行5个回合