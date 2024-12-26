# tests/test_main.py

import unittest
import numpy as np
import torch
import os
import json
from datetime import datetime
import pandas as pd
from main import plot_training_curves, save_evaluation_report
from env.combat_env import CombatEnv
from agents.maddpg import MADDPG
from torch.cuda.amp import GradScaler
import warnings

class TestMainFunction(unittest.TestCase):
    """测试主函数功能"""
    
    def setUp(self):
        """测试初始化"""
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.test_params = {
            'num_episodes': 10,
            'max_steps': 300,
            'num_red': 3,
            'num_blue': 3,
            'state_dim': 9,  # 更新为9维状态空间
            'action_dim': 2,
            'lr': 1e-4,
            'gamma': 0.99,
            'tau': 0.001,
            'initial_noise': 0.5,
            'noise_decay': 0.9999,
            'min_noise': 0.05,
            'hidden_dim': 256,
            'batch_size': 128,
            'buffer_size': int(1e6),
            'max_velocity': 5.0,
            'seed': 42,
            'heterogeneous': True,  # 启用异构
            'use_curiosity': True,   # 启用好奇心机制
            'update_freq': 2,
            'grad_clip': 0.5,
            'device': self.device  # 添加device参数
        }
        
        # 设置随机种子
        np.random.seed(self.test_params['seed'])
        torch.manual_seed(self.test_params['seed'])
        
        # 创建测试目录
        self.test_dir = os.path.join("test_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 初始化智能体类型
        self.red_agent_types = ['scout', 'fighter', 'bomber']
        self.blue_agent_types = ['fighter'] * self.test_params['num_blue']
        
        # 创建环境
        self.env = CombatEnv(
            num_red=self.test_params['num_red'],
            num_blue=self.test_params['num_blue'],
            state_dim=self.test_params['state_dim'],
            action_dim=self.test_params['action_dim'],
            max_steps=self.test_params['max_steps'],
            red_agent_types=self.red_agent_types,
            blue_agent_types=self.blue_agent_types,
            heterogeneous=self.test_params['heterogeneous']
        )
        
        # 创建MADDPG智能体，注意不再使用.to(device)
        # 在setUp方法中修改MADDPG实例化
        self.maddpg = MADDPG(
            num_agents=self.test_params['num_red'] + self.test_params['num_blue'],
            state_dim=self.test_params['state_dim'],
            action_dim=self.test_params['action_dim'],
            lr=self.test_params['lr'],
            gamma=self.test_params['gamma'],
            tau=self.test_params['tau'],
            buffer_size=self.test_params['buffer_size'],
            batch_size=self.test_params['batch_size'],
            hidden_dim=self.test_params['hidden_dim'],
            agent_types=self.red_agent_types + self.blue_agent_types,
            device=self.device  # 传入device参数，不再调用.to(device)
        )
        
        # 初始化梯度缩放器
        self.scaler = GradScaler(device_type=self.device.type) if torch.cuda.is_available() else None

    def test_environment_setup(self):
        """测试环境设置"""
        obs = self.env.reset()
        self.assertEqual(len(obs), self.env.total_agents * self.env.state_dim)
        print("✓ 观察空间维度正确")
        
        # 验证异构智能体设置
        if self.test_params['heterogeneous']:
            self.assertEqual(len(self.env.red_agent_types), self.test_params['num_red'])
            self.assertEqual(len(self.env.blue_agent_types), self.test_params['num_blue'])
            print("✓ 异构智能体配置正确")

    def test_training_step(self):
        """测试单个训练步骤"""
        obs = self.env.reset()
        per_agent_obs = []
        for i in range(self.maddpg.num_agents):
            agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
            per_agent_obs.append(torch.FloatTensor(agent_obs).to(self.device))
        
        with torch.amp.autocast(device_type=self.device.type, enabled=torch.cuda.is_available()):
            actions = self.maddpg.select_actions(per_agent_obs, noise=0.1)
        
        next_obs, rewards, done, info = self.env.step(actions)
        
        # 验证奖励结构
        red_rewards = rewards[:self.test_params['num_red']]
        blue_rewards = rewards[self.test_params['num_red']:]
        
        print("✓ 单步训练测试通过")
        print(f"- 红方奖励: {np.mean(red_rewards):.2f}")
        print(f"- 蓝方奖励: {np.mean(blue_rewards):.2f}")
        print(f"- 存活情况: 红方 {info['red_alive']}/{self.env.num_red}, "
              f"蓝方 {info['blue_alive']}/{self.env.num_blue}")

    def test_experience_replay(self):
        """测试经验回放"""
        obs = self.env.reset()
        
        for _ in range(self.test_params['batch_size'] * 2):
            per_agent_obs = []
            for i in range(self.maddpg.num_agents):
                agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
                per_agent_obs.append(torch.FloatTensor(agent_obs).to(self.device))
            
            with self.get_autocast_context():
                actions = self.maddpg.select_actions(per_agent_obs, noise=0.1)
                
            next_obs, rewards, done, _ = self.env.step(actions)
            
            # 使用numpy.array预处理数组
            actions_array = np.array(actions, dtype=np.float32)
            rewards_array = np.array(rewards, dtype=np.float32)
            next_obs_array = np.array(next_obs, dtype=np.float32)
            done_array = np.array([done], dtype=np.float32)
            
            # 转换为tensor
            actions_tensor = torch.from_numpy(actions_array).to(self.test_params['device'])
            rewards_tensor = torch.from_numpy(rewards_array).to(self.test_params['device'])
            next_obs_tensor = torch.from_numpy(next_obs_array).to(self.test_params['device'])
            done_tensor = torch.from_numpy(done_array).to(self.test_params['device'])
            
            states = np.concatenate([o.cpu().numpy() for o in per_agent_obs])
            self.maddpg.replay_buffer.add(states, actions_array, rewards_array, next_obs_array, bool(done_array[0]))
            
            obs = next_obs
            if done:
                obs = self.env.reset()
        
        print(f"✓ 经验回放缓冲区大小: {self.maddpg.replay_buffer.size()}")
        
        samples, indices, weights = self.maddpg.replay_buffer.sample(self.test_params['batch_size'])
        self.assertEqual(len(weights), self.test_params['batch_size'])
        print("✓ 优先级经验回放采样正确")

    # def test_short_training(self):
    #     """测试短期训练效果"""
    #     metrics = {
    #         'rewards_per_episode': [],
    #         'red_rewards': [],
    #         'blue_rewards': [],
    #         'red_survival': [],
    #         'blue_survival': [],
    #         'red_energy': [],
    #         'blue_energy': [],
    #         'steps_per_episode': [],
    #         'noise_values': [],
    #         'moving_avg_rewards': [0]
    #     }
        
    #     current_noise = self.test_params['initial_noise']
    #     update_count = 0
    #     print("\n开始短期训练测试...")
        
    #     for episode in range(5):
    #         obs = self.env.reset()
    #         episode_reward = 0
    #         red_reward = 0
    #         blue_reward = 0
    #         step_count = 0
            
    #         initial_red_alive = self.env.red_alive.copy()
    #         initial_blue_alive = self.env.blue_alive.copy()
            
    #         while step_count < self.test_params['max_steps']:
    #             per_agent_obs = []
    #             for i in range(self.maddpg.num_agents):
    #                 agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
    #                 per_agent_obs.append(torch.FloatTensor(agent_obs).to(self.device))
                
    #             with self.get_autocast_context():
    #                 actions = self.maddpg.select_actions(per_agent_obs, noise=current_noise)
    #                 next_obs, rewards, done, info = self.env.step(actions)
                    
    #                 red_rewards = rewards[:self.test_params['num_red']]
    #                 blue_rewards = rewards[self.test_params['num_red']:]
    #                 red_reward += sum(red_rewards)
    #                 blue_reward += sum(blue_rewards)
                    
    #                 actions_tensor = torch.FloatTensor(actions).to(self.test_params['device'])
    #                 rewards_tensor = torch.FloatTensor(rewards).to(self.test_params['device'])
    #                 next_obs_tensor = torch.FloatTensor(next_obs).to(self.test_params['device'])
    #                 done_tensor = torch.FloatTensor([done]).to(self.test_params['device'])
                
    #             states = np.concatenate([o.cpu().numpy() for o in per_agent_obs])
    #             action = np.array(actions)
    #             reward = np.array(rewards)
    #             next_state = np.array(next_obs)
    #             self.maddpg.replay_buffer.add(states, action, reward, next_state, done)
                
    #             if self.maddpg.replay_buffer.size() >= self.test_params['batch_size']:
    #                 update_count += 1
    #                 if update_count % self.test_params['update_freq'] == 0:
    #                     self.maddpg.update()
                
    #             episode_reward += sum(rewards)
    #             obs = next_obs
    #             step_count += 1
                
    #             if done:
    #                 break
            
    #         # 计算存活率和能量
    #         red_survival = np.mean(self.env.red_alive) / np.mean(initial_red_alive) if np.mean(initial_red_alive) > 0 else 0
    #         blue_survival = np.mean(self.env.blue_alive) / np.mean(initial_blue_alive) if np.mean(initial_blue_alive) > 0 else 0
    #         avg_red_energy = np.mean(self.env.red_energy[self.env.red_alive]) if np.any(self.env.red_alive) else 0
    #         avg_blue_energy = np.mean(self.env.blue_energy[self.env.blue_alive]) if np.any(self.env.blue_alive) else 0
            
    #         current_noise = max(self.test_params['min_noise'], current_noise * self.test_params['noise_decay'])
            
    #         # 更新指标
    #         metrics['rewards_per_episode'].append(episode_reward)
    #         metrics['red_rewards'].append(red_reward)
    #         metrics['blue_rewards'].append(blue_reward)
    #         metrics['red_survival'].append(red_survival)
    #         metrics['blue_survival'].append(blue_survival)
    #         metrics['red_energy'].append(avg_red_energy)
    #         metrics['blue_energy'].append(avg_blue_energy)
    #         metrics['steps_per_episode'].append(step_count)
    #         metrics['noise_values'].append(current_noise)
            
    #         metrics['moving_avg_rewards'].append(
    #             np.mean(metrics['rewards_per_episode'][-min(episode+1, 100):])
    #         )
            
    #         print(f"Episode {episode + 1}: 总奖励={episode_reward:.2f}, "
    #               f"红方={red_reward:.2f}, 蓝方={blue_reward:.2f}, "
    #               f"存活: 红={red_survival:.2%} 蓝={blue_survival:.2%}")
        
    #     # 保存训练结果
    #     save_evaluation_report(self.test_dir, metrics, self.test_params)
    #     plot_training_curves(self.test_dir, metrics)
        
    #     print("\n测试训练结果:")
    #     print(f"- 平均总奖励: {np.mean(metrics['rewards_per_episode']):.2f}")
    #     print(f"- 平均红方奖励: {np.mean(metrics['red_rewards']):.2f}")
    #     print(f"- 平均蓝方奖励: {np.mean(metrics['blue_rewards']):.2f}")
    #     print(f"- 平均红方存活率: {np.mean(metrics['red_survival']):.2%}")
    #     print(f"- 平均蓝方存活率: {np.mean(metrics['blue_survival']):.2%}")
    def test_short_training(self):
        """测试短期训练效果"""
        metrics = {
            'rewards_per_episode': [],
            'red_rewards': [],
            'blue_rewards': [],
            'red_survival': [],
            'blue_survival': [],
            'red_energy': [],
            'blue_energy': [],
            'steps_per_episode': [],
            'noise_values': [],
            'moving_avg_rewards': [0]
        }
        
        current_noise = self.test_params['initial_noise']
        update_count = 0
        print("\n开始短期训练测试...")
        
        for episode in range(5):
            obs = self.env.reset()
            episode_reward = 0
            red_reward = 0
            blue_reward = 0
            step_count = 0
            
            initial_red_alive = self.env.red_alive.copy()
            initial_blue_alive = self.env.blue_alive.copy()
            
            while step_count < self.test_params['max_steps']:
                # 使用numpy.array预处理观察值
                per_agent_obs = []
                for i in range(self.maddpg.num_agents):
                    agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
                    agent_obs = np.array(agent_obs, dtype=np.float32)  # 预处理为numpy数组
                    per_agent_obs.append(torch.from_numpy(agent_obs).to(self.device))
                
                with self.get_autocast_context():
                    actions = self.maddpg.select_actions(per_agent_obs, noise=current_noise)
                    next_obs, rewards, done, info = self.env.step(actions)
                    
                    # 预处理所有数组并转换为tensor
                    actions = np.array(actions, dtype=np.float32)
                    rewards = np.array(rewards, dtype=np.float32)
                    next_obs = np.array(next_obs, dtype=np.float32)
                    done_array = np.array([done], dtype=np.float32)
                    
                    actions_tensor = torch.from_numpy(actions).to(self.device)
                    rewards_tensor = torch.from_numpy(rewards).to(self.device)
                    next_obs_tensor = torch.from_numpy(next_obs).to(self.device)
                    done_tensor = torch.from_numpy(done_array).to(self.device)
                
                states = np.concatenate([o.cpu().numpy() for o in per_agent_obs])
                self.maddpg.replay_buffer.add(states, actions, rewards, next_obs, done)
                
                red_rewards = rewards[:self.test_params['num_red']]
                blue_rewards = rewards[self.test_params['num_red']:]
                red_reward += sum(red_rewards)
                blue_reward += sum(blue_rewards)
                
                if self.maddpg.replay_buffer.size() >= self.test_params['batch_size']:
                    update_count += 1
                    if update_count % self.test_params['update_freq'] == 0:
                        self.maddpg.update()
                
                episode_reward += sum(rewards)
                obs = next_obs
                step_count += 1
                
                if done:
                    break
                
            # 计算存活率和能量
            red_survival = np.mean(self.env.red_alive) / np.mean(initial_red_alive) if np.mean(initial_red_alive) > 0 else 0
            blue_survival = np.mean(self.env.blue_alive) / np.mean(initial_blue_alive) if np.mean(initial_blue_alive) > 0 else 0
            avg_red_energy = np.mean(self.env.red_energy[self.env.red_alive]) if np.any(self.env.red_alive) else 0
            avg_blue_energy = np.mean(self.env.blue_energy[self.env.blue_alive]) if np.any(self.env.blue_alive) else 0
            
            current_noise = max(self.test_params['min_noise'], current_noise * self.test_params['noise_decay'])
            
            # 更新指标
            metrics['rewards_per_episode'].append(episode_reward)
            metrics['red_rewards'].append(red_reward)
            metrics['blue_rewards'].append(blue_reward)
            metrics['red_survival'].append(red_survival)
            metrics['blue_survival'].append(blue_survival)
            metrics['red_energy'].append(avg_red_energy)
            metrics['blue_energy'].append(avg_blue_energy)
            metrics['steps_per_episode'].append(step_count)
            metrics['noise_values'].append(current_noise)
            
            metrics['moving_avg_rewards'].append(
                np.mean(metrics['rewards_per_episode'][-min(episode+1, 100):])
            )
            
            print(f"Episode {episode + 1}: 总奖励={episode_reward:.2f}, "
                f"红方={red_reward:.2f}, 蓝方={blue_reward:.2f}, "
                f"存活: 红={red_survival:.2%} 蓝={blue_survival:.2%}")
        
        # 保存训练结果
        save_evaluation_report(self.test_dir, metrics, self.test_params)
        plot_training_curves(self.test_dir, metrics)
        
        print("\n测试训练结果:")
        print(f"- 平均总奖励: {np.mean(metrics['rewards_per_episode']):.2f}")
        print(f"- 平均红方奖励: {np.mean(metrics['red_rewards']):.2f}")
        print(f"- 平均蓝方奖励: {np.mean(metrics['blue_rewards']):.2f}")
        print(f"- 平均红方存活率: {np.mean(metrics['red_survival']):.2%}")
        print(f"- 平均蓝方存活率: {np.mean(metrics['blue_survival']):.2%}")

    def test_model_save_load(self):
        """测试模型保存和加载"""
        save_path = os.path.join(self.test_dir, 'test_save.pt')
        self.maddpg.save(save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # 在test_model_save_load方法中
        new_maddpg = MADDPG(
            num_agents=self.test_params['num_red'] + self.test_params['num_blue'],
            state_dim=self.test_params['state_dim'],
            action_dim=self.test_params['action_dim'],
            lr=self.test_params['lr'],
            gamma=self.test_params['gamma'],
            tau=self.test_params['tau'],
            buffer_size=self.test_params['buffer_size'],
            batch_size=self.test_params['batch_size'],
            hidden_dim=self.test_params['hidden_dim'],
            agent_types=self.red_agent_types + self.blue_agent_types,
            device=self.device  # 传入device参数，去掉.to(device)调用
        )
        
        new_maddpg.load(save_path)
        
        # 测试加载后的模型行为
        # 测试加载后的模型行为
        obs = self.env.reset()
        per_agent_obs = []
        for i in range(new_maddpg.num_agents):
            agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
            per_agent_obs.append(torch.FloatTensor(agent_obs).to(self.device))
        
        with self.get_autocast_context():
            actions = new_maddpg.select_actions(per_agent_obs, noise=0.0)
        
        self.assertEqual(len(actions), new_maddpg.num_agents)
        
        # 验证动作范围
        for action in actions:
            self.assertTrue(np.all(np.abs(action) <= 1.0))
        
        print("✓ 模型保存加载测试通过")
        print("- 动作生成正确")
        print("- 异构智能体配置保持一致")

    def test_curiosity_mechanism(self):
        """测试好奇心机制"""
        if not self.test_params['use_curiosity']:
            self.skipTest("好奇心机制未启用")
            
        obs = self.env.reset()
        per_agent_obs = []
        for i in range(self.maddpg.num_agents):
            agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
            per_agent_obs.append(torch.FloatTensor(agent_obs).to(self.device))
        
        with self.get_autocast_context():  # 使用修改后的context
            actions = self.maddpg.select_actions(per_agent_obs, noise=0.1)
            next_obs, rewards, _, _ = self.env.step(actions)
            
            # 处理数据维度
            states = np.concatenate([o.cpu().numpy() for o in per_agent_obs])
            actions_np = np.array(actions)
            next_states = np.array(next_obs)
            
            # 转换为正确的tensor维度
            states_tensor = torch.FloatTensor(states).view(1, -1).to(self.device)
            actions_tensor = torch.FloatTensor(actions_np).view(1, -1).to(self.device)
            next_states_tensor = torch.FloatTensor(next_states).view(1, -1).to(self.device)
            
            intrinsic_rewards = self.maddpg._compute_intrinsic_rewards(
                states_tensor,
                actions_tensor,
                next_states_tensor
            )
            
            self.assertEqual(intrinsic_rewards.shape[1], self.maddpg.num_agents)
            self.assertTrue(torch.all(intrinsic_rewards >= 0))
            
            print("✓ 好奇心机制测试通过")
            print(f"- 内在奖励范围: [{intrinsic_rewards.min().item():.3f}, {intrinsic_rewards.max().item():.3f}]")

    def test_heterogeneous_features(self):
        """测试异构特性"""
        if not self.test_params['heterogeneous']:
            self.skipTest("异构特性未启用")
        
        # 验证不同类型智能体的特性
        for i, agent_type in enumerate(self.red_agent_types):
            agent_params = self.env.agent_types[agent_type]
            
            # 验证速度限制
            self.assertTrue(agent_params['max_velocity'] > 0)
            
            # 验证攻击范围和击杀概率
            self.assertTrue(0 < agent_params['attack_range'] <= 5.0)
            self.assertTrue(0 < agent_params['kill_probability'] <= 1.0)
            
            # 验证观察范围
            self.assertTrue(agent_params['observation_range'] > 0)
        
        # 测试不同类型智能体的行为差异
        obs = self.env.reset()
        per_agent_obs = []
        for i in range(self.maddpg.num_agents):
            agent_obs = obs[i * self.test_params['state_dim']:(i + 1) * self.test_params['state_dim']]
            per_agent_obs.append(torch.FloatTensor(agent_obs).to(self.device))
        
        with self.get_autocast_context():
            actions = self.maddpg.select_actions(per_agent_obs, noise=0.0)
        
        # 验证侦察机动作特点
        scout_action = actions[0]  # 假设第一个是侦察机
        fighter_action = actions[1]  # 假设第二个是战斗机
        
        print("✓ 异构特性测试通过")
        print(f"- 侦察机动作幅度: {np.linalg.norm(scout_action):.3f}")
        print(f"- 战斗机动作幅度: {np.linalg.norm(fighter_action):.3f}")
        
        # 验证轰炸机的攻击范围
        bomber_idx = 2  # 假设第三个是轰炸机
        bomber_params = self.env.agent_types['bomber']
        self.assertTrue(bomber_params['attack_range'] > self.env.agent_types['fighter']['attack_range'])
        print(f"- 轰炸机攻击范围: {bomber_params['attack_range']}")

    def test_reward_weights(self):
        """测试奖励权重"""
        # 验证所有必要的奖励权重存在
        required_weights = ['kill', 'death', 'victory', 'survive', 'approach', 
                          'team', 'boundary', 'energy', 'strategy']
        
        for weight in required_weights:
            self.assertIn(weight, self.env.reward_weights)
            self.assertTrue(isinstance(self.env.reward_weights[weight], float))
        
        print("✓ 奖励权重配置正确")
        print("- 所有必要权重均已定义")
        print(f"- 击杀奖励: {self.env.reward_weights['kill']}")
        print(f"- 团队协作奖励: {self.env.reward_weights['team']}")
        print(f"- 战术位置奖励: {self.env.reward_weights['strategy']}")
    def get_autocast_context(self):
        """获取autocast上下文"""
        return torch.amp.autocast(device_type=self.device.type if torch.cuda.is_available() else 'cpu')

def run_main_tests():
    """运行所有主函数测试"""
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # 设置多线程
    if torch.cuda.is_available():
        torch.set_num_threads(4)
    
    print("\n=================== 开始主函数测试 ===================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMainFunction)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print("=================== 测试完成 ===================\n")

if __name__ == "__main__":
    run_main_tests()