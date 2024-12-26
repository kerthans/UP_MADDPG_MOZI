# tests/test_env.py

import unittest
import numpy as np
from env.combat_env import CombatEnv

class TestCombatEnv(unittest.TestCase):
    def setUp(self):
        # 创建同质环境
        self.env = CombatEnv(
            num_red=3,
            num_blue=3,
            state_dim=9,  # 修正为9维：位置(2) + 速度(2) + 相对速度(2) + 加速度(2) + 状态(1)
            action_dim=2,
            max_steps=200,
            heterogeneous=False
        )
        
        # 创建异构环境
        self.hetero_env = CombatEnv(
            num_red=3,
            num_blue=3,
            state_dim=9,  # 同样修正为9维
            action_dim=2,
            max_steps=200,
            heterogeneous=True,
            red_agent_types=['scout', 'fighter', 'bomber'],
            blue_agent_types=['fighter', 'fighter', 'fighter']
        )
    
    def test_state_space(self):
        """测试状态空间"""
        obs = self.env.reset()
        
        # 验证状态维度
        self.assertEqual(len(obs), self.env.total_agents * self.env.state_dim)
        
        # 验证观察值范围
        self.assertTrue(np.all(np.isfinite(obs)))
        
        # 检查异构环境的状态空间
        hetero_obs = self.hetero_env.reset()
        self.assertEqual(len(hetero_obs), self.hetero_env.total_agents * self.hetero_env.state_dim)
        
        print("✓ 状态空间维度和范围正确")

    def test_initial_positions(self):
        """测试初始化位置"""
        obs = self.env.reset()
        
        # 检查红方位置是否在合理范围内
        red_positions = self.env.red_positions
        self.assertTrue(np.all(np.abs(red_positions[:, 0]) <= self.env.boundary))
        
        # 检查蓝方位置
        blue_positions = self.env.blue_positions
        self.assertTrue(np.all(np.abs(blue_positions[:, 0]) <= self.env.boundary))
        
        # 检查初始速度
        self.assertTrue(np.all(self.env.red_velocities == 0))
        self.assertTrue(np.all(self.env.blue_velocities == 0))
        
        # 检查初始能量
        self.assertTrue(np.all(self.env.red_energy == 100))
        self.assertTrue(np.all(self.env.blue_energy == 100))
        
        print("✓ 初始位置、速度和能量正确")

    def test_heterogeneous_properties(self):
        """测试异构特性"""
        self.hetero_env.reset()
        
        # 验证不同类型智能体的参数
        for i, agent_type in enumerate(self.hetero_env.red_agent_types):
            agent_params = self.hetero_env.agent_types[agent_type]
            
            # 检查速度限制
            max_velocity = agent_params['max_velocity']
            action = np.array([1.0, 1.0])
            action_norm = np.linalg.norm(action)
            if action_norm > 1:
                action = action / action_norm  # 归一化确保动作在[-1,1]范围内
                
            # 验证速度不超过限制
            self.assertTrue(max_velocity * np.linalg.norm(action) <= agent_params['max_velocity'])
            
            # 检查攻击范围
            self.assertTrue(agent_params['attack_range'] > 0)
            self.assertTrue(agent_params['kill_probability'] <= 1.0)
        
        print("✓ 异构特性正确")

    def test_observation_structure(self):
        """测试观察值结构"""
        obs = self.env.reset()
        
        # 检查每个智能体的观察
        for i in range(self.env.total_agents):
            agent_obs = obs[i*self.env.state_dim:(i+1)*self.env.state_dim]
            
            # 位置 (2维)
            self.assertTrue(np.all(np.abs(agent_obs[:2]) <= self.env.boundary))
            
            # 速度 (2维)
            if i < self.env.num_red:
                max_vel = self.env.agent_types[self.env.red_agent_types[i]]['max_velocity']
            else:
                max_vel = self.env.agent_types[self.env.blue_agent_types[i-self.env.num_red]]['max_velocity']
            self.assertTrue(np.all(np.abs(agent_obs[2:4]) <= max_vel))
            
            # 相对速度 (2维)
            self.assertTrue(np.all(np.isfinite(agent_obs[4:6])))
            
            # 加速度 (2维)
            self.assertTrue(np.all(np.isfinite(agent_obs[6:8])))
            
            # 存活状态 (1维)
            self.assertTrue(agent_obs[8] in [0, 1])
        
        print("✓ 观察值结构正确")

    def test_reward_mechanism(self):
        """测试奖励机制"""
        self.env.reset()
        
        # 测试击杀奖励
        self.env.red_positions[0] = np.array([0.0, 0.0])
        self.env.blue_positions[0] = np.array([1.0, 0.0])
        
        actions = [np.zeros(2) for _ in range(self.env.total_agents)]
        _, rewards, _, _ = self.env.step(actions)
        
        # 验证基础奖励
        self.assertTrue(np.all(np.abs(rewards) < np.inf))
        self.assertTrue(any(r != 0 for r in rewards))  # 至少有一些奖励
        
        # 测试团队奖励
        self.env.reset()
        self.env.red_positions = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0]
        ])
        
        _, rewards, _, _ = self.env.step(actions)
        self.assertTrue(all(r >= self.env.reward_weights['survive'] for r in rewards[:3]))
        
        print("✓ 奖励机制正确")

    def test_energy_mechanics(self):
        """测试能量机制"""
        self.env.reset()
        
        initial_energy = self.env.red_energy[0]
        
        # 执行高速运动
        actions = [np.array([1.0, 1.0])]
        actions.extend([np.zeros(2) for _ in range(self.env.total_agents - 1)])
        
        self.env.step(actions)
        
        # 验证能量消耗
        self.assertTrue(self.env.red_energy[0] < initial_energy)
        
        print("✓ 能量机制正确")

    def test_battle_dynamics(self):
        """测试战斗动态"""
        self.env.reset()
        
        # 设置对抗场景
        self.env.red_positions = np.array([
            [-2.0, 0.0],
            [-2.0, 2.0],
            [-2.0, -2.0]
        ])
        self.env.blue_positions = np.array([
            [2.0, 0.0],
            [2.0, 2.0],
            [2.0, -2.0]
        ])
        
        total_steps = 0
        done = False
        total_red_rewards = 0
        total_blue_rewards = 0
        
        while not done and total_steps < 50:
            # 模拟追击和规避行为
            actions = []
            for i in range(self.env.num_red):
                direction = self.env.blue_positions[i] - self.env.red_positions[i]
                direction = direction / np.maximum(np.linalg.norm(direction), 1e-6)
                actions.append(direction * 0.5)
            
            for i in range(self.env.num_blue):
                direction = self.env.blue_positions[i] - self.env.red_positions[i]
                direction = direction / np.maximum(np.linalg.norm(direction), 1e-6)
                actions.append(-direction * 0.5)
            
            _, rewards, done, info = self.env.step(actions)
            total_red_rewards += sum(rewards[:self.env.num_red])
            total_blue_rewards += sum(rewards[self.env.num_red:])
            total_steps += 1
        
        print(f"战斗动态测试结果:")
        print(f"- 持续步数: {total_steps}")
        print(f"- 红方累计奖励: {total_red_rewards:.2f}")
        print(f"- 蓝方累计奖励: {total_blue_rewards:.2f}")
        print(f"- 红方存活: {info['red_alive']}/{self.env.num_red}")
        print(f"- 蓝方存活: {info['blue_alive']}/{self.env.num_blue}")
        print("✓ 战斗动态正确")

    def test_tactical_advantage(self):
        """测试战术位置优势计算"""
        self.env.reset()
        
        # 测试理想攻击角度
        position = np.array([0.0, 0.0])
        velocity = np.array([np.sqrt(3)/2, 0.5])  # 30度角
        target_center = np.array([1.0, 0.0])
        
        advantage = self.env._calculate_tactical_advantage(position, velocity, target_center)
        self.assertTrue(0 <= advantage <= 1)
        
        # 测试非理想角度
        velocity_bad = np.array([-1.0, 0.0])  # 180度角
        advantage_bad = self.env._calculate_tactical_advantage(position, velocity_bad, target_center)
        self.assertTrue(advantage_bad < advantage)
        
        print("✓ 战术优势计算正确")


def run_env_tests():
    """运行所有环境测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCombatEnv)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    run_env_tests()