# tests/test_env.py

import unittest
import numpy as np
from env.combat_env import CombatEnv

class TestCombatEnv(unittest.TestCase):
    """测试优化后的战斗环境"""
    
    def setUp(self):
        self.env = CombatEnv(
            num_red=3,
            num_blue=3,
            max_steps=300
        )
    
    def test_initial_positions(self):
        """测试初始化位置"""
        obs = self.env.reset()
        
        # 检查红方位置
        red_positions = self.env.red_positions
        self.assertTrue(np.all(red_positions[:, 0] >= -8))
        self.assertTrue(np.all(red_positions[:, 0] <= -3))
        
        # 检查蓝方位置
        blue_positions = self.env.blue_positions
        self.assertTrue(np.all(blue_positions[:, 0] >= 3))
        self.assertTrue(np.all(blue_positions[:, 0] <= 8))
        
        print("✓ 初始位置分布正确")

    def test_reward_mechanism(self):
        """测试奖励机制"""
        self.env.reset()
        
        # 测试击落奖励
        self.env.red_positions = np.array([[0.0, 0.0]], dtype=np.float32)
        self.env.blue_positions = np.array([[0.5, 0.0]], dtype=np.float32)
        
        # 创建完整的动作列表
        actions = []
        for _ in range(self.env.num_red):
            actions.append(np.array([0.1, 0.0], dtype=np.float32))
        for _ in range(self.env.num_blue):
            actions.append(np.array([0.0, 0.0], dtype=np.float32))
            
        _, rewards, _, _ = self.env.step(actions)
        
        # 验证击落奖励
        self.assertTrue(rewards[0] >= self.env.kill_reward)
        self.assertTrue(rewards[self.env.num_red] <= self.env.death_penalty)
        
        print("✓ 击落奖励机制正确")
        
        # 测试生存奖励
        self.env.reset()
        zero_actions = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(self.env.total_agents)]
        _, rewards, _, _ = self.env.step(zero_actions)
        self.assertTrue(np.all(rewards[:self.env.num_red] >= self.env.survive_reward))
        
        print("✓ 生存奖励机制正确")

    def test_battle_duration(self):
        """测试战斗持续时间"""
        self.env.reset()
        steps = 0
        
        # 让双方保持距离
        self.env.red_positions = np.array([[-5.0, 0.0]], dtype=np.float32)
        self.env.blue_positions = np.array([[5.0, 0.0]], dtype=np.float32)
        
        done = False
        zero_actions = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(self.env.total_agents)]
        
        while not done and steps < 50:
            _, _, done, _ = self.env.step(zero_actions)
            steps += 1
        
        self.assertTrue(steps > 1)
        print(f"✓ 战斗可以持续多步: {steps} 步")

    def test_boundary_penalty(self):
        """测试边界惩罚"""
        self.env.reset()
        
        # 将红方移到边界
        self.env.red_positions = np.array([[self.env.boundary - 0.5, 0.0]], dtype=np.float32)
        
        # 创建完整的动作列表
        actions = []
        actions.append(np.array([1.0, 0.0], dtype=np.float32))  # 红方向边界移动
        for _ in range(self.env.total_agents - 1):
            actions.append(np.array([0.0, 0.0], dtype=np.float32))
            
        _, rewards, _, _ = self.env.step(actions)
        
        # 验证边界惩罚
        self.assertTrue(rewards[0] < 0)
        print("✓ 边界惩罚机制正确")

    def test_victory_condition(self):
        """测试胜利条件"""
        self.env.reset()
        
        # 将所有蓝方标记为击落
        self.env.blue_alive = np.zeros(self.env.num_blue, dtype=np.float32)
        
        # 创建完整的动作列表
        actions = [np.array([0.0, 0.0], dtype=np.float32) for _ in range(self.env.total_agents)]
        _, rewards, done, _ = self.env.step(actions)
        
        # 验证胜利奖励和结束状态
        self.assertTrue(done)
        self.assertTrue(np.any(rewards[:self.env.num_red] >= self.env.victory_reward))
        print("✓ 胜利条件和奖励正确")

    def test_episode_metrics(self):
        """测试完整回合的指标"""
        self.env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 100:
            # 创建随机动作
            actions = []
            for _ in range(self.env.total_agents):
                action = np.random.uniform(-1, 1, 2).astype(np.float32)
                actions.append(action)
                
            _, rewards, done, info = self.env.step(actions)
            total_reward += sum(rewards)
            steps += 1
        
        print(f"完整回合测试结果:")
        print(f"- 持续步数: {steps}")
        print(f"- 总奖励: {total_reward:.2f}")
        print(f"- 红方存活: {info['red_alive']}/{self.env.num_red}")
        print(f"- 蓝方存活: {info['blue_alive']}/{self.env.num_blue}")

    def test_realistic_scenario(self):
        """测试真实战斗场景"""
        self.env.reset()
        
        # 设置测试场景
        self.env.red_positions = np.array([
            [-4.0, 0.0],
            [-4.0, 2.0],
            [-4.0, -2.0]
        ], dtype=np.float32)
        
        self.env.blue_positions = np.array([
            [4.0, 0.0],
            [4.0, 2.0],
            [4.0, -2.0]
        ], dtype=np.float32)
        
        # 模拟追击行为
        actions = []
        for i in range(self.env.num_red):
            # 红方向蓝方移动
            direction = (self.env.blue_positions[i] - self.env.red_positions[i])
            direction = direction / np.linalg.norm(direction)
            actions.append(np.array(direction, dtype=np.float32))
        
        for i in range(self.env.num_blue):
            # 蓝方后退
            direction = (self.env.blue_positions[i] - self.env.red_positions[i])
            direction = direction / np.linalg.norm(direction)
            actions.append(np.array(direction, dtype=np.float32))
        
        _, rewards, done, info = self.env.step(actions)
        
        print(f"真实场景测试结果:")
        print(f"- 红方奖励: {rewards[:self.env.num_red]}")
        print(f"- 蓝方奖励: {rewards[self.env.num_red:]}")
        print(f"- 红方存活: {info['red_alive']}/{self.env.num_red}")
        print(f"- 蓝方存活: {info['blue_alive']}/{self.env.num_blue}")

def run_env_tests():
    """运行所有环境测试"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCombatEnv)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    run_env_tests()