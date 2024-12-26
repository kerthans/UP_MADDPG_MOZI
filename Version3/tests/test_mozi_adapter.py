# tests/test_mozi_adapter.py
import os
import sys
import numpy as np
from envs.mozi_adapter import MoziAdapter
from envs.mozi_env import MoziEnv
from envs.env_config import EnvConfig
from envs.state_handler import StateHandler
from envs.combat_mechanics import CombatMechanics



def setup_environment():
    """设置测试环境"""
    # 设置墨子路径
    if not os.environ.get("MOZIPATH"):
        os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
    print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")

    # 基础配置
    config = {
        'num_red': 2,
        'num_blue': 3,
        'max_steps': getattr(EnvConfig, 'MAX_STEPS', 30),
        'field_size': 100000.0,
        'attack_range': getattr(EnvConfig, 'ATTACK_RANGE', 25000.0),
        'min_speed': getattr(EnvConfig, 'MIN_SPEED', 150.0),
        'max_speed': getattr(EnvConfig, 'MAX_SPEED', 400.0),
        'max_turn_rate': np.pi / 6,
        'hit_probability': 0.8,
        'num_threads': 8
    }

    # 创建组件
    state_handler = StateHandler(config)
    combat_mechanics = CombatMechanics(config)

    return config, state_handler, combat_mechanics


def test_basic_env():
    """基础环境测试"""
    print("\n=== 测试基础墨子环境 ===")
    try:
        if not os.environ.get("MOZIPATH"):
            os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH

        # 创建基础环境
        env = MoziEnv(
            IP=EnvConfig.SERVER_IP,
            AIPort=EnvConfig.SERVER_PORT,
            platform=EnvConfig.PLATFORM,
            scenario_name=EnvConfig.SCENARIO_NAME,
            simulate_compression=EnvConfig.SIMULATE_COMPRESSION,
            duration_interval=EnvConfig.DURATION_INTERVAL,
            synchronous=EnvConfig.SYNCHRONOUS
        )

        # 测试环境重置
        scenario = env.reset()
        assert scenario is not None, "基础环境重置失败"

        # 获取双方
        redside = scenario.get_side_by_name("红方")
        blueside = scenario.get_side_by_name("蓝方")

        print(f"红方单位数量: {len(redside.aircrafts)}")
        print(f"蓝方单位数量: {len(blueside.aircrafts)}")

        return True
    except Exception as e:
        print(f"基础环境测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_adapter():
    """测试适配器环境"""
    try:
        print("\n=== 开始测试适配器环境 ===")

        # 设置环境
        config, state_handler, combat_mechanics = setup_environment()

        # 创建适配器环境
        print("\n1. 创建适配器环境...")
        env = MoziAdapter(
            num_red=config['num_red'],
            num_blue=config['num_blue'],
            max_steps=config['max_steps'],
            env_config=EnvConfig,
            state_handler=state_handler,
            combat_mechanics=combat_mechanics
        )

        # 测试环境重置
        print("\n2. 测试环境重置...")
        obs = env.reset()
        assert obs is not None, "重置失败：观察值为空"
        print("观察空间维度:", env.observation_space.shape)
        print("动作空间维度:", env.action_space.shape)

        # 验证观察格式
        print("\n3. 验证观察格式...")
        for unit_id, unit_obs in obs.items():
            print(f"\n单位 {unit_id} 观察:")
            print(f"维度: {unit_obs.shape}")
            print(f"数值范围: [{unit_obs.min():.3f}, {unit_obs.max():.3f}]")
            assert not np.any(np.isnan(unit_obs)), f"单位 {unit_id} 包含NaN值"
            assert not np.any(np.isinf(unit_obs)), f"单位 {unit_id} 包含Inf值"

        # 测试动作执行
        print("\n4. 测试动作执行...")
        test_actions = {
            'red_0': np.array([0.5, 0.8, 0.0]),  # 右转,加速,不攻击
            'red_1': np.array([-0.3, 0.6, 1.0]),  # 左转,中速,攻击
            'blue_0': np.array([0.2, 0.7, 0.0]),
            'blue_1': np.array([-0.1, 0.5, 1.0]),
            'blue_2': np.array([0.4, 0.9, 0.0])
        }

        next_obs, rewards, done, info = env.step(test_actions)

        print("\n=== 测试完成 ===")
        return True

    except Exception as e:
        print(f"\n测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 首先测试基础环境
    print("\n开始测试基础环境...")
    basic_test_result = test_basic_env()
    print(f"\n基础环境测试{'成功' if basic_test_result else '失败'}")

    if basic_test_result:
        # 测试适配器
        print("\n开始测试适配器...")
        adapter_test_result = test_adapter()
        print(f"\n适配器测试{'成功' if adapter_test_result else '失败'}")

        if adapter_test_result:
            print("\n所有测试通过!")
            sys.exit(0)
        else:
            print("\n适配器测试失败!")
            sys.exit(1)
    else:
        print("\n基础环境测试失败!")
        sys.exit(1)