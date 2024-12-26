import os
import sys
import numpy as np
from envs.mozi_env import MoziEnv
from envs.env_config import EnvConfig

def setup_test_env():
    """初始化测试环境"""
    print("\n=== 初始化测试环境 ===")
    try:
        # 设置墨子路径
        if not os.environ.get("MOZIPATH"):
            os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
        print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")
        
        # 创建环境
        env = MoziEnv(
            IP=EnvConfig.SERVER_IP,
            AIPort=EnvConfig.SERVER_PORT,
            platform=EnvConfig.PLATFORM,
            scenario_name=EnvConfig.SCENARIO_NAME,
            simulate_compression=EnvConfig.SIMULATE_COMPRESSION,
            duration_interval=EnvConfig.DURATION_INTERVAL,
            synchronous=True
        )
        
        # 重置环境
        scenario = env.reset()
        assert scenario is not None, "环境重置失败!"
        
        print("环境初始化成功")
        return env, scenario
        
    except Exception as e:
        print(f"环境初始化失败: {e}")
        raise

# def test_basic_actions():
#     """测试基本动作执行"""
#     print("\n=== 测试基本动作执行 ===")
#     try:
#         # 初始化环境
#         env, scenario = setup_test_env()
#
#         # 获取双方单位
#         redside = scenario.get_side_by_name("红方")
#         blueside = scenario.get_side_by_name("蓝方")
#
#         print("\n当前场景信息:")
#         print(f"红方单位数量: {len(redside.aircrafts)}")
#         print(f"蓝方单位数量: {len(blueside.aircrafts)}")
#
#         # 获取观察
#         obs = env.get_observations()
#         units = list(obs.keys())
#
#         if not units:
#             raise ValueError("没有找到可用的单位")
#
#         print(f"\n发现的单位: {units}")
#
#         # 测试转向动作
#         print("\n1. 测试转向动作")
#         test_unit = units[0]
#         test_action = {
#             test_unit: np.array([0.5, 0, 0])  # 右转,不变速,不攻击
#         }
#
#         before_state = obs[test_unit].copy()
#         env.execute_action(test_action)
#         after_state = env.get_observations()[test_unit]
#
#         print(f"转向测试结果:")
#         print(f"原航向: {before_state['heading']:.2f}")
#         print(f"新航向: {after_state['heading']:.2f}")
#
#         return True
#
#     except Exception as e:
#         print(f"基本动作测试失败: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
def test_basic_actions():
    """测试基本动作执行"""
    print("\n=== 测试基本动作 ===")
    try:
        # 初始化环境
        env, scenario = setup_test_env()

        # 获取观察
        obs = env.get_observations()
        units = list(obs.keys())

        if not units:
            raise ValueError("没有找到可用的单位")

        print(f"\n发现的单位: {units}")

        # 测试转向动作
        print("\n1. 测试转向动作")
        test_unit = units[0]
        print(f"测试单位: {test_unit}")

        test_action = {
            test_unit: np.array([0.5, 0, 0])  # 右转,不变速,不攻击
        }

        before_state = obs[test_unit].copy()
        success = env.execute_action(test_action)
        print(f"动作执行状态: {'成功' if success else '失败'}")

        after_state = env.get_observations()[test_unit]

        print(f"转向测试结果:")
        print(f"原航向: {before_state['heading']:.2f}")
        print(f"新航向: {after_state['heading']:.2f}")
        print(f"航向变化: {(after_state['heading'] - before_state['heading']):.2f}")

        return True

    except Exception as e:
        print(f"基本动作测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
def test_multi_unit_actions():
    """测试多单位动作执行"""
    print("\n=== 测试多单位动作执行 ===")
    try:
        # 初始化环境
        env, scenario = setup_test_env()
        
        # 获取观察
        obs = env.get_observations()
        
        # 构造多单位动作
        actions = {}
        for i, unit_name in enumerate(list(obs.keys())[:2]):  # 测试前两个单位
            actions[unit_name] = np.array([
                0.3 if i == 0 else -0.3,  # 转向
                0.5,                      # 速度
                0.0                       # 不攻击
            ])
            
        # 执行动作前记录状态
        before_states = {k: obs[k].copy() for k in actions.keys()}
        
        # 执行动作
        env.execute_action(actions)
        
        # 获取执行后状态
        after_states = env.get_observations()
        
        # 打印结果
        for unit in actions.keys():
            print(f"\n单位 {unit} 状态变化:")
            print(f"航向: {before_states[unit]['heading']:.2f} -> {after_states[unit]['heading']:.2f}")
            print(f"速度: {before_states[unit]['speed']:.2f} -> {after_states[unit]['speed']:.2f}")
            
        return True
        
    except Exception as e:
        print(f"多单位动作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_boundary_actions():
    """测试边界动作执行"""
    print("\n=== 测试边界动作执行 ===")
    try:
        # 初始化环境
        env, scenario = setup_test_env()
        
        # 获取观察
        obs = env.get_observations()
        test_unit = list(obs.keys())[0]
        
        # 测试边界值
        boundary_tests = [
            ([1.0, 1.0, 1.0], "最大值测试"),
            ([-1.0, -1.0, 0.0], "最小值测试"),
            ([0.0, 0.0, 0.0], "零值测试")
        ]
        
        for action, test_name in boundary_tests:
            print(f"\n执行{test_name}")
            before_state = env.get_observations()[test_unit].copy()
            
            # 执行动作
            env.execute_action({test_unit: np.array(action)})
            
            # 获取结果
            after_state = env.get_observations()[test_unit]
            
            print(f"状态变化:")
            print(f"航向: {before_state['heading']:.2f} -> {after_state['heading']:.2f}")
            print(f"速度: {before_state['speed']:.2f} -> {after_state['speed']:.2f}")
            
        return True
        
    except Exception as e:
        print(f"边界动作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("\n=== 开始动作系统测试 ===")
    
    test_results = {
        "基本动作测试": test_basic_actions(),
        "多单位动作测试": test_multi_unit_actions(),
        "边界动作测试": test_boundary_actions()
    }
    
    print("\n=== 测试结果总结 ===")
    for test_name, result in test_results.items():
        print(f"{test_name}: {'通过' if result else '失败'}")
        
    all_passed = all(test_results.values())
    print(f"\n总体测试结果: {'全部通过' if all_passed else '存在失败'}")
    
    return all_passed

if __name__ == "__main__":
    test_result = run_all_tests()
    sys.exit(0 if test_result else 1)