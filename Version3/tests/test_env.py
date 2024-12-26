# test_env.py
import os
from envs.mozi_env import MoziEnv
from envs.env_config import EnvConfig

def test_env():
    # 设置墨子安装目录
    if not os.environ.get("MOZIPATH"):
        os.environ["MOZIPATH"] = EnvConfig.MOZI_PATH
    print(f"当前MOZIPATH设置为: {os.environ['MOZIPATH']}")
    
    try:
        # 创建环境
        print("Creating environment...")
        env = MoziEnv(
            IP=EnvConfig.SERVER_IP,            # IP地址
            AIPort=EnvConfig.SERVER_PORT,      # 端口
            platform=EnvConfig.PLATFORM,       # 平台
            scenario_name=EnvConfig.SCENARIO_NAME,  # 想定名称
            simulate_compression=EnvConfig.SIMULATE_COMPRESSION,  # 推演压缩比
            duration_interval=EnvConfig.DURATION_INTERVAL,       # 决策步长
            synchronous=EnvConfig.SYNCHRONOUS  # 是否同步
        )
        
        # 重置环境
        print("Resetting environment...")
        scenario = env.reset()
        
        # 验证场景加载
        assert scenario is not None, "场景加载失败!"
        
        # 获取推演方
        redside = scenario.get_side_by_name("红方")
        blueside = scenario.get_side_by_name("蓝方")
        
        print("场景信息:")
        print(f"红方单位数量: {len(redside.aircrafts)}")
        print(f"蓝方单位数量: {len(blueside.aircrafts)}")
        
        # 列出所有单位
        print("\n红方单位:")
        for k, v in redside.aircrafts.items():
            print(f"ID: {k}, 名称: {v.strName}")
            
        print("\n蓝方单位:")
        for k, v in blueside.aircrafts.items():
            print(f"ID: {k}, 名称: {v.strName}")
        
        # 测试动作执行
        print("\nExecuting actions...")
        actions = {
            'red_0': [0.5, 0.8, 0],   # 右转,加速,不攻击
            'red_1': [-0.3, 0.6, 1],  # 左转,中速,攻击
            'blue_0': [0.2, 0.7, 0],
            'blue_1': [-0.1, 0.5, 1],
            'blue_2': [0.4, 0.9, 0]
        }
        
        env.execute_action(actions)
        
        # 获取观察
        obs = env.get_observations()
        print("\nObservations:")
        for k, v in obs.items():
            print(f"{k}: {v}")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTest finished.")

if __name__ == "__main__":
    test_env()