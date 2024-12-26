# env_config.py
import os

class EnvConfig:
    # 服务器参数
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = "6060"
    PLATFORM = "windows"
    
    # 想定参数
    SCENARIO_NAME = "maddpg.scen"    # 确保这个与你的想定文件名一致
    
    # 推演参数
    SIMULATE_COMPRESSION = 3          # 推演压缩比
    DURATION_INTERVAL = 15           # 决策步长，单位：秒
    SYNCHRONOUS = True               # 同步模式
    APP_MODE = 1                     # 应用模式：1-windows本地模式
    
    # 设置墨子安装目录下bin目录为MOZIPATH
    # 修改为你的墨子安装路径
    MOZI_PATH = "D:\\墨子平台\\Mozi\\MoziServer\\bin"  
    
    # 推演设置
    MAX_EPISODES = 5000      # 最大训练回合数
    MAX_STEPS = 200          # 每回合最大步数
    
    # 无人机参数
    MIN_SPEED = 150.0       # 最小速度(节)
    MAX_SPEED = 400.0       # 最大速度(节)
    ATTACK_RANGE = 25000.0  # 攻击范围(米)