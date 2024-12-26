import os
import sys
import time
import logging
import numpy as np
import torch
from datetime import datetime
from envs.mozi_adapter import MoziAdapter
from envs.env_config import EnvConfig
from agents.baseline import MADDPG
from envs.state_handler import StateHandler
from envs.combat_mechanics import CombatMechanics

# 配置日志
def setup_logging():
    """配置日志输出"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_environment(env_config=None):
    """设置仿真环境"""
    if not os.environ.get("MOZIPATH"):
        mozi_path = getattr(env_config or EnvConfig, 'MOZI_PATH', None)
        if mozi_path:
            os.environ["MOZIPATH"] = mozi_path
            logging.info(f"设置MOZIPATH为: {mozi_path}")
        else:
            raise EnvironmentError("未找到MOZIPATH环境变量或配置")
    
    # 环境配置
    config = {
        'num_red': 2,  # 红方单位数量
        'num_blue': 3,  # 蓝方单位数量
        'max_steps': getattr(env_config or EnvConfig, 'MAX_STEPS', 200),
        'field_size': 100000.0,  # 战场大小
        'attack_range': getattr(env_config or EnvConfig, 'ATTACK_RANGE', 25000.0),
        'min_speed': getattr(env_config or EnvConfig, 'MIN_SPEED', 150.0),
        'max_speed': getattr(env_config or EnvConfig, 'MAX_SPEED', 400.0),
        'random_seed': 99  # 固定随机种子以便复现
    }
    
    return config

def load_model(maddpg, model_path):
    """加载预训练模型"""
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            maddpg.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"成功加载模型: {model_path}")
            return True
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return False
    return False

def run_simulation(env, maddpg, config, logger):
    """运行单次推演仿真"""
    try:
        # 重置环境
        obs = env.reset()
        logger.info("环境已重置，开始推演")
        
        episode_reward = 0
        step_count = 0
        simulation_data = []
        
        # 推演主循环
        for step in range(config['max_steps']):
            step_count += 1
            
            # 获取动作
            with torch.no_grad():  # 关闭梯度计算
                actions = maddpg.select_actions(obs)
            
            # 执行动作
            next_obs, rewards, done, info = env.step(actions)
            
            # 记录推演数据
            step_data = {
                'step': step + 1,
                'red_alive': info['red_alive'],
                'blue_alive': info['blue_alive'],
                'rewards': rewards
            }
            simulation_data.append(step_data)
            
            # 更新状态
            obs = next_obs
            episode_reward += sum(rewards.values() if isinstance(rewards, dict) else rewards)
            
            # 输出推演信息
            if (step + 1) % 10 == 0:  # 每10步输出一次信息
                logger.info(f"Step {step + 1}/{config['max_steps']}")
                logger.info(f"红方单位数: {info['red_alive']}, 蓝方单位数: {info['blue_alive']}")
                logger.info(f"当前累计奖励: {episode_reward:.2f}")
            
            if done:
                logger.info(f"推演在第 {step + 1} 步结束")
                break
        
        return {
            'steps': step_count,
            'total_reward': episode_reward,
            'simulation_data': simulation_data
        }
        
    except Exception as e:
        logger.error(f"推演过程出错: {str(e)}")
        raise

def save_simulation_results(results, logger):
    """保存推演结果"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "simulation_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 保存结果到文件
        result_file = os.path.join(results_dir, f"sim_results_{timestamp}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"推演总步数: {results['steps']}\n")
            f.write(f"总奖励值: {results['total_reward']:.2f}\n\n")
            
            f.write("详细推演数据:\n")
            for step_data in results['simulation_data']:
                f.write(f"Step {step_data['step']}:\n")
                f.write(f"  红方单位数: {step_data['red_alive']}\n")
                f.write(f"  蓝方单位数: {step_data['blue_alive']}\n")
                f.write(f"  奖励: {step_data['rewards']}\n\n")
        
        logger.info(f"推演结果已保存到: {result_file}")
        
    except Exception as e:
        logger.error(f"保存结果失败: {str(e)}")

def main():
    """主函数"""
    # 设置日志
    logger = setup_logging()
    logger.info("=== 开始墨子推演仿真 ===")
    
    try:
        # 1. 环境设置
        config = setup_environment(EnvConfig)
        logger.info("环境配置完成")
        
        # 2. 创建环境
        env = MoziAdapter(
            num_red=config['num_red'],
            num_blue=config['num_blue'],
            max_steps=config['max_steps'],
            env_config=EnvConfig
        )
        logger.info("墨子环境创建成功")
        
        # 3. 初始化MADDPG网络
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        maddpg = MADDPG(config['num_red'], config['num_blue'], obs_dim, act_dim)
        logger.info(f"MADDPG网络初始化完成 (obs_dim: {obs_dim}, act_dim: {act_dim})")
        
        # 4. 加载预训练模型（如果有）
        model_path = "models/pretrained_maddpg.pth"
        if load_model(maddpg, model_path):
            logger.info("已加载预训练模型")
        else:
            logger.warning("未找到预训练模型，将使用初始化网络")
        
        # 5. 运行推演
        logger.info("开始运行推演...")
        simulation_results = run_simulation(env, maddpg, config, logger)
        
        # 6. 保存结果
        save_simulation_results(simulation_results, logger)
        
        logger.info("=== 推演仿真完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"推演过程发生错误: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        # 清理环境
        try:
            env.close()
            logger.info("环境已清理")
        except:
            pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)