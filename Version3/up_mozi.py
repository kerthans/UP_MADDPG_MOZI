import os
import sys
import time
import logging
import numpy as np
import torch
from datetime import datetime
from envs.mozi_adapter import MoziAdapter
from envs.env_config import EnvConfig
from agents.up import MADDPG
from envs.state_handler import StateHandler
from envs.combat_mechanics import CombatMechanics

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
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_environment():
    """设置仿真环境参数"""
    if not os.environ.get("MOZIPATH"):
        mozi_path = getattr(EnvConfig, 'MOZI_PATH', None)
        if mozi_path:
            os.environ["MOZIPATH"] = mozi_path
            logging.info(f"设置MOZIPATH为: {mozi_path}")
        else:
            raise EnvironmentError("未找到MOZIPATH环境变量或配置")
    
    # 环境配置
    config = {
        'num_red': 2,  # 红方单位数量
        'num_blue': 3,  # 蓝方单位数量
        'max_steps': getattr(EnvConfig, 'MAX_STEPS', 200),
        'field_size': 100000.0,  # 战场大小
        'attack_range': getattr(EnvConfig, 'ATTACK_RANGE', 25000.0),
        'min_speed': getattr(EnvConfig, 'MIN_SPEED', 150.0),
        'max_speed': getattr(EnvConfig, 'MAX_SPEED', 400.0)
    }
    
    # MADDPG特定配置
    config.update({
        'n_step': 3,  # n步经验回放
        'gamma': 0.99,  # 折扣因子
        'tau': 0.01,  # 目标网络软更新系数
        'hidden_dim': 512,  # 隐藏层维度
        'dropout': 0.3,  # Dropout比率
        'noise_scale': 0.1,  # 动作探索噪声比例
    })
    
    return config

def load_model(maddpg, model_path):
    """加载预训练模型"""
    try:
        if os.path.exists(model_path):
            maddpg.load(model_path)
            logging.info(f"成功加载模型: {model_path}")
            return True
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        return False
    return False

# def run_simulation(env, maddpg, config, logger):
#     """运行单次推演仿真"""
#     try:
#         # 重置环境和噪声
#         obs = env.reset()
#         for agent in maddpg.red_agents + maddpg.blue_agents:
#             agent.reset_noise()
#
#         logger.info("环境已重置，开始推演")
#
#         episode_reward = 0
#         step_count = 0
#         simulation_data = []
#
#         # 推演主循环
#         for step in range(config['max_steps']):
#             step_count += 1
#
#             # 选择动作并添加探索噪声
#             with torch.no_grad():
#                 actions = maddpg.select_actions(obs, noise_scale=config['noise_scale'])
#
#             # 执行动作
#             next_obs, rewards, done, info = env.step(actions)
#
#             # 记录本步推演数据
#             step_data = {
#                 'step': step + 1,
#                 'red_alive': info['red_alive'],
#                 'blue_alive': info['blue_alive'],
#                 'rewards': rewards,
#                 'actions': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in actions.items()}
#             }
#             simulation_data.append(step_data)
#
#             # 存储经验到优先经验回放缓冲区
#             maddpg.store_transition(obs, actions, rewards, next_obs, done)
#
#             # 更新状态
#             obs = next_obs
#             episode_reward += sum(rewards.values() if isinstance(rewards, dict) else rewards)
#
#             # 定期输出状态信息
#             if (step + 1) % 10 == 0 or done:
#                 logger.info(f"\nStep {step + 1}/{config['max_steps']}")
#                 logger.info(f"红方单位数: {info['red_alive']}")
#                 logger.info(f"蓝方单位数: {info['blue_alive']}")
#                 logger.info(f"当前累计奖励: {episode_reward:.2f}")
#
#             if done:
#                 logger.info(f"\n推演在第 {step + 1} 步结束")
#                 break
#
#         return {
#             'steps': step_count,
#             'total_reward': episode_reward,
#             'simulation_data': simulation_data,
#             'final_state': {
#                 'red_alive': info['red_alive'],
#                 'blue_alive': info['blue_alive']
#             }
#         }
#
#     except Exception as e:
#         logger.error(f"推演过程出错: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         raise
def run_simulation(env, maddpg, config, logger):
    """运行单次推演仿真"""
    try:
        # 重置环境和噪声
        obs = env.reset()
        for agent in maddpg.red_agents + maddpg.blue_agents:
            agent.reset_noise()

        logger.info("环境已重置，开始推演")

        episode_reward = 0
        step_count = 0
        simulation_data = []

        # 推演主循环
        for step in range(config['max_steps']):
            step_count += 1

            # 选择动作并添加探索噪声
            with torch.no_grad():
                actions = maddpg.select_actions(obs, noise_scale=config['noise_scale'])

            # 执行动作
            next_obs, rewards, done, info = env.step(actions)

            # 检查是否所有单位都已经阵亡
            all_units_dead = (info['red_alive'] == 0 and info['blue_alive'] == 0)

            # 记录本步推演数据
            step_data = {
                'step': step + 1,
                'red_alive': info['red_alive'],
                'blue_alive': info['blue_alive'],
                'rewards': rewards,
                'actions': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in actions.items()}
            }
            simulation_data.append(step_data)

            # 存储经验到优先经验回放缓冲区
            maddpg.store_transition(obs, actions, rewards, next_obs, done)

            # 更新状态
            obs = next_obs
            episode_reward += sum(rewards.values() if isinstance(rewards, dict) else rewards)

            # 定期输出状态信息
            if (step + 1) % 10 == 0 or done:
                logger.info(f"\nStep {step + 1}/{config['max_steps']}")
                logger.info(f"红方单位数: {info['red_alive']}")
                logger.info(f"蓝方单位数: {info['blue_alive']}")
                logger.info(f"当前累计奖励: {episode_reward:.2f}")

            # 强化终止条件判断
            if done or all_units_dead:
                logger.info(f"\n推演在第 {step + 1} 步结束")
                logger.info(f"终止原因: {'所有单位阵亡' if all_units_dead else '推演完成'}")
                break

        return {
            'steps': step_count,
            'total_reward': episode_reward,
            'simulation_data': simulation_data,
            'final_state': {
                'red_alive': info['red_alive'],
                'blue_alive': info['blue_alive']
            }
        }

    except Exception as e:
        logger.error(f"推演过程出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise
def save_simulation_results(results, logger):
    """保存推演结果"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "simulation_results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # 保存详细结果
        result_file = os.path.join(results_dir, f"sim_results_{timestamp}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("=== 推演结果摘要 ===\n")
            f.write(f"总步数: {results['steps']}\n")
            f.write(f"总奖励: {results['total_reward']:.2f}\n")
            f.write(f"最终红方单位数: {results['final_state']['red_alive']}\n")
            f.write(f"最终蓝方单位数: {results['final_state']['blue_alive']}\n\n")
            
            f.write("=== 详细推演数据 ===\n")
            for step_data in results['simulation_data']:
                f.write(f"\nStep {step_data['step']}:\n")
                f.write(f"  红方单位数: {step_data['red_alive']}\n")
                f.write(f"  蓝方单位数: {step_data['blue_alive']}\n")
                f.write(f"  动作: {step_data['actions']}\n")
                f.write(f"  奖励: {step_data['rewards']}\n")
        
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
        config = setup_environment()
        logger.info("环境配置完成")
        
        # 2. 创建环境
        env = MoziAdapter(
            num_red=config['num_red'],
            num_blue=config['num_blue'],
            max_steps=config['max_steps'],
            env_config=EnvConfig
        )
        logger.info("墨子环境创建成功")
        
        # 3. 初始化改进版MADDPG
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        maddpg = MADDPG(
            n_red=config['num_red'],
            n_blue=config['num_blue'],
            obs_dim=obs_dim,
            act_dim=act_dim,
            n_step=config['n_step'],
            gamma=config['gamma'],
            tau=config['tau'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        )
        logger.info(f"改进版MADDPG初始化完成 (obs_dim: {obs_dim}, act_dim: {act_dim})")
        
        # 4. 加载预训练模型
        model_path = "G:\work\danzi\mozi\mozi_ai_sdk\testxiangding\results\mozi_training\20241220_023536\checkpoint_episode_final.pt"
        if load_model(maddpg, model_path):
            logger.info("已加载预训练模型")
        else:
            logger.warning("未找到预训练模型，将使用初始化网络")
        
        # 5. 运行推演
        logger.info("\n开始运行推演...")
        simulation_results = run_simulation(env, maddpg, config, logger)
        
        # 6. 保存结果
        save_simulation_results(simulation_results, logger)
        
        logger.info("\n=== 推演仿真完成 ===")
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