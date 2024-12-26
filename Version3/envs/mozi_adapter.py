# combat_sim/mozi_adapter.py
import time


from envs.mozi_env import MoziEnv
from envs.state_handler import StateHandler
from envs.combat_mechanics import CombatMechanics
import numpy as np
import gym
from gym import spaces
from typing import Dict, List, Tuple
import math


class MoziAdapter(gym.Env):
    """将墨子环境适配为与原有CombatEnv兼容的格式"""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_red=2, num_blue=3, max_steps=30,
                 env_config=None, state_handler=None, combat_mechanics=None):
        super().__init__()

        # 基础配置
        self.config = {
            'num_red': num_red,
            'num_blue': num_blue,
            'max_steps': max_steps,
            'field_size': 100000.0,  # 100km转换为墨子的米单位
            'attack_range': 25000.0,  # 25km攻击范围
            'min_speed': 150.0,  # 最小速度150节
            'max_speed': 400.0,  # 最大速度400节
            'max_turn_rate': math.pi / 6,
            'hit_probability': 0.8,
            'num_threads': 8
        }

        # 从env_config更新配置
        if env_config:
            self.config.update({
                'attack_range': getattr(env_config, 'ATTACK_RANGE', 25000.0),
                'min_speed': getattr(env_config, 'MIN_SPEED', 150.0),
                'max_speed': getattr(env_config, 'MAX_SPEED', 400.0),
                'max_steps': getattr(env_config, 'MAX_STEPS', 30),
                'simulate_compression': getattr(env_config, 'SIMULATE_COMPRESSION', 3),
                'duration_interval': getattr(env_config, 'DURATION_INTERVAL', 15),
            })

        # 初始化ID映射
        self.red_id_map = {}  # GUID 到索引的映射
        self.blue_id_map = {}
        self.red_id_reverse_map = {}  # 索引到 GUID 的映射
        self.blue_id_reverse_map = {}

        # 初始化墨子环境
        self.mozi_env = MoziEnv(
            IP=getattr(env_config, 'SERVER_IP', "127.0.0.1"),
            AIPort=getattr(env_config, 'SERVER_PORT', "6060"),
            platform=getattr(env_config, 'PLATFORM', "windows"),
            scenario_name=getattr(env_config, 'SCENARIO_NAME', "maddpg.scen"),
            simulate_compression=self.config['simulate_compression'],
            duration_interval=self.config['duration_interval'],
            synchronous=getattr(env_config, 'SYNCHRONOUS', True)
        )

        # 初始化状态处理器和战斗机制
        self.state_handler = state_handler or StateHandler(self.config)
        self.combat_mechanics = combat_mechanics or CombatMechanics(self.config)

        # 动作空间: [转向率, 速度, 攻击]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # 初始化观察空间
        obs_dim = self.state_handler.calculate_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.current_step = 0
        self.episode_rewards = []


    def _update_id_maps(self, obs: Dict):
        """更新单位ID映射"""
        try:
            # 重置映射
            self.red_id_map = {}
            self.blue_id_map = {}
            self.red_id_reverse_map = {}
            self.blue_id_reverse_map = {}

            # 收集所有红方单位
            red_units = [(k, v) for k, v in obs.items() if k.startswith('red_')]
            for idx, (guid, _) in enumerate(sorted(red_units)):
                if idx >= self.config['num_red']:
                    break
                clean_guid = guid.replace('red_', '')
                self.red_id_map[clean_guid] = idx
                self.red_id_reverse_map[idx] = clean_guid
                print(f"Red mapping: GUID {clean_guid} <-> Index {idx}")

            # 收集所有蓝方单位
            blue_units = [(k, v) for k, v in obs.items() if k.startswith('blue_')]
            for idx, (guid, _) in enumerate(sorted(blue_units)):
                if idx >= self.config['num_blue']:
                    break
                clean_guid = guid.replace('blue_', '')
                self.blue_id_map[clean_guid] = idx
                self.blue_id_reverse_map[idx] = clean_guid
                print(f"Blue mapping: GUID {clean_guid} <-> Index {idx}")

        except Exception as e:
            print(f"Error in _update_id_maps: {str(e)}")
            import traceback
            traceback.print_exc()
    def reset(self):
        """重置环境"""
        # 重置墨子环境
        self.mozi_env.reset()
        self.current_step = 0

        # 获取初始观察
        raw_obs = self.mozi_env.get_observations()

        # 更新ID映射
        self._update_id_maps(raw_obs)

        # 转换为MADDPG格式
        maddpg_obs = self._convert_mozi_obs_to_maddpg(raw_obs)

        return maddpg_obs

    # def step(self, actions: Dict):
    #     """执行一步交互"""
    #     self.current_step += 1
    #
    #     # 转换动作格式
    #     mozi_actions = {}
    #     for unit_id, action in actions.items():
    #         try:
    #             # 获取GUID
    #             side_prefix, idx = unit_id.split('_')
    #             idx = int(idx)
    #             guid = self.red_id_reverse_map.get(idx) if side_prefix == 'red' else self.blue_id_reverse_map.get(idx)
    #
    #             if guid:
    #                 # 将[-1,1]的动作值转换为实际控制值
    #                 turn_rate = float(action[0])  # 转向比例
    #                 speed_ratio = float(action[1])  # 速度比例
    #                 attack = float(action[2]) > 0.5  # 攻击决策
    #
    #                 # 构造墨子API所需的动作格式
    #                 current_obs = self.mozi_env.get_observations()
    #                 if side_prefix + '_' + guid in current_obs:
    #                     current_heading = current_obs[side_prefix + '_' + guid]['heading']
    #
    #                     # 计算新航向
    #                     heading_change = turn_rate * 30  # 最大转向角为30度
    #                     new_heading = (current_heading + heading_change) % 360
    #
    #                     # 计算新速度
    #                     base_speed = self.config['min_speed']
    #                     speed_range = self.config['max_speed'] - self.config['min_speed']
    #                     new_speed = base_speed + (speed_ratio + 1) * 0.5 * speed_range
    #
    #                     # 创建标准格式的动作指令
    #                     mozi_actions[side_prefix + '_' + guid] = {
    #                         'heading': new_heading,
    #                         'speed': new_speed,
    #                         'attack': attack
    #                     }
    #                 else:
    #                     print(f"Warning: Unit {unit_id} not found in current observations")
    #
    #         except Exception as e:
    #             print(f"Error processing action for unit {unit_id}: {str(e)}")
    #             continue
    #
    #     # 执行动作
    #     try:
    #         if mozi_actions:
    #             self.mozi_env.execute_action(mozi_actions)
    #     except Exception as e:
    #         print(f"Error executing actions: {str(e)}")
    #
    #     # 获取新的观察
    #     raw_obs = self.mozi_env.get_observations()
    #     self._update_id_maps(raw_obs)  # 更新ID映射
    #     next_obs = self._convert_mozi_obs_to_maddpg(raw_obs)
    #
    #     # 计算奖励和完成状态
    #     rewards = self._compute_rewards(raw_obs)
    #     done = self._check_done(raw_obs)
    #
    #     # 构建信息字典
    #     info = {
    #         'red_alive': sum(1 for k in raw_obs if 'red' in k),
    #         'blue_alive': sum(1 for k in raw_obs if 'blue' in k),
    #         'current_step': self.current_step
    #     }
    #
    #     return next_obs, rewards, done, info
    def step(self, actions: Dict):
        """执行一步仿真
        Args:
            actions: {'red_0': [turn_rate, speed_rate, attack]}
        Returns:
            next_obs: 下一步观察
            rewards: 奖励
            done: 是否结束
            info: 额外信息
        """
        try:
            # 1. 执行动作
            success = self.execute_action(actions)
            if not success:
                print("Warning: Action execution failed")

            # 2. 推进墨子推演
            for _ in range(3):  # 多尝试几次确保推进
                if self.mozi_env.step():
                    break
                time.sleep(0.1)

            # 3. 获取新的观察
            raw_obs = self.mozi_env.get_observations()
            self._update_id_maps(raw_obs)
            next_obs = self._convert_mozi_obs_to_maddpg(raw_obs)

            # 4. 计算奖励和结束状态
            rewards = self._compute_rewards(raw_obs)
            done = self._check_done(raw_obs)

            # 5. 构建信息字典
            info = {
                'red_alive': sum(1 for k in raw_obs if 'red' in k),
                'blue_alive': sum(1 for k in raw_obs if 'blue' in k),
                'current_step': self.current_step
            }

            self.current_step += 1
            print(f"\nStep {self.current_step} completed:")
            print(f"Action success: {success}")
            print(f"Units remaining - Red: {info['red_alive']}, Blue: {info['blue_alive']}")

            return next_obs, rewards, done, info

        except Exception as e:
            print(f"Error in step: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}, {}, True, {}

    def execute_action(self, action_dict):
        """执行动作"""
        try:
            print("\nExecuting actions:")
            for unit_id, action in action_dict.items():
                try:
                    # 解析单位ID
                    if '_' not in unit_id:
                        print(f"Invalid unit ID format: {unit_id}")
                        continue

                    side_prefix, unit_idx = unit_id.split('_')
                    unit_idx = int(unit_idx)

                    # 获取GUID
                    if side_prefix == 'red':
                        if unit_idx not in self.red_id_reverse_map:
                            print(f"Red unit index {unit_idx} not found in reverse map")
                            continue
                        guid = self.red_id_reverse_map[unit_idx]
                    else:
                        if unit_idx not in self.blue_id_reverse_map:
                            print(f"Blue unit index {unit_idx} not found in reverse map")
                            continue
                        guid = self.blue_id_reverse_map[unit_idx]

                    print(f"\nProcessing unit {unit_id} (GUID: {guid}):")
                    print(f"Action: {action}")

                    # 1. 创建或更新任务
                    mission_name = f"mission_{guid}_{int(time.time())}"
                    cmd = f"ScenEdit_AddMission('{guid}','{mission_name}','Patrol',{{type='AAW'}})"
                    self.mozi_env.mozi_server.send_and_recv(cmd)

                    # 2. 设置任务参数
                    cmd = f"ScenEdit_SetMission('{guid}', '{mission_name}', {{flightSize=1}})"
                    self.mozi_env.mozi_server.send_and_recv(cmd)

                    # 3. 执行具体动作
                    # 航向控制
                    if abs(action[0]) > 0.001:
                        heading_change = float(action[0]) * 30
                        current_obs = self.mozi_env.get_observations()
                        current_heading = current_obs[f"{side_prefix}_{guid}"]['heading']
                        new_heading = (current_heading + heading_change) % 360
                        cmd = f"ScenEdit_SetUnit({{guid='{guid}', heading={new_heading}}})"
                        print(f"Heading command: {cmd}")
                        self.mozi_env.mozi_server.send_and_recv(cmd)

                    # 速度控制
                    if abs(action[1]) > 0.001:
                        speed_factor = (action[1] + 1) * 0.5
                        new_speed = 150 + speed_factor * 250  # [150,400]节
                        new_speed_kmh = new_speed * 1.852
                        cmd = f"ScenEdit_SetUnit({{guid='{guid}', speed={new_speed_kmh}}})"
                        print(f"Speed command: {cmd}")
                        self.mozi_env.mozi_server.send_and_recv(cmd)

                    # 攻击控制
                    if action[2] > 0.5:
                        enemy_side = self.mozi_env.blueside if side_prefix == 'red' else self.mozi_env.redside
                        if enemy_side and enemy_side.contacts:
                            for contact in enemy_side.contacts.values():
                                if contact.m_ContactType == 2:
                                    cmd = f"ScenEdit_AssignUnitToTarget('{guid}', '{contact.strGuid}')"
                                    print(f"Attack command: {cmd}")
                                    self.mozi_env.mozi_server.send_and_recv(cmd)
                                    break

                    print(f"Action execution completed for unit {unit_id}")

                except Exception as e:
                    print(f"Error processing action for unit {unit_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

            return True

        except Exception as e:
            print(f"Error in execute_action: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    # def _convert_mozi_obs_to_maddpg(self, mozi_obs: Dict) -> Dict:
    #     """将墨子观察转换为MADDPG格式"""
    #     maddpg_obs = {}
    #
    #     # 基础状态维度
    #     base_dim = 7  # 基础状态向量维度
    #
    #     # 处理红方单位
    #     for guid, idx in self.red_id_map.items():
    #         if guid in mozi_obs:
    #             obs = mozi_obs[guid]
    #             # 标准化状态向量
    #             state_vector = np.zeros(base_dim)
    #             try:
    #                 state_vector[0] = obs['latitude'] / 90.0 if abs(obs['latitude']) <= 90.0 else np.sign(
    #                     obs['latitude'])
    #                 state_vector[1] = obs['longitude'] / 180.0 if abs(obs['longitude']) <= 180.0 else np.sign(
    #                     obs['longitude'])
    #                 state_vector[2] = obs['altitude'] / 10000.0 if obs['altitude'] is not None else 0.0
    #                 state_vector[3] = obs['speed'] / self.config['max_speed'] if obs['speed'] is not None else 0.0
    #                 heading_rad = np.radians(obs['heading']) if obs['heading'] is not None else 0.0
    #                 state_vector[4] = np.cos(heading_rad)
    #                 state_vector[5] = np.sin(heading_rad)
    #                 state_vector[6] = obs['fuel'] / 100.0 if obs['fuel'] is not None else 1.0
    #             except (KeyError, TypeError) as e:
    #                 print(f"警告: 处理红方单位 {guid} 状态时出错: {str(e)}")
    #
    #             maddpg_obs[f'red_{idx}'] = self.state_handler.enhance_observation(
    #                 state_vector,
    #                 f'red_{idx}',
    #                 True,
    #                 mozi_obs
    #             )
    #
    #     # 处理蓝方单位
    #     for guid, idx in self.blue_id_map.items():
    #         if guid in mozi_obs:
    #             obs = mozi_obs[guid]
    #             state_vector = np.zeros(base_dim)
    #             try:
    #                 state_vector[0] = obs['latitude'] / 90.0 if abs(obs['latitude']) <= 90.0 else np.sign(
    #                     obs['latitude'])
    #                 state_vector[1] = obs['longitude'] / 180.0 if abs(obs['longitude']) <= 180.0 else np.sign(
    #                     obs['longitude'])
    #                 state_vector[2] = obs['altitude'] / 10000.0 if obs['altitude'] is not None else 0.0
    #                 state_vector[3] = obs['speed'] / self.config['max_speed'] if obs['speed'] is not None else 0.0
    #                 heading_rad = np.radians(obs['heading']) if obs['heading'] is not None else 0.0
    #                 state_vector[4] = np.cos(heading_rad)
    #                 state_vector[5] = np.sin(heading_rad)
    #                 state_vector[6] = obs['fuel'] / 100.0 if obs['fuel'] is not None else 1.0
    #             except (KeyError, TypeError) as e:
    #                 print(f"警告: 处理蓝方单位 {guid} 状态时出错: {str(e)}")
    #
    #             maddpg_obs[f'blue_{idx}'] = self.state_handler.enhance_observation(
    #                 state_vector,
    #                 f'blue_{idx}',
    #                 False,
    #                 mozi_obs
    #             )
    #
    #     return maddpg_obs
    def _convert_mozi_obs_to_maddpg(self, mozi_obs: Dict) -> Dict:
        """将墨子观察转换为MADDPG格式"""
        maddpg_obs = {}
        try:
            print("\nConverting observations:")
            # 扩展状态维度以匹配网络输入 (7 * 7 = 49)
            base_dim = 7  # 基础维度
            total_dim = 49  # 网络输入维度

            # 处理红方单位
            for guid, idx in self.red_id_map.items():
                unit_id = f'red_{guid}'
                if unit_id in mozi_obs:
                    obs = mozi_obs[unit_id]
                    # 创建基础状态向量
                    base_vector = np.zeros(base_dim, dtype=np.float32)
                    try:
                        # 标准化基础状态
                        base_vector[0] = obs['latitude'] / 90.0  # 纬度归一化
                        base_vector[1] = obs['longitude'] / 180.0  # 经度归一化
                        base_vector[2] = obs['speed'] / self.config['max_speed']  # 速度归一化
                        heading_rad = np.radians(obs['heading'])
                        base_vector[3] = np.cos(heading_rad)  # 航向cos分量
                        base_vector[4] = np.sin(heading_rad)  # 航向sin分量
                        base_vector[5] = obs['altitude'] / 10000.0  # 高度归一化
                        base_vector[6] = obs.get('fuel', 100.0) / 100.0  # 燃料归一化

                        # 扩展状态向量到所需维度
                        state_vector = np.zeros(total_dim, dtype=np.float32)
                        # 复制基础状态到扩展向量的前7个位置
                        state_vector[:base_dim] = base_vector

                        # 添加与其他单位的相对信息
                        offset = base_dim
                        for other_guid, other_idx in self.red_id_map.items():
                            if other_guid != guid and f'red_{other_guid}' in mozi_obs:
                                other_obs = mozi_obs[f'red_{other_guid}']
                                # 计算相对位置和状态
                                rel_lat = (other_obs['latitude'] - obs['latitude']) / 90.0
                                rel_lon = (other_obs['longitude'] - obs['longitude']) / 180.0
                                rel_speed = (other_obs['speed'] - obs['speed']) / self.config['max_speed']
                                state_vector[offset:offset + 3] = [rel_lat, rel_lon, rel_speed]
                                offset += 3

                        for other_guid, other_idx in self.blue_id_map.items():
                            if f'blue_{other_guid}' in mozi_obs:
                                other_obs = mozi_obs[f'blue_{other_guid}']
                                # 计算相对位置和状态
                                rel_lat = (other_obs['latitude'] - obs['latitude']) / 90.0
                                rel_lon = (other_obs['longitude'] - obs['longitude']) / 180.0
                                rel_speed = (other_obs['speed'] - obs['speed']) / self.config['max_speed']
                                state_vector[offset:offset + 3] = [rel_lat, rel_lon, rel_speed]
                                offset += 3

                        maddpg_obs[f'red_{idx}'] = state_vector
                        print(f"Converted red unit {idx} (GUID: {guid})")
                        print(f"State vector shape: {state_vector.shape}")

                    except Exception as e:
                        print(f"Error converting red unit {guid}: {str(e)}")
                        continue

            # 处理蓝方单位，类似的处理方式
            for guid, idx in self.blue_id_map.items():
                unit_id = f'blue_{guid}'
                if unit_id in mozi_obs:
                    obs = mozi_obs[unit_id]
                    base_vector = np.zeros(base_dim, dtype=np.float32)
                    try:
                        # 标准化基础状态
                        base_vector[0] = obs['latitude'] / 90.0
                        base_vector[1] = obs['longitude'] / 180.0
                        base_vector[2] = obs['speed'] / self.config['max_speed']
                        heading_rad = np.radians(obs['heading'])
                        base_vector[3] = np.cos(heading_rad)
                        base_vector[4] = np.sin(heading_rad)
                        base_vector[5] = obs['altitude'] / 10000.0
                        base_vector[6] = obs.get('fuel', 100.0) / 100.0

                        # 扩展状态向量
                        state_vector = np.zeros(total_dim, dtype=np.float32)
                        state_vector[:base_dim] = base_vector

                        # 添加与其他单位的相对信息
                        offset = base_dim
                        for other_guid, other_idx in self.blue_id_map.items():
                            if other_guid != guid and f'blue_{other_guid}' in mozi_obs:
                                other_obs = mozi_obs[f'blue_{other_guid}']
                                rel_lat = (other_obs['latitude'] - obs['latitude']) / 90.0
                                rel_lon = (other_obs['longitude'] - obs['longitude']) / 180.0
                                rel_speed = (other_obs['speed'] - obs['speed']) / self.config['max_speed']
                                state_vector[offset:offset + 3] = [rel_lat, rel_lon, rel_speed]
                                offset += 3

                        for other_guid, other_idx in self.red_id_map.items():
                            if f'red_{other_guid}' in mozi_obs:
                                other_obs = mozi_obs[f'red_{other_guid}']
                                rel_lat = (other_obs['latitude'] - obs['latitude']) / 90.0
                                rel_lon = (other_obs['longitude'] - obs['longitude']) / 180.0
                                rel_speed = (other_obs['speed'] - obs['speed']) / self.config['max_speed']
                                state_vector[offset:offset + 3] = [rel_lat, rel_lon, rel_speed]
                                offset += 3

                        maddpg_obs[f'blue_{idx}'] = state_vector
                        print(f"Converted blue unit {idx} (GUID: {guid})")
                        print(f"State vector shape: {state_vector.shape}")

                    except Exception as e:
                        print(f"Error converting blue unit {guid}: {str(e)}")
                        continue

            print(f"Converted observations for {len(maddpg_obs)} units")
            return maddpg_obs

        except Exception as e:
            print(f"Error in observation conversion: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def _get_red_states(self, obs: Dict) -> np.ndarray:
        """提取红方状态矩阵"""
        states = np.zeros((self.config['num_red'], 5))

        for guid, idx in self.red_id_map.items():
            if guid in obs:
                unit_obs = obs[guid]
                states[idx] = [
                    unit_obs['latitude'],
                    unit_obs['longitude'],
                    np.radians(unit_obs['heading']),
                    unit_obs['speed'],
                    1.0  # 存活状态
                ]

        return states

    def _get_blue_states(self, obs: Dict) -> np.ndarray:
        """提取蓝方状态矩阵"""
        states = np.zeros((self.config['num_blue'], 5))

        for guid, idx in self.blue_id_map.items():
            if guid in obs:
                unit_obs = obs[guid]
                states[idx] = [
                    unit_obs['latitude'],
                    unit_obs['longitude'],
                    np.radians(unit_obs['heading']),
                    unit_obs['speed'],
                    1.0  # 存活状态
                ]

        return states

    def _check_done(self, obs: Dict) -> bool:
        """检查是否结束"""
        if self.current_step >= self.config['max_steps']:
            return True

        if not obs:
            print("警告: 未获取到任何单位观察")
            return False

        red_alive = any('red' in k for k in obs.keys())
        blue_alive = any('blue' in k for k in obs.keys())

        if not (red_alive and blue_alive):
            print(f"战斗结束 - 红方存活: {red_alive}, 蓝方存活: {blue_alive}")

        return not (red_alive and blue_alive)

    def _compute_rewards(self, obs: Dict) -> Dict:
        """计算奖励"""
        if not obs:
            return {f'red_{i}': 0.0 for i in range(self.config['num_red'])}

        rewards = {}
        red_states = self._get_red_states(obs)
        blue_states = self._get_blue_states(obs)

        base_rewards = self.combat_mechanics.compute_rewards(
            red_states,
            blue_states,
            [],
            []
        ) if len(red_states) > 0 and len(blue_states) > 0 else {}

        # 计算红方奖励
        for idx in range(self.config['num_red']):
            unit_id = f'red_{idx}'
            guid = self.red_id_reverse_map.get(idx)

            if guid and guid in obs:
                reward = base_rewards.get(unit_id, 0.0)
                reward += 0.1  # 存活奖励
                speed_ratio = obs[guid]['speed'] / self.config['max_speed']
                reward += 0.05 * speed_ratio
            else:
                reward = 0.0

            rewards[unit_id] = reward

        # 计算蓝方奖励
        for idx in range(self.config['num_blue']):
            unit_id = f'blue_{idx}'
            guid = self.blue_id_reverse_map.get(idx)

            if guid and guid in obs:
                reward = base_rewards.get(unit_id, 0.0)
                reward += 0.1
                speed_ratio = obs[guid]['speed'] / self.config['max_speed']
                reward += 0.05 * speed_ratio
            else:
                reward = 0.0

            rewards[unit_id] = reward

        return rewards

    def close(self):
        """关闭环境"""
        if hasattr(self, 'mozi_env'):
            self.mozi_env.close()

    @property
    def field_size(self):
        return self.config['field_size']

    @property
    def max_steps(self):
        return self.config['max_steps']

    @property
    def reward_weights(self):
        return self.combat_mechanics.reward_weights