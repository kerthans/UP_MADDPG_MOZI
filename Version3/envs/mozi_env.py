# mozi_env.py

from mozi_simu_sdk.mozi_server import MoziServer
from mozi_ai_sdk.base_env import BaseEnvironment
import numpy as np
import time

class MoziEnv(BaseEnvironment):
    """
    墨子环境类
    """
    def __init__(self, IP, AIPort, platform, scenario_name, simulate_compression, 
                duration_interval, synchronous, app_mode=1):
        """
        构造函数
        参数：
            IP (str): 服务器IP
            AIPort (str): 端口号
            platform (str): 平台 windows/linux
            scenario_name (str): 想定文件名
            simulate_compression (int): 推演压缩比
            duration_interval (int): 决策步长(秒)
            synchronous (bool): 是否同步
            app_mode (int): 应用模式，默认为1(windows本地模式)
        """
        super().__init__(IP, AIPort, platform, scenario_name, simulate_compression,
                        duration_interval, synchronous)
        self.app_mode = app_mode
        self.scenario = None
        self.redside = None
        self.blueside = None
        print("MoziEnv initialized with:")
        print(f"IP: {IP}")
        print(f"Port: {AIPort}")
        print(f"Platform: {platform}")
        print(f"Scenario: {scenario_name}")
        print(f"App Mode: {app_mode}")

    def step(self):
        """单步决策"""
        try:
            self.situation = self.mozi_server.update_situation(self.scenario, self.app_mode)
            # 更新推演方
            if self.redside:
                self.redside.static_update()
            if self.blueside:
                self.blueside.static_update()
            # 运行仿真
            self.mozi_server.run_grpc_simulate()
            return self.scenario
        except Exception as e:
            print(f"Error in step: {e}")
            raise
    def reset(self):
        """重置环境"""
        try:
            # 创建并连接服务器
            if not self.mozi_server:
                print("Creating MoziServer...")
                self.mozi_server = MoziServer(
                    self.server_ip,
                    self.aiPort,
                    self.platform,
                    self.scenario_name,
                    self.simulate_compression,
                    self.synchronous
                )
                time.sleep(4.0)
            # 设置运行模式和决策步长
            print("Setting simulation parameters...")
            self.mozi_server.set_run_mode(self.synchronous)
            self.mozi_server.set_decision_step_length(self.duration_interval)
            # 初始化推演想定
            print("Loading scenario...")
            self.mozi_server.send_and_recv("IsMasterControl")
            self.scenario = self.mozi_server.load_scenario()
            assert self.scenario is not None, "Failed to load scenario"
            # 设置推演速度
            self.mozi_server.set_simulate_compression(self.simulate_compression)
            # 初始化态势
            print("Initializing situation...")
            self.mozi_server.init_situation(self.scenario, self.app_mode)
            # 获取推演方
            print("Getting sides...")
            self.redside = self.scenario.get_side_by_name("红方")
            self.redside.static_construct()
            self.blueside = self.scenario.get_side_by_name("蓝方")
            self.blueside.static_construct()
            # 启动推演
            print("Starting simulation...")
            self.mozi_server.run_simulate()
            # 更新态势
            print("Updating situation...")
            self.scenario = self.step()
            print("Reset completed successfully")
            return self.scenario
        except Exception as e:
            print(f"Error in reset: {e}")
            raise
    def get_observations(self):
        """获取状态观察信息"""
        try:
            obs = {}
            # 获取红方单位状态，不限制无人机
            if self.redside:
                for k, v in self.redside.aircrafts.items():
                    obs[f'red_{k}'] = {
                        'latitude': v.dLatitude,
                        'longitude': v.dLongitude,
                        'altitude': v.fCurrentAltitude_ASL,
                        'speed': v.fCurrentSpeed,
                        'heading': v.fCurrentHeading,
                        'fuel': v.iCurrentFuel if hasattr(v, 'iCurrentFuel') else 100
                    }
            # 获取蓝方单位状态，不限制无人机
            if self.blueside:
                for k, v in self.blueside.aircrafts.items():
                    obs[f'blue_{k}'] = {
                        'latitude': v.dLatitude,
                        'longitude': v.dLongitude,
                        'altitude': v.fCurrentAltitude_ASL,
                        'speed': v.fCurrentSpeed,
                        'heading': v.fCurrentHeading,
                        'fuel': v.iCurrentFuel if hasattr(v, 'iCurrentFuel') else 100
                    }
            # 打印调试信息
            print(f"观察到的单位: {list(obs.keys())}")
            return obs
        except Exception as e:
            print(f"Error getting observations: {e}")
            raise
    def execute_action(self, action_dict):
        """执行动作
        支持两种格式的动作输入:
        1. {'red_0': [turn_rate, speed_rate, attack]} - 索引格式
        2. {'red_guid': [turn_rate, speed_rate, attack]} - GUID格式
        """
        try:
            for unit_id, action in action_dict.items():
                try:
                    # 解析单位ID
                    if '_' not in unit_id:
                        print(f"Invalid unit ID format: {unit_id}")
                        continue

                    side_prefix, unit_id_part = unit_id.split('_')

                    # 获取实际的GUID
                    unit_guid = unit_id_part  # 默认假设是GUID格式

                    # 如果是数字索引格式，则需要转换
                    if unit_id_part.isdigit():
                        if side_prefix == 'red' and int(unit_id_part) in self.red_id_reverse_map:
                            unit_guid = self.red_id_reverse_map[int(unit_id_part)]
                        elif side_prefix == 'blue' and int(unit_id_part) in self.blue_id_reverse_map:
                            unit_guid = self.blue_id_reverse_map[int(unit_id_part)]
                        else:
                            print(f"Invalid unit index: {unit_id}")
                            continue

                    # 验证单位是否存在
                    current_obs = self.get_observations()
                    full_unit_id = f"{side_prefix}_{unit_guid}"
                    if full_unit_id not in current_obs:
                        print(f"Unit {full_unit_id} not found in current observations")
                        continue

                    # 执行动作控制
                    try:
                        action = np.array(action, dtype=np.float32)

                        # 1. 航向控制
                        if abs(action[0]) > 0.001:
                            heading_change = float(action[0]) * 30  # [-1,1] -> [-30,30]度
                            current_heading = current_obs[full_unit_id]['heading']
                            new_heading = (current_heading + heading_change) % 360
                            cmd = f"ScenEdit_SetUnit({{guid='{unit_guid}', heading={new_heading}}})"
                            self.mozi_server.send_and_recv(cmd)

                        # 2. 速度控制
                        if abs(action[1]) > 0.001:
                            speed_factor = (action[1] + 1) * 0.5  # [-1,1] -> [0,1]
                            new_speed = 150 + speed_factor * 250  # [150,400]节
                            # 转换节到千米/小时
                            new_speed_kmh = new_speed * 1.852  # 1节 = 1.852千米/小时
                            cmd = f"ScenEdit_SetUnit({{guid='{unit_guid}', speed={new_speed_kmh}}})"
                            self.mozi_server.send_and_recv(cmd)

                        # 3. 攻击控制
                        if action[2] > 0.5:
                            enemy_side = self.blueside if side_prefix == 'red' else self.redside
                            if enemy_side and enemy_side.contacts:
                                for contact in enemy_side.contacts.values():
                                    if contact.m_ContactType == 2:  # 空中目标
                                        strike_name = f"strike_{unit_guid}_{int(time.time())}"
                                        cmd = f"ScenEdit_AssignUnitToTarget('{unit_guid}', '{contact.strGuid}')"
                                        self.mozi_server.send_and_recv(cmd)
                                        break

                    except Exception as e:
                        print(f"Error executing action for unit {full_unit_id}: {str(e)}")
                        continue

                except Exception as e:
                    print(f"Error processing action for unit {unit_id}: {str(e)}")
                    continue

            # 推进仿真一步
            self.scenario = self.step()
            return True

        except Exception as e:
            print(f"Error in execute_action: {str(e)}")
            return False

    def _find_targets(self, aircraft):

        """寻找目标"""

        try:

            if '红方' in aircraft.strName:
                side = self.blueside
            else:
                side = self.redside
            if not side:

                return []
            return [contact for k, contact in side.contacts.items() 

                    if contact.m_ContactType == 2]  # 2表示敌方飞机
        except Exception as e:

            print(f"Error finding targets: {e}")

            return []