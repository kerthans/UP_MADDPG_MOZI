# 墨子仿真平台与MADDPG算法集成实现文档

## 1. 系统概述

本文档详细说明了将MADDPG（Multi-Agent Deep Deterministic Policy Gradient）算法与墨子仿真平台进行集成的实现方案。该系统实现了强化学习算法在复杂空战仿真环境中的应用。

### 1.1 核心组件

系统由以下主要组件构成：

- MoziAdapter: MADDPG与墨子平台的适配层
- MoziEnv: 墨子仿真环境封装
- EnvConfig: 环境配置管理
- 状态和动作空间转换模块

### 1.2 关键特性

- 支持多智能体并行交互
- 连续动作空间控制
- 完整的态势感知和信息获取
- 灵活的任务配置和场景管理
- 实时监控和数据记录

## 2. 系统架构

### 2.1 整体架构

```
MADDPG算法层
    ↓ ↑
适配层(MoziAdapter)
    ↓ ↑ 
墨子环境层(MoziEnv)
    ↓ ↑
墨子仿真平台
```

### 2.2 关键接口

1. 观察空间接口：
```python
def get_observations(self):
    """
    获取态势观察信息
    返回: {
        'red_<guid>': {
            'latitude': float,
            'longitude': float,
            'altitude': float,
            'speed': float,
            'heading': float,
            'fuel': float
        },
        'blue_<guid>': {...}
    }
    """
```

2. 动作执行接口：
```python
def execute_action(self, action_dict):
    """
    执行动作控制
    参数: {
        'red_0': [turn_rate, speed_rate, attack],
        'blue_0': [turn_rate, speed_rate, attack]
    }
    """
```

## 3. 核心实现细节

### 3.1 状态空间设计

状态空间包含以下关键信息：

1. 基础状态向量（7维）：
```python
base_vector = [
    normalized_latitude,    # 归一化纬度
    normalized_longitude,   # 归一化经度
    normalized_speed,      # 归一化速度
    heading_cos,           # 航向余弦分量
    heading_sin,           # 航向正弦分量
    normalized_altitude,   # 归一化高度
    normalized_fuel        # 归一化燃料
]
```

2. 扩展状态信息（49维）：
- 基础状态（7维）
- 相对态势信息（42维）
  - 友方单位相对信息
  - 敌方单位相对信息

### 3.2 动作空间设计

动作空间为3维连续空间：
```python
action_space = spaces.Box(
    low=np.array([-1, -1, 0]),   # 最小值
    high=np.array([1, 1, 1]),    # 最大值
    dtype=np.float32
)
```

动作维度定义：
1. 转向控制：[-1, 1] → [-30°, 30°]
2. 速度控制：[-1, 1] → [150节, 400节]
3. 攻击决策：[0, 1] → [不攻击, 攻击]

### 3.3 状态转换实现

```python
def _convert_mozi_obs_to_maddpg(self, mozi_obs: Dict) -> Dict:
    """状态转换核心实现"""
    maddpg_obs = {}
    base_dim = 7
    total_dim = 49
    
    # 处理每个单位的状态
    for guid, idx in self.red_id_map.items():
        unit_id = f'red_{guid}'
        if unit_id in mozi_obs:
            # 基础状态向量构建
            base_vector = self._create_base_vector(mozi_obs[unit_id])
            
            # 扩展状态向量
            state_vector = self._extend_state_vector(
                base_vector,
                unit_id,
                mozi_obs
            )
            
            maddpg_obs[f'red_{idx}'] = state_vector
```

### 3.4 动作执行实现

动作执行流程：

1. 动作解析：
```python
def execute_action(self, action_dict):
    for unit_id, action in action_dict.items():
        # 解析单位ID和动作值
        side_prefix, unit_idx = unit_id.split('_')
        guid = self._get_unit_guid(side_prefix, unit_idx)
        
        # 执行具体控制
        self._execute_unit_action(guid, action)
```

2. 航向控制：
```python
# 航向控制实现
heading_change = float(action[0]) * 30
new_heading = (current_heading + heading_change) % 360
cmd = f"ScenEdit_SetUnit({{guid='{guid}', heading={new_heading}}})"
```

3. 速度控制：
```python
# 速度控制实现
speed_factor = (action[1] + 1) * 0.5
new_speed = 150 + speed_factor * 250  # [150,400]节
new_speed_kmh = new_speed * 1.852     # 转换为千米/小时
```

4. 攻击控制：
```python
# 攻击控制实现
if action[2] > 0.5:
    for contact in enemy_side.contacts.values():
        if contact.m_ContactType == 2:  # 空中目标
            cmd = f"ScenEdit_AssignUnitToTarget('{guid}', '{contact.strGuid}')"
```

## 4. 环境配置管理

### 4.1 配置参数

```python
class EnvConfig:
    # 服务器配置
    SERVER_IP = "127.0.0.1"
    SERVER_PORT = "6060"
    
    # 推演参数
    SIMULATE_COMPRESSION = 3
    DURATION_INTERVAL = 15
    
    # 单位参数
    MIN_SPEED = 150.0  # 节
    MAX_SPEED = 400.0  # 节
    ATTACK_RANGE = 25000.0  # 米
```

### 4.2 场景配置

```python
scenario_config = {
    'num_red': 2,
    'num_blue': 3,
    'max_steps': 200,
    'field_size': 100000.0,  # 100km
    'attack_range': 25000.0, # 25km
}
```

## 5. 性能优化

### 5.1 状态缓存优化

```python
# 使用ID映射加速查找
self.red_id_map = {}      # GUID到索引的映射
self.blue_id_map = {}
self.red_id_reverse_map = {}  # 索引到GUID的映射
self.blue_id_reverse_map = {}
```

### 5.2 动作执行优化

1. 批量命令处理：
```python
# 合并相似命令减少通信
commands = []
for unit_id, action in action_dict.items():
    cmd = self._prepare_command(unit_id, action)
    commands.append(cmd)
self.mozi_server.send_and_recv(";".join(commands))
```

2. 异常处理和重试机制：
```python
def execute_with_retry(self, cmd, max_retries=3):
    for i in range(max_retries):
        try:
            return self.mozi_server.send_and_recv(cmd)
        except Exception as e:
            if i == max_retries - 1:
                raise
            time.sleep(0.1)
```

## 6. 使用指南

### 6.1 环境初始化

```python
# 创建适配器环境
env = MoziAdapter(
    num_red=2,
    num_blue=3,
    max_steps=200,
    env_config=EnvConfig()
)

# 重置环境
initial_obs = env.reset()
```

### 6.2 训练循环

```python
# 训练循环示例
while True:
    # 获取MADDPG动作
    actions = maddpg.get_actions(obs)
    
    # 执行动作
    next_obs, rewards, done, info = env.step(actions)
    
    # 存储经验
    maddpg.store_experience(obs, actions, rewards, next_obs, done)
    
    if done:
        break
```

## 7. 调试和监控

### 7.1 日志记录

关键节点日志记录：
```python
print(f"\nStep {self.current_step} completed:")
print(f"Action success: {success}")
print(f"Units remaining - Red: {info['red_alive']}, Blue: {info['blue_alive']}")
```

### 7.2 状态监控

实时状态监控：
```python
def _print_state_info(self, obs):
    for unit_id, state in obs.items():
        print(f"Unit {unit_id}:")
        print(f"Position: ({state['latitude']:.2f}, {state['longitude']:.2f})")
        print(f"Speed: {state['speed']:.2f}")
        print(f"Heading: {state['heading']:.2f}")
```
