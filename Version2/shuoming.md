# 改进型 MADDPG 算法技术文档

## 1. 概述

本文档详细介绍了一个基于 MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 的改进算法实现，该实现针对多智能体空战环境进行了特定优化，包含了多项创新性改进。

### 1.1 核心特点

- 支持异构智能体
- 引入内在激励机制
- 优化的注意力机制
- 优先级经验回放
- N步回报计算
- 改进的探索策略

## 2. 环境设计

### 2.1 空战环境（CombatEnv）

空战环境实现了一个多智能体对抗场景，主要特点包括：

- 支持红蓝双方智能体
- 可配置的智能体数量
- 异构智能体支持
- 连续动作空间
- 复杂的奖励系统

### 2.2 状态空间设计

每个智能体的状态由9维向量表示：
- 位置 (2维)
- 速度 (2维)
- 相对速度 (2维)
- 加速度 (2维)
- 存活状态 (1维)

### 2.3 智能体类型

环境支持三种类型的智能体，各具特色：

1. 侦察机 (Scout)
   - 最大速度：6.0
   - 攻击范围：3.0
   - 击杀概率：0.6
   - 观察范围：8.0

2. 战斗机 (Fighter)
   - 最大速度：5.0
   - 攻击范围：4.0
   - 击杀概率：0.8
   - 观察范围：6.0

3. 轰炸机 (Bomber)
   - 最大速度：4.0
   - 攻击范围：5.0
   - 击杀概率：0.9
   - 观察范围：5.0

### 2.4 奖励系统设计

实现了多层次的奖励机制：

```python
reward_weights = {
    'kill': 10.0,           # 击杀奖励
    'death': -8.0,          # 死亡惩罚
    'victory': 20.0,        # 胜利奖励
    'survive': 0.05,        # 生存奖励
    'approach': 0.1,        # 接近奖励
    'team': 0.15,           # 团队协作奖励
    'boundary': -0.3,       # 边界惩罚
    'energy': -0.01,        # 能量消耗惩罚
    'strategy': 0.2         # 战术位置奖励
}
```

## 3. 算法改进

### 3.1 网络架构

#### 3.1.1 Actor网络改进

Actor网络包含三个主要组件：
- 特征提取网络
- 多头注意力层
- 策略网络

关键改进：
```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, agent_type=None):
        # 特征提取
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 多头注意力层
        self.attention = MultiHeadAttention(hidden_dim)
        
        # 策略网络
        self.policy_net = nn.Sequential(...)
```

#### 3.1.2 Critic网络改进

Critic网络的主要改进：
- 独立的状态和动作编码器
- 改进的特征融合
- 多头注意力机制的应用

### 3.2 内在激励机制

实现了基于ICM (Intrinsic Curiosity Module) 的内在激励机制：

1. 前向动力学模型
   - 预测下一个状态
   - 计算预测误差作为奖励

2. 反向动力学模型
   - 从状态转换预测动作
   - 提供额外的学习信号

### 3.3 优先级经验回放

实现了基于TD-error的优先级经验回放：

```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.priorities = np.zeros(capacity)
        self.max_priority = 1.0
```

主要改进：
- 动态优先级更新
- 基于重要性采样的偏差修正
- 高效的样本存储和检索

### 3.4 探索策略

采用改进的OU噪声过程进行探索：

```python
def _ou_noise(self, x, scale, mu=0., theta=0.15, sigma=0.2):
    return theta * (mu - x) + sigma * np.random.randn(*x.shape) * scale
```

特点：
- 自适应噪声衰减
- 智能体类型特定的探索参数
- 改进的时间相关性

## 4. 训练优化

### 4.1 梯度处理

- 采用LayerNorm进行梯度稳定
- 正交初始化改善训练初期表现
- 梯度裁剪防止梯度爆炸

### 4.2 多步学习

实现了n步回报计算以改善价值估计：

```python
def _compute_n_step_returns(self, rewards, next_q_values, dones, gamma):
    returns = rewards.clone()
    future_return = next_q_values
    for i in range(1, self.n_step + 1):
        future_return = gamma * future_return * (1 - dones)
        returns = returns + (gamma ** i) * future_return
```

### 4.3 注意力机制

改进的多头注意力实现：
- 独立的QKV投影
- 改进的维度缩放
- LayerNorm稳定化

## 5. 实施建议

### 5.1 参数配置

推荐的基础参数设置：
- 学习率：1e-3
- 折扣因子：0.95
- Tau（软更新系数）：0.01
- 批量大小：64
- 隐藏层维度：256
- N步回报：3

### 5.2 训练技巧

1. 逐步增加环境复杂度
2. 使用课程学习方式
3. 定期保存检查点
4. 监控关键指标：
   - 平均奖励
   - 胜率
   - TD误差
   - 策略熵

## 6. 总结与展望

### 6.1 主要优势

1. 异构智能体支持提高了算法的适用性
2. 内在激励机制改善了探索效率
3. 优先级经验回放提高了样本利用效率
4. 注意力机制增强了智能体间的协作


### 6.3 应用建议

1. 根据具体场景调整奖励权重
2. 针对不同智能体类型优化网络结构
3. 适当调整探索策略参数
4. 重视经验回放机制的调优