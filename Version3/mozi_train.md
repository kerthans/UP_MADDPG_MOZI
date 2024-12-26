# 墨子仿真平台MADDPG训练与推演系统技术文档

## 1. 系统概述

本系统实现了基于MADDPG(Multi-Agent Deep Deterministic Policy Gradient)算法的空战智能体在墨子仿真平台上的训练与推演。系统包含基础版本和增强版本的实现，以及完整的训练过程监控和推演结果分析功能。

### 1.1 核心组件

系统由以下主要模块构成：

- 训练系统
  - MoziBaselineTrainer：基础版训练器
  - MoziEnhancedTrainer：增强版训练器
  - 训练监控与可视化模块
  
- 推演系统
  - baseline_mozi.py：基础版推演执行
  - up_mozi.py：增强版推演执行
  - 推演记录与分析模块

### 1.2 主要特性

- 完整的训练流程管理
- 丰富的训练指标监控
- 训练过程可视化
- 模型保存与加载
- 详细的推演记录
- 推演结果分析

## 2. 系统架构

### 2.1 整体架构

```
训练系统                     推演系统
   ↓                           ↓
MADDPG算法层   ←→    模型加载/保存接口
   ↓                           ↓
墨子环境适配层(MoziAdapter)
   ↓
墨子仿真平台(MoziServer)
```

### 2.2 训练系统架构

```python
class MoziEnhancedTrainer:
    def __init__(self, config):
        self.env = MoziAdapter(...)
        self.maddpg = MADDPG(...)
        self.metrics = {...}
        
    def train(self):
        # 训练主循环
        
    def _train_episode(self):
        # 单回合训练
```

### 2.3 推演系统架构

```python
def run_simulation(env, maddpg, config, logger):
    # 推演主循环
    for step in range(config['max_steps']):
        actions = maddpg.select_actions(obs)
        next_obs, rewards, done, info = env.step(actions)
        # 记录推演数据
```

## 3. 训练系统实现

### 3.1 训练配置管理

基础配置参数：
```python
config = {
    'env_config': {
        'num_red': 2,
        'num_blue': 3,
        'max_steps': 200,
        'field_size': 100000.0,
        'attack_range': 25000.0,
    },
    'model_config': {
        'n_step': 3,
        'gamma': 0.99,
        'hidden_dim': 256,
    },
    'training': {
        'n_episodes': 5000,
        'initial_noise': 0.3,
        'noise_decay': 0.9995,
    }
}
```

### 3.2 训练流程管理

1. 回合训练流程：
```python
def _train_episode(self):
    obs = self.env.reset()
    
    while True:
        # 1. 动作选择
        actions = self.maddpg.select_actions(obs, self.noise)
        
        # 2. 环境交互
        next_obs, rewards, done, info = self.env.step(actions)
        
        # 3. 经验存储
        self.maddpg.store_transition(...)
        
        # 4. 网络更新
        if self.maddpg.can_train():
            self.maddpg.train()
```

2. 性能评估：
```python
def _evaluate(self, num_episodes=10):
    eval_stats = []
    for episode in range(num_episodes):
        episode_reward = 0
        obs = self.env.reset()
        
        while True:
            actions = self.maddpg.select_actions(obs, 0)  # 无噪声
            next_obs, rewards, done, info = self.env.step(actions)
            episode_reward += sum(rewards.values())
            if done:
                break
                
        eval_stats.append(episode_reward)
```

### 3.3 训练监控系统

1. 指标跟踪：
```python
self.metrics = {
    'episode_rewards': deque(maxlen=10000),
    'red_win_rates': deque(maxlen=10000),
    'hit_rates': deque(maxlen=10000),
    'casualties': deque(maxlen=10000),
}
```

2. 可视化实现：
```python
def _plot_metrics(self):
    plt.figure(figsize=(20, 15))
    # 奖励曲线
    plt.subplot(3, 2, 1)
    self._plot_smoothed_curve(
        self.metrics['episode_rewards'],
        'Total Reward',
        window=50
    )
    # 胜率曲线
    plt.subplot(3, 2, 2)
    self._plot_smoothed_curve(
        self.metrics['red_win_rates'],
        'Red Win Rate'
    )
```

## 4. 推演系统实现

### 4.1 推演配置

```python
def setup_environment():
    config = {
        'num_red': 2,
        'num_blue': 3,
        'max_steps': 200,
        'noise_scale': 0.1,
    }
    return config
```

### 4.2 推演执行流程

```python
def run_simulation(env, maddpg, config, logger):
    obs = env.reset()
    simulation_data = []
    
    for step in range(config['max_steps']):
        # 1. 动作选择
        actions = maddpg.select_actions(obs)
        
        # 2. 执行动作
        next_obs, rewards, done, info = env.step(actions)
        
        # 3. 记录数据
        step_data = {
            'step': step + 1,
            'red_alive': info['red_alive'],
            'blue_alive': info['blue_alive'],
            'actions': actions,
            'rewards': rewards
        }
        simulation_data.append(step_data)
        
        # 4. 状态更新
        obs = next_obs
        
        if done:
            break
```

### 4.3 结果记录与分析

```python
def save_simulation_results(results, logger):
    # 保存推演摘要
    with open(result_file, 'w') as f:
        f.write("=== 推演结果摘要 ===\n")
        f.write(f"总步数: {results['steps']}\n")
        f.write(f"总奖励: {results['total_reward']:.2f}\n")
        
        # 保存详细推演数据
        f.write("=== 详细推演数据 ===\n")
        for step_data in results['simulation_data']:
            f.write(f"\nStep {step_data['step']}:\n")
            f.write(f"  红方单位数: {step_data['red_alive']}\n")
            f.write(f"  蓝方单位数: {step_data['blue_alive']}\n")
```

## 5. 核心优化与改进

### 5.1 训练系统优化

1. 经验回放优化：
```python
# 优先经验回放
class PrioritizedReplay:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        
    def add(self, error, sample):
        priority = (error + 1e-5) ** self.alpha
        self.tree.add(priority, sample)
```

2. 网络架构优化：
```python
class EnhancedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
```

### 5.2 推演系统优化

1. 强化终止条件：
```python
def check_termination(info):
    all_units_dead = (info['red_alive'] == 0 and 
                     info['blue_alive'] == 0)
    return all_units_dead
```

2. 日志记录优化：
```python
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simulation.log'),
            logging.StreamHandler()
        ]
    )
```

## 6. 使用指南

### 6.1 训练系统使用

```python
# 1. 创建训练器
trainer = MoziEnhancedTrainer(config)

# 2. 开始训练
trainer.train()

# 3. 评估模型
eval_results = trainer.evaluate(num_episodes=5)
```

### 6.2 推演系统使用

```python
# 1. 环境设置
config = setup_environment()

# 2. 创建环境
env = MoziAdapter(
    num_red=config['num_red'],
    num_blue=config['num_blue'],
    max_steps=config['max_steps']
)

# 3. 运行推演
simulation_results = run_simulation(env, maddpg, config, logger)
```

## 7. 性能优化建议

1. 训练效率优化：
- 使用批量经验采样
- 实现多进程训练
- 优化网络更新频率

2. 推演效率优化：
- 优化动作执行逻辑
- 改进终止条件判断
- 实现增量数据记录
