# UAV-Combat-MADDPG

## 项目描述
这是一个基于多智能体强化学习的多无人机对抗仿真系统，模拟了红蓝双方无人机的空战场景。系统以 MADDPG 算法为核心，结合动态博弈模型，实现了智能体的协同与对抗。

### 主要功能
- 支持多种对抗场景，包括不同数量和能力的无人机对抗。
- 仿真无人机的运动学模型，支持灵活的动作和策略。
- 可配置的攻击范围、命中概率等对抗参数。
- 灵活的奖励机制设计，支持不同策略的优化。
- 使用墨子平台进行仿真推演和结果可视化。

## 系统要求
- Python 3.8+
- PyTorch 1.8+
- NumPy
- PyYAML
- 墨子平台（可选）

---


### 项目架构解析

#### 1. **主目录**
- **`README.md`**：存储项目简介、使用说明与开发文档。
- **`requirements.txt`**：列出项目依赖的第三方库，方便安装。
- **`main.py`**：项目的主入口，负责解析命令行参数、初始化配置、调用训练或评估流程。

#### 2. **配置文件目录 (`config/`)**
- **`config.yaml`**：用于存储环境和仿真参数（如无人机数量、战场大小、步长等）。
- **`hyperparameters.yaml`**：独立管理模型超参数（如学习率、目标网络更新频率等），便于快速调整。

#### 3. **源码目录 (`src/`)**
- **`environment/`**  
  - `uav_env.py`：实现空战仿真环境，包括无人机运动模型、交互逻辑、状态更新等。
  - `space.py`：定义状态和动作空间，确保与智能体接口一致。

- **`agents/`**  
  - `base_agent.py`：智能体的抽象类，封装通用行为和接口。
  - `maddpg_agent.py`：基于 MADDPG 的智能体具体实现。

- **`models/`**  
  - `actor.py` 和 `critic.py`：定义 Actor-Critic 网络结构，支持连续动作空间。

- **`memory/`**  
  - `replay_buffer.py`：经验回放缓冲区，支持高效存储与采样。

- **`utils/`**  
  - `helpers.py`：存储通用工具函数，如日志处理、参数加载等。
  - `visualization.py`：负责绘制训练曲线、策略图等。

- **`training/`**  
  - `trainer.py`：封装训练逻辑（如采样、梯度更新、网络同步等）。
  - `evaluator.py`：封装模型评估逻辑（如策略测试、性能分析等）。

#### 4. **测试目录 (`tests/`)**
- 针对关键模块编写单元测试，用于验证代码的正确性和鲁棒性。


#### 5. **结果目录 (`results/`)**
- **`checkpoints/`**：存储训练过程中保存的模型检查点。
- **`metrics/`**：存储训练与评估的指标文件。
- **`figures/`**：存储可视化图表文件，便于分析结果。

#### 6. **日志目录 (`logs/`)**
- 存储训练和评估的日志文件，便于调试和问题排查。

---
  
## 项目结构预览：
```
UAV-Combat-MADDPG/
├── README.md
├── requirements.txt
├── main.py
├── config/
│   ├── config.yaml               # 主配置文件
│   └── hyperparameters.yaml      # 超参数配置
├── src/
│   ├── __init__.py
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── uav_env.py            # 仿真环境主文件
│   │   └── space.py              # 定义动作和状态空间
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── maddpg_agent.py       # MADDPG 智能体
│   │   └── base_agent.py         # 基础智能体抽象类
│   ├── models/
│   │   ├── __init__.py
│   │   ├── actor.py              # Actor 网络
│   │   └── critic.py             # Critic 网络
│   ├── memory/
│   │   ├── __init__.py
│   │   └── replay_buffer.py      # 经验回放缓冲区
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── helpers.py            # 辅助工具函数
│   │   └── visualization.py      # 数据可视化工具
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # 训练逻辑模块
│   │   └── evaluator.py          # 模型评估模块
├── tests/
│   ├── __init__.py
│   ├── test_agents.py            # 智能体模块单元测试
│   ├── test_environment.py       # 环境模块单元测试
│   └── test_models.py            # 神经网络模块单元测试
├── logs/
│   ├── training.log              # 训练日志
│   └── evaluation.log            # 评估日志
├── results/
│   ├── checkpoints/              # 模型检查点存储
│   │   └── checkpoint_001.pth
│   ├── metrics/                  # 训练与评估指标
│   │   └── episode_rewards.csv
│   └── figures/                  # 训练过程的可视化图表
│       └── reward_curve.png

```
---

## 快速开始
1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 运行训练
```bash
python main.py --mode train
```

3. 运行评估
```bash
python main.py --mode eval
```

## 配置说明
在 config/config.yaml 中可以配置以下参数：
- 红蓝双方无人机数量
- 环境参数（战场大小、时间步长等）
- 训练参数（学习率、批次大小等）
- 奖励函数参数
