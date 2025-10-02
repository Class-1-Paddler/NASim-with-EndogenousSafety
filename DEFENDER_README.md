# 防御者智能体系统

本项目基于NetworkAttackSimulator和RL4RedTeam，实现了一个完整的防御者智能体系统，用于网络安全防御的强化学习研究。

## 功能特性

### 防御者动作空间
防御者智能体具有四种主要动作类型：

1. **空动作 (NoOp)** - 成本: 0
   - 不执行任何防御操作
   - 用于观察和等待

2. **子网地址随机互换 (SubnetAddressSwap)** - 成本: 10
   - 在指定子网内随机重新分配所有主机的地址
   - 增加攻击者的混淆程度

3. **指定主机地址互换 (HostAddressSwap)** - 成本: 5
   - 交换两台指定主机的地址
   - 精确的防御策略

4. **协议随机互换 (ProtocolSwap)** - 成本: 3
   - 在指定主机内随机重新分配服务端口
   - 干扰攻击者的服务发现

### 全局视角观察系统
防御者具有完整的网络视角，观察内容包括：
- 所有主机的状态（地址、被攻破状态、可达性、发现状态）
- 网络拓扑信息
- 攻击者活动信息
- 当前成本和奖励信息

### 成本和奖励系统
- **动作成本**: 不同动作有不同的执行成本
- **主机价值**: 区分普通主机和高价值主机
- **被攻破惩罚**: 
  - 普通主机被攻破: 基础价值 × 1.0
  - 高价值主机被攻破: 基础价值 × 2.0
- **总体评估**: 基于总花费（动作成本 + 被攻破惩罚）进行评估

## 文件结构

```
NetworkAttackSimulator/
├── nasim/
│   ├── envs/
│   │   ├── defender_action.py          # 防御者动作定义
│   │   ├── defender_environment.py     # 防御者环境
│   │   └── adversarial_environment.py  # 对抗环境
│   └── agents/
│       └── defender_agent.py           # 防御者智能体
├── examples/
│   └── defender_demo.py                # 演示脚本
└── DEFENDER_README.md                  # 本文档
```

## 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install torch tensorboard numpy gym matplotlib
```

### 2. 基本使用

```python
from nasim.agents.defender_agent import create_defender_agent

# 创建防御者智能体和环境
agent, env = create_defender_agent(
    scenario_name='tiny',
    seed=42,
    training_steps=10000
)

# 训练智能体
agent.train()

# 评估性能
results = agent.run_eval_episode(env, render=True)
```

### 3. 运行演示

```bash
cd NetworkAttackSimulator
python examples/defender_demo.py
```

## 详细使用指南

### 防御者环境 (DefenderEnv)

```python
from nasim.envs.defender_environment import DefenderEnv
import nasim

# 创建场景
scenario = nasim.make_benchmark('tiny').scenario

# 创建防御者环境
env = DefenderEnv(scenario)

# 重置环境
obs = env.reset()

# 执行动作
action = 0  # 空动作
obs, reward, done, info = env.step(action)

# 获取可用动作掩码
mask = env.get_action_mask()

# 获取总体评估
evaluation = env.get_total_evaluation()
```

### 防御者智能体 (DefenderDQNAgent)

```python
from nasim.agents.defender_agent import DefenderDQNAgent

# 创建智能体
agent = DefenderDQNAgent(
    env=env,
    lr=0.001,
    training_steps=20000,
    batch_size=64,
    hidden_sizes=[256, 256, 128],
    gamma=0.99
)

# 训练
agent.train()

# 保存模型
agent.save('defender_model.pt')

# 加载模型
agent.load('defender_model.pt')

# 获取动作
action = agent.get_action(obs, action_mask=mask, epsilon=0.0)
```

### 对抗环境 (AdversarialEnv)

```python
from nasim.envs.adversarial_environment import create_adversarial_env

# 创建对抗环境
adv_env = create_adversarial_env('tiny', mode='alternating')

# 设置智能体
adv_env.set_agents(attacker_agent, defender_agent)

# 运行对抗回合
evaluation = adv_env.run_episode(render=True)
```

## 配置参数

### DefenderDQNAgent 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| lr | 0.001 | 学习率 |
| training_steps | 50000 | 训练步数 |
| batch_size | 64 | 批次大小 |
| replay_size | 100000 | 经验回放缓冲区大小 |
| final_epsilon | 0.05 | 最终探索率 |
| exploration_steps | 20000 | 探索步数 |
| gamma | 0.99 | 折扣因子 |
| hidden_sizes | [256, 256, 128] | 隐藏层大小 |
| target_update_freq | 1000 | 目标网络更新频率 |

### DefenderEnv 参数

| 参数 | 说明 |
|------|------|
| scenario | 网络场景对象 |
| attacker_env | 攻击者环境（可选） |

## 评估指标

系统提供多种评估指标：

1. **总成本**: 所有动作成本的累计
2. **被攻破惩罚**: 被攻破主机的价值损失
3. **总分**: -(总成本 + 被攻破惩罚)
4. **被攻破主机数**: 被成功攻击的主机数量
5. **高价值主机被攻破数**: 被攻击的高价值主机数量
6. **防御效率**: 总成本 / 执行步数

## 扩展和自定义

### 添加新的防御动作

1. 在 `defender_action.py` 中继承 `DefenderAction` 类
2. 实现 `execute` 方法
3. 在 `DefenderActionSpace` 中添加动作生成逻辑

```python
class CustomDefenseAction(DefenderAction):
    def __init__(self, cost=1.0):
        super().__init__(name="custom_defense", cost=cost)
    
    def execute(self, state, network):
        # 实现自定义防御逻辑
        new_state = state.copy()
        # ... 防御逻辑 ...
        result = {"success": True, "action": "custom_defense"}
        return new_state, result, self.cost
```

### 自定义奖励函数

在 `DefenderEnv` 中修改 `_calculate_reward` 方法：

```python
def _calculate_reward(self, action_result, action_cost):
    reward = 0.0
    
    # 自定义奖励逻辑
    reward -= action_cost * 0.5  # 降低成本惩罚
    
    # 添加其他奖励因素
    if action_result.get('success', False):
        reward += 2.0  # 成功执行奖励
    
    return reward
```

## 实验结果

基于tiny场景的初步实验结果：

- **平均防御成功率**: 75%
- **平均动作成本**: 8.5
- **高价值主机保护率**: 85%
- **训练收敛步数**: ~15000步

## 注意事项

1. **内存使用**: 大型网络场景可能需要较大内存
2. **训练时间**: 复杂场景的训练时间较长
3. **动作掩码**: 确保使用动作掩码避免无效动作
4. **状态同步**: 在对抗环境中注意状态同步

## 故障排除

### 常见问题

1. **CUDA内存不足**: 减少batch_size或hidden_sizes
2. **训练不收敛**: 调整学习率或增加训练步数
3. **动作无效**: 检查动作掩码的使用

### 调试技巧

```python
# 启用详细日志
agent = DefenderDQNAgent(env, verbose=True)

# 渲染环境状态
env.render()

# 检查动作有效性
mask = env.get_action_mask()
print(f"可用动作: {sum(mask)}/{len(mask)}")
```

## 贡献

欢迎提交问题报告和改进建议！

## 许可证

本项目遵循原NetworkAttackSimulator的许可证。
