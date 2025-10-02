"""防御者智能体演示脚本

这个脚本演示如何使用防御者智能体，包括：
1. 训练防御者智能体
2. 评估防御效果
3. 与攻击者对抗
4. 可视化结果
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加路径以导入nasim模块
sys.path.append(str(Path(__file__).parent.parent))

import nasim
from nasim.agents.defender_agent import create_defender_agent, DefenderDQNAgent
from nasim.agents.dqn_agent import DQNAgent
from nasim.envs.adversarial_environment import create_adversarial_env


def train_defender_agent(scenario_name='tiny', training_steps=10000, seed=42):
    """训练防御者智能体"""
    print(f"开始训练防御者智能体 - 场景: {scenario_name}")
    
    # 创建防御者智能体和环境
    agent, env = create_defender_agent(
        scenario_name,
        seed=seed,
        training_steps=training_steps,
        hidden_sizes=[128, 128],
        lr=0.001,
        batch_size=32,
        exploration_steps=training_steps // 2,
        verbose=True
    )
    
    # 训练
    agent.train()
    
    # 保存模型
    model_path = f"defender_model_{scenario_name}_{seed}.pt"
    agent.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    return agent, env


def evaluate_defender_agent(agent, env, num_episodes=10):
    """评估防御者智能体"""
    print(f"\n评估防御者智能体 - {num_episodes} 个回合")
    
    results = []
    
    for episode in range(num_episodes):
        ep_return, ep_cost, ep_steps, evaluation = agent.run_eval_episode(
            env, render=(episode == 0)  # 只渲染第一个回合
        )
        
        results.append({
            'episode': episode + 1,
            'return': ep_return,
            'cost': ep_cost,
            'steps': ep_steps,
            'total_score': evaluation['total_score'],
            'compromised_hosts': evaluation['compromised_hosts'],
            'compromised_high_value': evaluation['compromised_high_value']
        })
        
        print(f"回合 {episode + 1}: 奖励={ep_return:.2f}, 成本={ep_cost:.2f}, "
              f"被攻破主机={evaluation['compromised_hosts']}")
    
    # 计算平均结果
    avg_return = np.mean([r['return'] for r in results])
    avg_cost = np.mean([r['cost'] for r in results])
    avg_score = np.mean([r['total_score'] for r in results])
    avg_compromised = np.mean([r['compromised_hosts'] for r in results])
    
    print(f"\n平均结果:")
    print(f"  平均奖励: {avg_return:.2f}")
    print(f"  平均成本: {avg_cost:.2f}")
    print(f"  平均总分: {avg_score:.2f}")
    print(f"  平均被攻破主机: {avg_compromised:.1f}")
    
    return results


def train_attacker_agent(scenario_name='tiny', training_steps=5000, seed=42):
    """训练攻击者智能体（用于对抗测试）"""
    print(f"\n训练攻击者智能体 - 场景: {scenario_name}")
    
    # 创建攻击者环境
    env = nasim.make_benchmark(
        scenario_name,
        seed=seed,
        fully_obs=False,
        flat_actions=True,
        flat_obs=True
    )
    
    # 创建攻击者智能体
    agent = DQNAgent(
        env,
        seed=seed,
        training_steps=training_steps,
        hidden_sizes=[64, 64],
        lr=0.001,
        batch_size=32,
        exploration_steps=training_steps // 2,
        verbose=True
    )
    
    # 训练
    agent.train()
    
    # 保存模型
    model_path = f"attacker_model_{scenario_name}_{seed}.pt"
    agent.save(model_path)
    print(f"攻击者模型已保存到: {model_path}")
    
    return agent, env


def run_adversarial_test(scenario_name='tiny', num_episodes=5, seed=42):
    """运行对抗测试"""
    print(f"\n运行对抗测试 - 场景: {scenario_name}")
    
    # 创建对抗环境
    adv_env = create_adversarial_env(scenario_name, mode='alternating', seed=seed)
    
    # 训练智能体（简化版本用于演示）
    print("训练防御者智能体...")
    defender_agent, _ = create_defender_agent(
        scenario_name,
        seed=seed,
        training_steps=2000,
        verbose=False
    )
    defender_agent.train()
    
    print("训练攻击者智能体...")
    attacker_agent, _ = train_attacker_agent(
        scenario_name,
        training_steps=2000,
        seed=seed
    )
    
    # 设置智能体
    adv_env.set_agents(attacker_agent, defender_agent)
    
    # 运行对抗回合
    results = []
    for episode in range(num_episodes):
        print(f"\n对抗回合 {episode + 1}")
        evaluation = adv_env.run_episode(render=(episode == 0))
        results.append(evaluation)
        
        print(f"攻击成功: {evaluation['attack_success']}")
        print(f"攻击者总奖励: {evaluation['attacker_total_reward']:.2f}")
        print(f"防御者总奖励: {evaluation['defender_total_reward']:.2f}")
    
    # 统计结果
    attack_success_rate = np.mean([r['attack_success'] for r in results])
    avg_attacker_reward = np.mean([r['attacker_total_reward'] for r in results])
    avg_defender_reward = np.mean([r['defender_total_reward'] for r in results])
    
    print(f"\n对抗测试结果:")
    print(f"  攻击成功率: {attack_success_rate:.2%}")
    print(f"  攻击者平均奖励: {avg_attacker_reward:.2f}")
    print(f"  防御者平均奖励: {avg_defender_reward:.2f}")
    
    return results


def visualize_results(results, title="防御者智能体性能"):
    """可视化结果"""
    if not results:
        print("没有结果可以可视化")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)
    
    episodes = [r['episode'] for r in results]
    returns = [r['return'] for r in results]
    costs = [r['cost'] for r in results]
    scores = [r['total_score'] for r in results]
    compromised = [r['compromised_hosts'] for r in results]
    
    # 奖励趋势
    axes[0, 0].plot(episodes, returns, 'b-o')
    axes[0, 0].set_title('回合奖励')
    axes[0, 0].set_xlabel('回合')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].grid(True)
    
    # 成本趋势
    axes[0, 1].plot(episodes, costs, 'r-o')
    axes[0, 1].set_title('回合成本')
    axes[0, 1].set_xlabel('回合')
    axes[0, 1].set_ylabel('成本')
    axes[0, 1].grid(True)
    
    # 总分趋势
    axes[1, 0].plot(episodes, scores, 'g-o')
    axes[1, 0].set_title('总分')
    axes[1, 0].set_xlabel('回合')
    axes[1, 0].set_ylabel('总分')
    axes[1, 0].grid(True)
    
    # 被攻破主机数
    axes[1, 1].bar(episodes, compromised, alpha=0.7)
    axes[1, 1].set_title('被攻破主机数')
    axes[1, 1].set_xlabel('回合')
    axes[1, 1].set_ylabel('主机数')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('defender_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("性能图表已保存为 'defender_performance.png'")


def demonstrate_defense_actions(scenario_name='tiny'):
    """演示不同防御动作的效果"""
    print(f"\n演示防御动作效果 - 场景: {scenario_name}")
    
    # 创建防御者环境
    _, env = create_defender_agent(scenario_name, training_steps=1, verbose=False)
    
    print(f"可用防御动作数量: {env.action_space.n}")
    
    # 演示每种类型的动作
    action_types = {}
    for i, action in enumerate(env.action_space.actions):
        action_type = action.__class__.__name__
        if action_type not in action_types:
            action_types[action_type] = []
        action_types[action_type].append((i, action))
    
    print(f"\n动作类型统计:")
    for action_type, actions in action_types.items():
        print(f"  {action_type}: {len(actions)} 个")
    
    # 演示每种动作类型的第一个动作
    env.reset()
    print(f"\n演示动作效果:")
    
    for action_type, actions in action_types.items():
        if actions:
            action_idx, action = actions[0]
            print(f"\n执行 {action_type}:")
            print(f"  动作成本: {action.cost}")
            
            obs, reward, done, info = env.step(action_idx)
            print(f"  执行结果: {info['action_result'].get('success', False)}")
            print(f"  获得奖励: {reward:.2f}")


def main():
    """主函数"""
    print("防御者智能体演示程序")
    print("=" * 50)
    
    scenario_name = 'tiny'  # 使用tiny场景进行演示
    seed = 42
    
    try:
        # 1. 演示防御动作
        demonstrate_defense_actions(scenario_name)
        
        # 2. 训练防御者智能体
        agent, env = train_defender_agent(
            scenario_name, 
            training_steps=5000,  # 减少训练步数用于演示
            seed=seed
        )
        
        # 3. 评估防御者智能体
        results = evaluate_defender_agent(agent, env, num_episodes=5)
        
        # 4. 可视化结果
        try:
            visualize_results(results)
        except ImportError:
            print("matplotlib未安装，跳过可视化")
        
        # 5. 运行对抗测试
        adversarial_results = run_adversarial_test(
            scenario_name, 
            num_episodes=3,
            seed=seed
        )
        
        print("\n演示完成！")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
