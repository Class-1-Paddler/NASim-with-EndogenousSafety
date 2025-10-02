#!/usr/bin/env python3
"""简化的防御者智能体演示脚本"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

def demo_defender_actions():
    """演示防御者动作"""
    print("演示防御者动作...")
    
    try:
        from nasim.envs.defender_action import (
            DefenderNoOp, SubnetAddressSwap, HostAddressSwap, 
            ProtocolSwap, DefenderActionSpace
        )
        
        # 创建简单的场景对象模拟
        class MockScenario:
            def __init__(self):
                self.subnets = [0, 3, 2]  # internet, subnet1(3 hosts), subnet2(2 hosts)
                self.address_space = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        
        scenario = MockScenario()
        
        # 创建动作空间
        action_space = DefenderActionSpace(scenario)
        print(f"创建了 {action_space.n} 个防御动作")
        
        # 展示不同类型的动作
        action_types = {}
        for i, action in enumerate(action_space.actions):
            action_type = action.__class__.__name__
            if action_type not in action_types:
                action_types[action_type] = []
            action_types[action_type].append(action)
        
        print("\n动作类型统计:")
        for action_type, actions in action_types.items():
            print(f"  {action_type}: {len(actions)} 个动作")
            if actions:
                print(f"    示例: 成本={actions[0].cost}")
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_defender_environment():
    """演示防御者环境"""
    print("\n演示防御者环境...")
    
    try:
        import nasim
        from nasim.envs.defender_environment import DefenderEnv
        
        # 创建场景
        env = nasim.make_benchmark('tiny', seed=42, fully_obs=True, flat_actions=True, flat_obs=True)
        scenario = env.scenario
        
        # 创建防御者环境
        defender_env = DefenderEnv(scenario)
        print(f"防御者环境创建成功")
        print(f"  观察空间维度: {defender_env.observation_space.shape}")
        print(f"  动作空间大小: {defender_env.action_space.n}")
        
        # 重置环境
        obs = defender_env.reset()
        print(f"环境重置成功，观察维度: {obs.shape}")
        
        # 执行几个动作
        for step in range(3):
            # 获取可用动作
            action_mask = defender_env.get_action_mask()
            available_actions = np.where(action_mask)[0]
            
            if len(available_actions) > 0:
                action = np.random.choice(available_actions)
                obs, reward, done, info = defender_env.step(action)
                
                action_obj = defender_env.action_space.get_action(action)
                print(f"  步骤 {step+1}: {action_obj.__class__.__name__}, 奖励={reward:.2f}, 成本={info.get('action_cost', 0):.2f}")
                
                if done:
                    break
            else:
                print(f"  步骤 {step+1}: 没有可用动作")
                break
        
        # 获取最终评估
        evaluation = defender_env.get_total_evaluation()
        print(f"\n最终评估:")
        for key, value in evaluation.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_simple_training():
    """演示简单训练"""
    print("\n演示简单训练...")
    
    try:
        from nasim.agents.defender_agent import create_defender_agent
        
        # 创建防御者智能体（使用很少的训练步数）
        agent, env = create_defender_agent(
            'tiny',
            seed=42,
            training_steps=50,  # 很少的训练步数用于演示
            hidden_sizes=[32, 32],
            batch_size=8,
            verbose=False
        )
        print("防御者智能体创建成功")
        
        # 简短训练
        print("开始训练...")
        agent.train()
        print("训练完成")
        
        # 评估
        ep_return, ep_cost, ep_steps, evaluation = agent.run_eval_episode(env, render=False)
        print(f"评估结果:")
        print(f"  回合奖励: {ep_return:.2f}")
        print(f"  回合成本: {ep_cost:.2f}")
        print(f"  总分: {evaluation['total_score']:.2f}")
        print(f"  被攻破主机: {evaluation['compromised_hosts']}")
        
        return True
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("防御者智能体演示程序")
    print("=" * 50)
    
    demos = [
        demo_defender_actions,
        demo_defender_environment,
        demo_simple_training,
    ]
    
    passed = 0
    total = len(demos)
    
    for demo in demos:
        if demo():
            passed += 1
        print()  # 空行分隔
    
    print("=" * 50)
    print(f"演示结果: {passed}/{total} 成功")
    
    if passed == total:
        print("[SUCCESS] 所有演示成功！防御者智能体系统工作正常。")
    else:
        print("[WARNING] 部分演示失败。")


if __name__ == "__main__":
    main()
