#!/usr/bin/env python3
"""
防御者智能体测试脚本

这个脚本用于测试防御者智能体的基本功能
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_defender_actions():
    """测试防御者动作"""
    print("测试防御者动作...")
    
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
        
        # 测试动作空间创建
        action_space = DefenderActionSpace(scenario)
        print(f"[OK] 创建动作空间成功，包含 {action_space.n} 个动作")
        
        # 测试各种动作类型
        noop = DefenderNoOp()
        print(f"[OK] 空动作创建成功，成本: {noop.cost}")
        
        subnet_swap = SubnetAddressSwap(1, cost=10.0)
        print(f"[OK] 子网地址交换创建成功，成本: {subnet_swap.cost}")
        
        host_swap = HostAddressSwap((1, 0), (1, 1), cost=5.0)
        print(f"[OK] 主机地址交换创建成功，成本: {host_swap.cost}")
        
        protocol_swap = ProtocolSwap((1, 0), cost=3.0)
        print(f"[OK] 协议交换创建成功，成本: {protocol_swap.cost}")
        
        print("防御者动作测试通过！\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] 防御者动作测试失败: {e}")
        return False


def test_defender_environment():
    """测试防御者环境"""
    print("测试防御者环境...")
    
    try:
        import nasim
        from nasim.envs.defender_environment import DefenderEnv
        
        # 创建场景
        env = nasim.make_benchmark('tiny', seed=42, fully_obs=True, flat_actions=True, flat_obs=True)
        scenario = env.scenario
        
        # 创建防御者环境
        defender_env = DefenderEnv(scenario)
        print(f"[OK] 防御者环境创建成功")
        print(f"  观察空间维度: {defender_env.observation_space.shape}")
        print(f"  动作空间大小: {defender_env.action_space.n}")
        
        # 测试重置
        obs = defender_env.reset()
        print(f"[OK] 环境重置成功，观察维度: {obs.shape}")
        
        # 测试动作执行
        action = 0  # 空动作
        obs, reward, done, info = defender_env.step(action)
        print(f"[OK] 动作执行成功，奖励: {reward:.2f}")
        
        # 测试动作掩码
        mask = defender_env.get_action_mask()
        print(f"[OK] 动作掩码获取成功，可用动作: {sum(mask)}/{len(mask)}")
        
        # 测试评估
        evaluation = defender_env.get_total_evaluation()
        print(f"[OK] 总体评估成功，总分: {evaluation['total_score']:.2f}")
        
        print("防御者环境测试通过！\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] 防御者环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_defender_agent():
    """测试防御者智能体"""
    print("测试防御者智能体...")
    
    try:
        from nasim.agents.defender_agent import create_defender_agent
        
        # 创建防御者智能体（使用较小的参数进行快速测试）
        agent, env = create_defender_agent(
            'tiny',
            seed=42,
            training_steps=100,  # 很少的训练步数用于测试
            hidden_sizes=[32, 32],
            batch_size=16,
            verbose=False
        )
        print(f"[OK] 防御者智能体创建成功")
        
        # 测试获取动作
        obs = env.reset()
        action_mask = env.get_action_mask()
        action = agent.get_action(obs, action_mask, epsilon=1.0)  # 完全随机
        print(f"[OK] 动作获取成功，动作索引: {action}")
        
        # 测试简短训练
        print("开始简短训练测试...")
        agent.train()
        print(f"[OK] 训练完成")
        
        # 测试评估
        ep_return, ep_cost, ep_steps, evaluation = agent.run_eval_episode(env, render=False)
        print(f"[OK] 评估完成，回合奖励: {ep_return:.2f}, 成本: {ep_cost:.2f}")
        
        print("防御者智能体测试通过！\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] 防御者智能体测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adversarial_environment():
    """测试对抗环境"""
    print("测试对抗环境...")
    
    try:
        from nasim.envs.adversarial_environment import create_adversarial_env
        
        # 创建对抗环境
        adv_env = create_adversarial_env('tiny', mode='alternating', seed=42)
        print(f"[OK] 对抗环境创建成功")
        print(f"  观察空间维度: {adv_env.observation_space.shape}")
        print(f"  当前轮次: {adv_env.current_turn}")
        
        # 测试重置
        obs = adv_env.reset()
        print(f"[OK] 环境重置成功，观察维度: {obs.shape}")
        
        # 测试几步动作
        for step in range(3):
            action = adv_env.action_space.sample()
            obs, reward, done, info = adv_env.step(action)
            print(f"  步骤 {step+1}: 轮次={info['turn']}, 奖励={reward:.2f}")
            if done:
                break
        
        # 测试评估
        evaluation = adv_env.get_evaluation()
        print(f"[OK] 对抗评估成功，攻击成功: {evaluation['attack_success']}")
        
        print("对抗环境测试通过！\n")
        return True
        
    except Exception as e:
        print(f"[FAIL] 对抗环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("防御者智能体系统测试")
    print("=" * 50)
    
    tests = [
        test_defender_actions,
        test_defender_environment,
        test_defender_agent,
        test_adversarial_environment
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("[SUCCESS] 所有测试通过！防御者智能体系统工作正常。")
        return True
    else:
        print("[ERROR] 部分测试失败，请检查错误信息。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
