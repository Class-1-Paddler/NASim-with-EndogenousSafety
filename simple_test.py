#!/usr/bin/env python3
"""
简单的防御者智能体测试脚本
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_basic_imports():
    """测试基本导入"""
    print("测试基本导入...")
    
    try:
        # 测试防御者动作导入
        from nasim.envs.defender_action import (
            DefenderNoOp, SubnetAddressSwap, HostAddressSwap, 
            ProtocolSwap, DefenderActionSpace
        )
        print("[OK] 防御者动作模块导入成功")
        
        # 测试基本动作创建
        noop = DefenderNoOp()
        print(f"[OK] 空动作创建成功，成本: {noop.cost}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 基本导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_scenario():
    """测试模拟场景"""
    print("测试模拟场景...")
    
    try:
        from nasim.envs.defender_action import DefenderActionSpace
        
        # 创建简单的场景对象模拟
        class MockScenario:
            def __init__(self):
                self.subnets = [0, 3, 2]  # internet, subnet1(3 hosts), subnet2(2 hosts)
                self.address_space = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
        
        scenario = MockScenario()
        
        # 测试动作空间创建
        action_space = DefenderActionSpace(scenario)
        print(f"[OK] 创建动作空间成功，包含 {action_space.n} 个动作")
        
        # 测试获取动作
        for i in range(min(3, action_space.n)):
            action = action_space.get_action(i)
            print(f"[OK] 动作 {i}: {action.__class__.__name__}, 成本: {action.cost}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] 模拟场景测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 50)
    print("防御者智能体简单测试")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_mock_scenario,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # 空行分隔
    
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("[SUCCESS] 基本测试通过！")
        return True
    else:
        print("[ERROR] 部分测试失败。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
