#!/usr/bin/env python3
"""调试tiny场景加载问题"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def debug_tiny_scenario():
    try:
        import nasim
        print("尝试加载tiny场景...")
        env = nasim.make_benchmark('tiny', seed=42, fully_obs=True, flat_actions=True, flat_obs=True)
        print("成功加载tiny场景！")
        print(f"场景名称: {env.scenario.name}")
        print(f"子网配置: {env.scenario.subnets}")
        print(f"敏感主机: {env.scenario.sensitive_hosts}")
        print(f"地址空间: {env.scenario.address_space}")
        
    except Exception as e:
        print(f"加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tiny_scenario()
