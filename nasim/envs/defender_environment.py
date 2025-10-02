"""防御者环境类

这个模块实现了防御者智能体的环境，包括：
1. 全局观察系统
2. 成本计算系统
3. 奖励评估系统
4. 主机价值管理
"""

import gym
import numpy as np
from gym import spaces
from .environment import NASimEnv
from .defender_action import DefenderActionSpace, DefenderActionResult
from .observation import Observation
from .state import State


class DefenderEnv(gym.Env):
    """防御者环境类
    
    为防御者智能体提供环境接口，具有全局视角
    """
    
    def __init__(self, scenario, attacker_env=None):
        """初始化防御者环境
        
        Args:
            scenario: 场景对象
            attacker_env: 攻击者环境（可选）
        """
        self.scenario = scenario
        self.attacker_env = attacker_env
        
        # 初始化网络和状态
        from .network import Network
        self.network = Network(scenario)
        self.current_state = State.generate_initial_state(self.network)
        
        # 初始化动作空间
        self.action_space = DefenderActionSpace(scenario)
        
        # 初始化观察空间（全局视角）
        self._setup_observation_space()
        
        # 成本和奖励系统
        self.total_cost = 0.0
        self.host_values = self._initialize_host_values()
        self.compromised_penalty = 0.0
        
        # 环境状态
        self.steps = 0
        self.max_steps = scenario.step_limit or 1000
        
    def _setup_observation_space(self):
        """设置观察空间（全局视角）"""
        # 全局观察包括：
        # 1. 网络拓扑信息
        # 2. 所有主机状态
        # 3. 攻击者活动信息
        # 4. 当前成本信息
        
        num_hosts = len(self.scenario.address_space)
        num_subnets = len(self.scenario.subnets)
        
        # 主机状态特征：地址、被攻破状态、可达性、发现状态、访问级别、价值、服务、OS、进程
        host_features = (
            2 +  # 地址 (subnet, host)
            3 +  # 被攻破、可达、发现状态
            1 +  # 访问级别
            2 +  # 价值、发现价值
            self.scenario.num_services +  # 服务
            self.scenario.num_os +  # 操作系统
            self.scenario.num_processes  # 进程
        )
        
        # 全局特征：总成本、被攻破主机数、高价值主机被攻破数
        global_features = 3
        
        # 网络拓扑特征
        topology_features = num_subnets * num_subnets
        
        total_features = num_hosts * host_features + global_features + topology_features
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(total_features,), dtype=np.float32
        )
    
    def _initialize_host_values(self):
        """初始化主机价值"""
        host_values = {}
        
        for host_addr in self.scenario.address_space:
            # 区分普通主机和高价值主机
            if host_addr in self.scenario.sensitive_addresses:
                # 高价值主机
                base_value = self.scenario.sensitive_hosts[host_addr]
                host_values[host_addr] = {
                    'type': 'high_value',
                    'base_value': base_value,
                    'compromise_penalty': base_value * 2.0  # 被攻破时的惩罚
                }
            else:
                # 普通主机
                host = self.network.hosts[host_addr]
                base_value = host.value
                host_values[host_addr] = {
                    'type': 'normal',
                    'base_value': base_value,
                    'compromise_penalty': base_value * 1.0  # 被攻破时的惩罚
                }
        
        return host_values
    
    def reset(self):
        """重置环境"""
        self.current_state = self.network.reset(self.current_state)
        self.total_cost = 0.0
        self.compromised_penalty = 0.0
        self.steps = 0
        
        return self._get_observation()
    
    def step(self, action):
        """执行一步动作
        
        Args:
            action: 动作索引或动作对象
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        if isinstance(action, (int, np.integer)):
            action_obj = self.action_space.get_action(int(action))
        else:
            action_obj = action
        
        # 执行防御动作
        new_state, action_result, action_cost = action_obj.execute(
            self.current_state, self.network
        )
        
        # 更新状态
        self.current_state = new_state
        self.total_cost += action_cost
        
        # 计算奖励
        reward = self._calculate_reward(action_result, action_cost)
        
        # 检查是否结束
        done = self._is_done()
        
        # 更新步数
        self.steps += 1
        
        # 构建信息
        info = {
            'action_result': action_result,
            'action_cost': action_cost,
            'total_cost': self.total_cost,
            'compromised_penalty': self.compromised_penalty,
            'steps': self.steps
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """获取全局观察"""
        observation = []
        
        # 1. 主机状态信息
        for host_addr in self.scenario.address_space:
            host = self.current_state.get_host(host_addr)
            host_obs = self._get_host_observation(host_addr, host)
            observation.extend(host_obs)
        
        # 2. 全局统计信息
        global_obs = self._get_global_observation()
        observation.extend(global_obs)
        
        # 3. 网络拓扑信息
        topology_obs = self._get_topology_observation()
        observation.extend(topology_obs)
        
        return np.array(observation, dtype=np.float32)
    
    def _get_host_observation(self, host_addr, host):
        """获取单个主机的观察"""
        obs = []
        
        # 地址信息（归一化）
        max_subnet = len(self.scenario.subnets) - 1
        max_host = max(self.scenario.subnets)
        obs.append(host_addr[0] / max_subnet if max_subnet > 0 else 0)
        obs.append(host_addr[1] / max_host if max_host > 0 else 0)
        
        # 状态信息
        obs.append(1.0 if host.compromised else 0.0)
        obs.append(1.0 if host.reachable else 0.0)
        obs.append(1.0 if host.discovered else 0.0)
        
        # 访问级别（归一化）
        if hasattr(host.access, 'value'):
            obs.append(host.access.value / 2.0)  # ROOT=2, USER=1, NONE=0
        else:
            obs.append(float(host.access) / 2.0)  # 如果是数值类型
        
        # 价值信息（归一化）
        max_value = max(self.host_values[addr]['base_value'] 
                       for addr in self.scenario.address_space)
        obs.append(host.value / max_value if max_value > 0 else 0)
        obs.append(host.discovery_value / max_value if max_value > 0 else 0)
        
        # 服务信息
        obs.extend([1.0 if service else 0.0 for service in host.services])
        
        # 操作系统信息
        obs.extend([1.0 if os else 0.0 for os in host.os])
        
        # 进程信息
        obs.extend([1.0 if process else 0.0 for process in host.processes])
        
        return obs
    
    def _get_global_observation(self):
        """获取全局统计观察"""
        obs = []
        
        # 总成本（归一化）
        max_possible_cost = len(self.action_space.actions) * max(self.action_space.get_action_costs())
        obs.append(self.total_cost / max_possible_cost if max_possible_cost > 0 else 0)
        
        # 被攻破主机数量（归一化）
        compromised_count = sum(1 for addr in self.scenario.address_space 
                               if self.current_state.host_compromised(addr))
        obs.append(compromised_count / len(self.scenario.address_space))
        
        # 高价值主机被攻破数量（归一化）
        high_value_compromised = sum(1 for addr in self.scenario.sensitive_addresses
                                   if self.current_state.host_compromised(addr))
        obs.append(high_value_compromised / len(self.scenario.sensitive_addresses) 
                  if self.scenario.sensitive_addresses else 0)
        
        return obs
    
    def _get_topology_observation(self):
        """获取网络拓扑观察"""
        # 将拓扑矩阵展平
        if hasattr(self.scenario.topology, 'flatten'):
            topology_flat = self.scenario.topology.flatten()
            return topology_flat.tolist()
        else:
            # 如果是嵌套列表，手动展平
            topology_flat = []
            for row in self.scenario.topology:
                if isinstance(row, list):
                    topology_flat.extend(row)
                else:
                    topology_flat.append(row)
            return topology_flat
    
    def _calculate_reward(self, action_result, action_cost):
        """计算奖励
        
        奖励函数考虑：
        1. 动作成本（负奖励）
        2. 防止主机被攻破的收益
        3. 高价值主机的额外保护收益
        """
        reward = 0.0
        
        # 动作成本惩罚
        reward -= action_cost
        
        # 计算当前被攻破主机的惩罚
        current_penalty = 0.0
        for host_addr in self.scenario.address_space:
            if self.current_state.host_compromised(host_addr):
                current_penalty += self.host_values[host_addr]['compromise_penalty']
        
        # 如果惩罚减少了，给予正奖励（表示防御成功）
        penalty_reduction = self.compromised_penalty - current_penalty
        reward += penalty_reduction
        
        # 更新惩罚记录
        self.compromised_penalty = current_penalty
        
        # 成功执行防御动作的小额奖励
        if action_result.get('success', False) and action_result.get('action') != 'noop':
            reward += 1.0
        
        return reward
    
    def _is_done(self):
        """检查是否结束"""
        # 达到最大步数
        if self.steps >= self.max_steps:
            return True
        
        # 所有高价值主机都被攻破
        if all(self.current_state.host_compromised(addr) 
               for addr in self.scenario.sensitive_addresses):
            return True
        
        return False
    
    def get_action_mask(self):
        """获取可用动作掩码"""
        mask = np.ones(self.action_space.n, dtype=bool)
        
        # 检查每个动作是否可执行
        for i, action in enumerate(self.action_space.actions):
            if hasattr(action, 'subnet_id'):
                # 子网地址交换：检查子网是否有足够主机
                subnet_hosts = [addr for addr in self.scenario.address_space 
                               if addr[0] == action.subnet_id]
                if len(subnet_hosts) < 2:
                    mask[i] = False
            
            elif hasattr(action, 'host1_addr'):
                # 主机地址交换：检查主机是否存在
                if (action.host1_addr not in self.scenario.address_space or
                    action.host2_addr not in self.scenario.address_space):
                    mask[i] = False
            
            elif hasattr(action, 'host_addr'):
                # 协议交换：检查主机是否有足够服务
                if action.host_addr not in self.scenario.address_space:
                    mask[i] = False
                else:
                    host = self.current_state.get_host(action.host_addr)
                    # 计算运行中的服务数量
                    running_services = sum(1 for service in host.services if service)
                    if running_services < 2:
                        mask[i] = False
        
        return mask
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== 防御者环境状态 (步数: {self.steps}) ===")
            print(f"总成本: {self.total_cost:.2f}")
            print(f"被攻破惩罚: {self.compromised_penalty:.2f}")
            
            print("\n主机状态:")
            for host_addr in self.scenario.address_space:
                host = self.current_state.get_host(host_addr)
                host_type = self.host_values[host_addr]['type']
                status = "被攻破" if host.compromised else "安全"
                print(f"  {host_addr} ({host_type}): {status}")
            
            print(f"\n可用动作数: {sum(self.get_action_mask())}/{self.action_space.n}")
    
    def get_total_evaluation(self):
        """获取总体评估
        
        Returns:
            dict: 包含各种评估指标的字典
        """
        evaluation = {
            'total_cost': self.total_cost,
            'compromised_penalty': self.compromised_penalty,
            'total_score': -(self.total_cost + self.compromised_penalty),
            'compromised_hosts': sum(1 for addr in self.scenario.address_space 
                                   if self.current_state.host_compromised(addr)),
            'compromised_high_value': sum(1 for addr in self.scenario.sensitive_addresses
                                        if self.current_state.host_compromised(addr)),
            'steps_taken': self.steps,
            'efficiency': self.total_cost / max(self.steps, 1)
        }
        
        return evaluation
