"""防御者动作相关类

这个模块包含防御者智能体的不同动作类，包括：
1. 空动作 (NoOp)
2. 同子网中全部主机地址随机互换 (SubnetAddressSwap)
3. 指定两台主机地址互换 (HostAddressSwap)
4. 同主机内全部协议随机互换 (ProtocolSwap)
"""

import random
import numpy as np
from gym import spaces
from .action import Action
from .utils import AccessLevel


class DefenderAction(Action):
    """防御者动作基类
    
    继承自基础Action类，添加防御者特有的属性
    """
    
    def __init__(self, name, cost, **kwargs):
        # 防御者动作不需要target，使用默认值
        super().__init__(name=name, target=(0, 0), cost=cost, **kwargs)
        self.is_defender_action = True


class DefenderNoOp(DefenderAction):
    """防御者空动作
    
    不执行任何防御操作，成本为0
    """
    
    def __init__(self):
        super().__init__(name="defender_noop", cost=0.0)
    
    def execute(self, state, network):
        """执行空动作
        
        Returns:
            tuple: (new_state, action_result, cost)
        """
        return state.copy(), {"success": True, "action": "noop"}, 0.0


class SubnetAddressSwap(DefenderAction):
    """同子网中全部主机地址随机互换
    
    在指定子网内随机重新分配所有主机的地址
    """
    
    def __init__(self, subnet_id, cost=10.0):
        super().__init__(name="subnet_address_swap", cost=cost)
        self.subnet_id = subnet_id
    
    def execute(self, state, network):
        """执行子网地址互换
        
        Args:
            state: 当前网络状态
            network: 网络对象
            
        Returns:
            tuple: (new_state, action_result, cost)
        """
        new_state = state.copy()
        
        # 获取指定子网中的所有主机
        subnet_hosts = []
        for host_addr in network.address_space:
            if host_addr[0] == self.subnet_id:
                subnet_hosts.append(host_addr)
        
        if len(subnet_hosts) < 2:
            return new_state, {"success": False, "reason": "子网主机数量不足"}, 0.0
        
        # 创建地址映射
        old_addresses = subnet_hosts.copy()
        new_addresses = subnet_hosts.copy()
        random.shuffle(new_addresses)
        
        # 确保至少有一个地址发生变化
        if old_addresses == new_addresses:
            # 交换前两个地址
            new_addresses[0], new_addresses[1] = new_addresses[1], new_addresses[0]
        
        # 执行地址交换
        address_mapping = dict(zip(old_addresses, new_addresses))
        self._swap_addresses(new_state, network, address_mapping)
        
        result = {
            "success": True,
            "action": "subnet_address_swap",
            "subnet_id": self.subnet_id,
            "swapped_hosts": len(subnet_hosts),
            "mapping": address_mapping
        }
        
        return new_state, result, self.cost
    
    def _swap_addresses(self, state, network, address_mapping):
        """执行地址交换的具体逻辑"""
        # 创建临时存储
        temp_hosts = {}
        
        # 保存原始主机状态
        for old_addr in address_mapping:
            temp_hosts[old_addr] = state.get_host(old_addr).copy()
        
        # 重新分配主机状态
        for old_addr, new_addr in address_mapping.items():
            if old_addr != new_addr:
                host_state = temp_hosts[old_addr]
                # 直接更新状态，不修改地址属性
                state.update_host(new_addr, host_state)


class HostAddressSwap(DefenderAction):
    """指定两台主机地址互换
    
    交换两台指定主机的地址
    """
    
    def __init__(self, host1_addr, host2_addr, cost=5.0):
        super().__init__(name="host_address_swap", cost=cost)
        self.host1_addr = host1_addr
        self.host2_addr = host2_addr
    
    def execute(self, state, network):
        """执行主机地址互换
        
        Args:
            state: 当前网络状态
            network: 网络对象
            
        Returns:
            tuple: (new_state, action_result, cost)
        """
        new_state = state.copy()
        
        # 检查主机是否存在
        if (self.host1_addr not in network.address_space or 
            self.host2_addr not in network.address_space):
            return new_state, {"success": False, "reason": "指定主机不存在"}, 0.0
        
        # 检查是否为同一主机
        if self.host1_addr == self.host2_addr:
            return new_state, {"success": False, "reason": "不能与自己交换地址"}, 0.0
        
        # 执行地址交换 - 直接交换主机状态
        host1_state = new_state.get_host(self.host1_addr)
        host2_state = new_state.get_host(self.host2_addr)
        
        # 创建临时副本
        temp_host1 = host1_state.copy()
        temp_host2 = host2_state.copy()
        
        # 交换状态
        new_state.update_host(self.host1_addr, temp_host2)
        new_state.update_host(self.host2_addr, temp_host1)
        
        result = {
            "success": True,
            "action": "host_address_swap",
            "host1": self.host1_addr,
            "host2": self.host2_addr
        }
        
        return new_state, result, self.cost


class ProtocolSwap(DefenderAction):
    """同主机内全部协议随机互换
    
    在指定主机内随机重新分配所有服务的端口
    """
    
    def __init__(self, host_addr, cost=3.0):
        super().__init__(name="protocol_swap", cost=cost)
        self.host_addr = host_addr
    
    def execute(self, state, network):
        """执行协议互换
        
        Args:
            state: 当前网络状态
            network: 网络对象
            
        Returns:
            tuple: (new_state, action_result, cost)
        """
        new_state = state.copy()
        
        # 检查主机是否存在
        if self.host_addr not in network.address_space:
            return new_state, {"success": False, "reason": "指定主机不存在"}, 0.0
        
        host = new_state.get_host(self.host_addr)
        
        # 获取主机上运行的所有服务
        running_services = []
        for service_idx, is_running in enumerate(host.services):
            if is_running:
                running_services.append(service_idx)
        
        if len(running_services) < 2:
            return new_state, {"success": False, "reason": "主机服务数量不足"}, 0.0
        
        # 创建服务重新映射
        old_services = running_services.copy()
        new_services = running_services.copy()
        random.shuffle(new_services)
        
        # 确保至少有一个服务发生变化
        if old_services == new_services:
            new_services[0], new_services[1] = new_services[1], new_services[0]
        
        # 执行协议交换
        service_mapping = dict(zip(old_services, new_services))
        self._swap_protocols(host, service_mapping)
        
        result = {
            "success": True,
            "action": "protocol_swap",
            "host": self.host_addr,
            "swapped_services": len(running_services),
            "mapping": service_mapping
        }
        
        return new_state, result, self.cost
    
    def _swap_protocols(self, host, service_mapping):
        """执行协议交换的具体逻辑"""
        # 保存原始服务状态
        original_services = host.services.copy()
        
        # 重新分配服务
        for old_service, new_service in service_mapping.items():
            if old_service != new_service:
                # 交换服务状态
                host.services[new_service] = original_services[old_service]
                host.services[old_service] = original_services[new_service]


class DefenderActionSpace(spaces.Discrete):
    """防御者动作空间
    
    包含所有可能的防御者动作
    """
    
    def __init__(self, scenario):
        """初始化防御者动作空间
        
        Args:
            scenario: 场景对象
        """
        self.scenario = scenario
        self.actions = self._generate_actions()
        super().__init__(len(self.actions))
    
    def _generate_actions(self):
        """生成所有可能的防御者动作"""
        actions = []
        
        # 添加空动作
        actions.append(DefenderNoOp())
        
        # 为每个子网添加地址交换动作
        for subnet_id in range(1, len(self.scenario.subnets)):  # 跳过internet子网
            if self.scenario.subnets[subnet_id] > 1:  # 至少需要2台主机
                actions.append(SubnetAddressSwap(subnet_id))
        
        # 为每对主机添加地址交换动作
        host_addresses = list(self.scenario.address_space)
        for i in range(len(host_addresses)):
            for j in range(i + 1, len(host_addresses)):
                # 只允许同子网内的主机交换地址
                if host_addresses[i][0] == host_addresses[j][0]:
                    actions.append(HostAddressSwap(host_addresses[i], host_addresses[j]))
        
        # 为每台主机添加协议交换动作
        for host_addr in self.scenario.address_space:
            actions.append(ProtocolSwap(host_addr))
        
        return actions
    
    def get_action(self, action_idx):
        """获取指定索引的动作
        
        Args:
            action_idx: 动作索引
            
        Returns:
            DefenderAction: 对应的防御者动作
        """
        return self.actions[action_idx]
    
    def get_action_costs(self):
        """获取所有动作的成本
        
        Returns:
            list: 所有动作的成本列表
        """
        return [action.cost for action in self.actions]


class DefenderActionResult:
    """防御者动作结果
    
    存储防御者动作执行的结果
    """
    
    def __init__(self, success, action_type, cost, details=None):
        """初始化动作结果
        
        Args:
            success: 动作是否成功
            action_type: 动作类型
            cost: 动作成本
            details: 详细信息
        """
        self.success = success
        self.action_type = action_type
        self.cost = cost
        self.details = details or {}
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "success": self.success,
            "action_type": self.action_type,
            "cost": self.cost,
            "details": self.details
        }
