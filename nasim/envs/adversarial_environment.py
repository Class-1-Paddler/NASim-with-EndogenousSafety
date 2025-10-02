"""对抗环境类

这个模块实现了攻击者和防御者的对抗环境，包括：
1. 双智能体交互
2. 同步和异步执行模式
3. 综合评估系统
"""

import gym
import numpy as np
from gym import spaces
from .environment import NASimEnv
from .defender_environment import DefenderEnv


class AdversarialEnv(gym.Env):
    """攻击者-防御者对抗环境
    
    支持两个智能体在同一网络环境中进行对抗
    """
    
    def __init__(self, scenario, mode='alternating', defender_frequency=1):
        """初始化对抗环境
        
        Args:
            scenario: 场景对象
            mode: 执行模式 ('alternating', 'simultaneous')
            defender_frequency: 防御者动作频率（每N个攻击者动作执行一次防御者动作）
        """
        self.scenario = scenario
        self.mode = mode
        self.defender_frequency = defender_frequency
        
        # 创建攻击者和防御者环境
        self.attacker_env = NASimEnv(scenario, fully_obs=False, flat_actions=True, flat_obs=True)
        self.defender_env = DefenderEnv(scenario, self.attacker_env)
        
        # 动作空间（联合动作空间）
        if mode == 'simultaneous':
            self.action_space = spaces.Tuple([
                self.attacker_env.action_space,
                self.defender_env.action_space
            ])
        else:
            # 交替模式下，动作空间取决于当前轮到谁
            self.action_space = self.attacker_env.action_space
        
        # 观察空间（联合观察空间）
        attacker_obs_dim = self.attacker_env.observation_space.shape[0]
        defender_obs_dim = self.defender_env.observation_space.shape[0]
        total_obs_dim = attacker_obs_dim + defender_obs_dim + 2  # +2 for turn indicator
        
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(total_obs_dim,), dtype=np.float32
        )
        
        # 环境状态
        self.current_turn = 'attacker'  # 'attacker' or 'defender'
        self.attacker_steps = 0
        self.defender_steps = 0
        self.total_steps = 0
        self.max_steps = scenario.step_limit or 1000
        
        # 评估指标
        self.attacker_rewards = []
        self.defender_rewards = []
        self.attack_success_rate = 0.0
        self.defense_effectiveness = 0.0
    
    def reset(self):
        """重置环境"""
        # 重置两个环境
        attacker_obs = self.attacker_env.reset()
        defender_obs = self.defender_env.reset()
        
        # 重置状态
        self.current_turn = 'attacker'
        self.attacker_steps = 0
        self.defender_steps = 0
        self.total_steps = 0
        
        self.attacker_rewards = []
        self.defender_rewards = []
        
        return self._get_joint_observation(attacker_obs, defender_obs)
    
    def step(self, action):
        """执行一步动作
        
        Args:
            action: 动作（格式取决于执行模式）
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.mode == 'simultaneous':
            return self._step_simultaneous(action)
        else:
            return self._step_alternating(action)
    
    def _step_simultaneous(self, actions):
        """同时执行模式"""
        attacker_action, defender_action = actions
        
        # 同时执行两个动作
        attacker_obs, attacker_reward, attacker_done, attacker_info = \
            self.attacker_env.step(attacker_action)
        
        defender_obs, defender_reward, defender_done, defender_info = \
            self.defender_env.step(defender_action)
        
        # 同步状态（防御者的动作可能影响攻击者的环境）
        self._sync_environments()
        
        # 计算联合奖励
        joint_reward = self._calculate_joint_reward(
            attacker_reward, defender_reward, attacker_info, defender_info
        )
        
        # 检查结束条件
        done = attacker_done or defender_done or self.total_steps >= self.max_steps
        
        # 更新统计
        self.attacker_rewards.append(attacker_reward)
        self.defender_rewards.append(defender_reward)
        self.total_steps += 1
        
        # 构建信息
        info = {
            'attacker_info': attacker_info,
            'defender_info': defender_info,
            'attacker_reward': attacker_reward,
            'defender_reward': defender_reward,
            'total_steps': self.total_steps
        }
        
        return self._get_joint_observation(attacker_obs, defender_obs), joint_reward, done, info
    
    def _step_alternating(self, action):
        """交替执行模式"""
        if self.current_turn == 'attacker':
            return self._step_attacker(action)
        else:
            return self._step_defender(action)
    
    def _step_attacker(self, action):
        """执行攻击者动作"""
        attacker_obs, attacker_reward, attacker_done, attacker_info = \
            self.attacker_env.step(action)
        
        # 获取防御者观察（可能因攻击者动作而改变）
        defender_obs = self.defender_env._get_observation()
        
        self.attacker_steps += 1
        self.total_steps += 1
        
        # 检查是否轮到防御者
        if self.attacker_steps % self.defender_frequency == 0:
            self.current_turn = 'defender'
            self.action_space = self.defender_env.action_space
        
        # 检查结束条件
        done = attacker_done or self.total_steps >= self.max_steps
        
        self.attacker_rewards.append(attacker_reward)
        
        info = {
            'turn': 'attacker',
            'attacker_info': attacker_info,
            'attacker_reward': attacker_reward,
            'next_turn': self.current_turn,
            'total_steps': self.total_steps
        }
        
        return self._get_joint_observation(attacker_obs, defender_obs), attacker_reward, done, info
    
    def _step_defender(self, action):
        """执行防御者动作"""
        defender_obs, defender_reward, defender_done, defender_info = \
            self.defender_env.step(action)
        
        # 同步环境状态
        self._sync_environments()
        
        # 获取攻击者观察（可能因防御者动作而改变）
        attacker_obs = self.attacker_env.last_obs.numpy_flat() if hasattr(self.attacker_env, 'last_obs') else np.zeros(self.attacker_env.observation_space.shape)
        
        self.defender_steps += 1
        self.total_steps += 1
        
        # 切换回攻击者
        self.current_turn = 'attacker'
        self.action_space = self.attacker_env.action_space
        
        # 检查结束条件
        done = defender_done or self.total_steps >= self.max_steps
        
        self.defender_rewards.append(defender_reward)
        
        info = {
            'turn': 'defender',
            'defender_info': defender_info,
            'defender_reward': defender_reward,
            'next_turn': self.current_turn,
            'total_steps': self.total_steps
        }
        
        return self._get_joint_observation(attacker_obs, defender_obs), defender_reward, done, info
    
    def _sync_environments(self):
        """同步两个环境的状态
        
        防御者的动作（如地址交换）会影响攻击者环境的状态
        """
        # 将防御者环境的状态同步到攻击者环境
        self.attacker_env.current_state = self.defender_env.current_state.copy()
        
        # 更新攻击者的观察
        if hasattr(self.attacker_env, 'last_obs'):
            self.attacker_env.last_obs = self.attacker_env.current_state.get_initial_observation(
                self.attacker_env.fully_obs
            )
    
    def _get_joint_observation(self, attacker_obs, defender_obs):
        """获取联合观察"""
        # 添加轮次指示器
        turn_indicator = [1.0, 0.0] if self.current_turn == 'attacker' else [0.0, 1.0]
        
        joint_obs = np.concatenate([
            attacker_obs.flatten(),
            defender_obs.flatten(),
            turn_indicator
        ])
        
        return joint_obs.astype(np.float32)
    
    def _calculate_joint_reward(self, attacker_reward, defender_reward, attacker_info, defender_info):
        """计算联合奖励
        
        这里可以实现不同的奖励策略：
        1. 零和游戏：attacker_reward - defender_reward
        2. 合作游戏：attacker_reward + defender_reward
        3. 竞争游戏：基于目标达成情况
        """
        # 实现零和游戏策略
        return attacker_reward - defender_reward
    
    def get_action_mask(self):
        """获取当前玩家的动作掩码"""
        if self.current_turn == 'attacker':
            return self.attacker_env.get_action_mask()
        else:
            return self.defender_env.get_action_mask()
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            print(f"\n=== 对抗环境状态 (总步数: {self.total_steps}) ===")
            print(f"当前轮次: {self.current_turn}")
            print(f"攻击者步数: {self.attacker_steps}")
            print(f"防御者步数: {self.defender_steps}")
            
            print("\n--- 攻击者环境 ---")
            self.attacker_env.render(mode)
            
            print("\n--- 防御者环境 ---")
            self.defender_env.render(mode)
            
            if self.attacker_rewards and self.defender_rewards:
                print(f"\n--- 奖励统计 ---")
                print(f"攻击者平均奖励: {np.mean(self.attacker_rewards):.2f}")
                print(f"防御者平均奖励: {np.mean(self.defender_rewards):.2f}")
    
    def get_evaluation(self):
        """获取对抗评估结果"""
        # 计算攻击成功率
        if hasattr(self.attacker_env, 'goal_reached'):
            attack_success = self.attacker_env.goal_reached()
        else:
            attack_success = False
        
        # 计算防御效果
        defender_eval = self.defender_env.get_total_evaluation()
        
        evaluation = {
            'attack_success': attack_success,
            'attacker_total_reward': sum(self.attacker_rewards) if self.attacker_rewards else 0,
            'defender_total_reward': sum(self.defender_rewards) if self.defender_rewards else 0,
            'attacker_avg_reward': np.mean(self.attacker_rewards) if self.attacker_rewards else 0,
            'defender_avg_reward': np.mean(self.defender_rewards) if self.defender_rewards else 0,
            'total_steps': self.total_steps,
            'attacker_steps': self.attacker_steps,
            'defender_steps': self.defender_steps,
            'defender_evaluation': defender_eval
        }
        
        return evaluation
    
    def set_agents(self, attacker_agent=None, defender_agent=None):
        """设置智能体（用于自动对战）"""
        self.attacker_agent = attacker_agent
        self.defender_agent = defender_agent
    
    def run_episode(self, max_steps=None, render=False):
        """运行一个完整的对抗回合"""
        if max_steps is None:
            max_steps = self.max_steps
        
        obs = self.reset()
        done = False
        step_count = 0
        
        if render:
            self.render()
        
        while not done and step_count < max_steps:
            if self.mode == 'simultaneous':
                # 同时获取两个智能体的动作
                if hasattr(self, 'attacker_agent') and hasattr(self, 'defender_agent'):
                    attacker_action = self.attacker_agent.get_action(obs[:self.attacker_env.observation_space.shape[0]])
                    defender_action = self.defender_agent.get_action(obs[self.attacker_env.observation_space.shape[0]:-2])
                    action = (attacker_action, defender_action)
                else:
                    # 随机动作
                    action = (self.attacker_env.action_space.sample(), self.defender_env.action_space.sample())
            else:
                # 交替模式
                if self.current_turn == 'attacker':
                    if hasattr(self, 'attacker_agent'):
                        action = self.attacker_agent.get_action(obs[:self.attacker_env.observation_space.shape[0]])
                    else:
                        action = self.attacker_env.action_space.sample()
                else:
                    if hasattr(self, 'defender_agent'):
                        action = self.defender_agent.get_action(obs[self.attacker_env.observation_space.shape[0]:-2])
                    else:
                        action = self.defender_env.action_space.sample()
            
            obs, reward, done, info = self.step(action)
            step_count += 1
            
            if render:
                print(f"\n步骤 {step_count}, 奖励: {reward:.2f}")
                self.render()
        
        return self.get_evaluation()


def create_adversarial_env(scenario_name, mode='alternating', **kwargs):
    """创建对抗环境的便捷函数"""
    import nasim
    
    # 加载场景
    env = nasim.make_benchmark(scenario_name, **kwargs)
    scenario = env.scenario
    
    # 创建对抗环境
    adv_env = AdversarialEnv(scenario, mode=mode)
    
    return adv_env


if __name__ == "__main__":
    # 测试对抗环境
    env = create_adversarial_env('tiny', mode='alternating')
    evaluation = env.run_episode(render=True)
    print("\n最终评估:")
    for key, value in evaluation.items():
        print(f"{key}: {value}")
