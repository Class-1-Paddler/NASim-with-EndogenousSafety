"""防御者智能体实现

这个模块实现了基于深度强化学习的防御者智能体，包括：
1. DQN防御者智能体
2. 防御策略网络
3. 训练和评估功能
"""

import random
import numpy as np
from gym import error
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ..envs.defender_environment import DefenderEnv


class DefenderReplayMemory:
    """防御者经验回放缓冲区"""
    
    def __init__(self, capacity, s_dims, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.a_buf = np.zeros((capacity, 1), dtype=np.int64)
        self.next_s_buf = np.zeros((capacity, *s_dims), dtype=np.float32)
        self.r_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.mask_buf = None  # 动作掩码，稍后初始化
        self.ptr, self.size = 0, 0
    
    def store(self, s, a, next_s, r, done, mask=None):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.next_s_buf[self.ptr] = next_s
        self.r_buf[self.ptr] = r
        self.done_buf[self.ptr] = done
        if mask is not None:
            # 初始化mask_buf如果还没有初始化
            if self.mask_buf is None:
                self.mask_buf = np.zeros((self.capacity, len(mask)), dtype=np.bool_)
            self.mask_buf[self.ptr] = mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample_batch(self, batch_size):
        sample_idxs = np.random.choice(self.size, batch_size)
        batch = [
            self.s_buf[sample_idxs],
            self.a_buf[sample_idxs],
            self.next_s_buf[sample_idxs],
            self.r_buf[sample_idxs],
            self.done_buf[sample_idxs],
        ]
        # 只有在mask_buf存在时才添加
        if self.mask_buf is not None:
            batch.append(self.mask_buf[sample_idxs])
        else:
            # 创建一个全True的掩码
            mask_shape = (batch_size, self.s_buf.shape[0])  # 假设动作数量
            batch.append(np.ones(mask_shape, dtype=np.bool_))
        
        return [torch.from_numpy(buf).to(self.device) for buf in batch]


class DefenderDQN(nn.Module):
    """防御者DQN网络"""
    
    def __init__(self, input_dim, layers, num_actions):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, layers[0])])
        for l in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[l-1], layers[l]))
        
        # 添加批归一化和dropout
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(layer_size) for layer_size in layers])
        self.dropout = nn.Dropout(0.1)
        
        self.out = nn.Linear(layers[-1], num_actions)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
    
    def forward(self, x, action_mask=None):
        for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            if x.size(0) > 1:  # 只在batch_size > 1时使用批归一化
                x = bn(x)
            x = F.relu(x)
            if i < len(self.layers) - 1:  # 不在最后一层使用dropout
                x = self.dropout(x)
        
        q_values = self.out(x)
        
        # 应用动作掩码
        if action_mask is not None:
            q_values = q_values.masked_fill(~action_mask, float('-inf'))
        
        return q_values
    
    def save_DQN(self, file_path):
        torch.save(self.state_dict(), file_path)
    
    def load_DQN(self, file_path):
        self.load_state_dict(torch.load(file_path))
    
    def get_action(self, x, action_mask=None):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.view(1, -1)
            q_values = self.forward(x, action_mask)
            return q_values.max(1)[1]


class DefenderDQNAgent:
    """防御者DQN智能体"""
    
    def __init__(self,
                 env,
                 seed=None,
                 lr=0.001,
                 training_steps=50000,
                 batch_size=64,
                 replay_size=100000,
                 final_epsilon=0.05,
                 exploration_steps=20000,
                 gamma=0.99,
                 hidden_sizes=[256, 256, 128],
                 target_update_freq=1000,
                 use_double_dqn=True,
                 use_prioritized_replay=False,
                 verbose=True,
                 **kwargs):
        
        self.verbose = verbose
        if self.verbose:
            print(f"\n运行防御者DQN配置:")
            pprint(locals())
        
        # 设置随机种子
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # 环境设置
        self.env = env
        self.num_actions = self.env.action_space.n
        self.obs_dim = self.env.observation_space.shape
        
        # 日志设置
        self.logger = SummaryWriter(log_dir=f"runs/defender_dqn_{seed}")
        
        # 训练相关参数
        self.lr = lr
        self.exploration_steps = exploration_steps
        self.final_epsilon = final_epsilon
        self.epsilon_schedule = np.linspace(1.0, self.final_epsilon, self.exploration_steps)
        self.batch_size = batch_size
        self.discount = gamma
        self.training_steps = training_steps
        self.steps_done = 0
        self.use_double_dqn = use_double_dqn
        
        # 神经网络相关参数
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DefenderDQN(self.obs_dim[0], hidden_sizes, self.num_actions).to(self.device)
        
        if self.verbose:
            print(f"\n使用神经网络运行在设备={self.device}:")
            print(self.dqn)
        
        self.target_dqn = DefenderDQN(self.obs_dim[0], hidden_sizes, self.num_actions).to(self.device)
        self.target_update_freq = target_update_freq
        self._update_target_network()
        
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
        
        # 经验回放设置
        self.replay = DefenderReplayMemory(replay_size, self.obs_dim, self.device)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_lengths = []
    
    def save(self, save_path):
        """保存模型"""
        self.dqn.save_DQN(save_path)
    
    def load(self, load_path):
        """加载模型"""
        self.dqn.load_DQN(load_path)
        self._update_target_network()
    
    def get_epsilon(self):
        """获取当前探索率"""
        if self.steps_done < self.exploration_steps:
            return self.epsilon_schedule[self.steps_done]
        return self.final_epsilon
    
    def get_action(self, obs, action_mask=None, epsilon=None):
        """获取动作（带探索）"""
        if epsilon is None:
            epsilon = self.get_epsilon()
        
        if random.random() > epsilon:
            obs_tensor = torch.from_numpy(obs).float().to(self.device)
            if action_mask is not None:
                mask_tensor = torch.from_numpy(action_mask).bool().to(self.device)
            else:
                mask_tensor = None
            return self.dqn.get_action(obs_tensor, mask_tensor).cpu().item()
        else:
            # 随机选择可用动作
            if action_mask is not None:
                available_actions = np.where(action_mask)[0]
                if len(available_actions) > 0:
                    return np.random.choice(available_actions)
            return random.randint(0, self.num_actions - 1)
    
    def _update_target_network(self):
        """更新目标网络"""
        self.target_dqn.load_state_dict(self.dqn.state_dict())
    
    def optimize(self):
        """优化网络"""
        if self.replay.size < self.batch_size:
            return 0.0, 0.0
        
        batch = self.replay.sample_batch(self.batch_size)
        s_batch, a_batch, next_s_batch, r_batch, d_batch, mask_batch = batch
        
        # 计算当前Q值
        q_vals_raw = self.dqn(s_batch)
        q_vals = q_vals_raw.gather(1, a_batch).squeeze()
        
        # 计算目标Q值
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN
                next_actions = self.dqn(next_s_batch).max(1)[1].unsqueeze(1)
                target_q_vals = self.target_dqn(next_s_batch).gather(1, next_actions).squeeze()
            else:
                # 标准DQN
                target_q_vals = self.target_dqn(next_s_batch).max(1)[0]
            
            target = r_batch + self.discount * (1 - d_batch) * target_q_vals
        
        # 计算损失
        loss = self.loss_fn(q_vals, target)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps_done % self.target_update_freq == 0:
            self._update_target_network()
        
        q_vals_max = q_vals_raw.max(1)[0]
        mean_v = q_vals_max.mean().item()
        return loss.item(), mean_v
    
    def train(self):
        """训练智能体"""
        if self.verbose:
            print("\n开始训练防御者智能体")
        
        num_episodes = 0
        training_steps_remaining = self.training_steps
        
        while self.steps_done < self.training_steps:
            ep_results = self.run_train_episode(training_steps_remaining)
            ep_return, ep_cost, ep_steps, ep_info = ep_results
            num_episodes += 1
            training_steps_remaining -= ep_steps
            
            # 记录统计信息
            self.episode_rewards.append(ep_return)
            self.episode_costs.append(ep_cost)
            self.episode_lengths.append(ep_steps)
            
            # 记录到tensorboard
            self.logger.add_scalar("episode/return", ep_return, self.steps_done)
            self.logger.add_scalar("episode/cost", ep_cost, self.steps_done)
            self.logger.add_scalar("episode/length", ep_steps, self.steps_done)
            self.logger.add_scalar("episode/epsilon", self.get_epsilon(), self.steps_done)
            self.logger.add_scalar("episode/compromised_hosts", 
                                 ep_info.get('compromised_hosts', 0), self.steps_done)
            
            if num_episodes % 10 == 0 and self.verbose:
                print(f"\n回合 {num_episodes}:")
                print(f"\t已完成步数 = {self.steps_done} / {self.training_steps}")
                print(f"\t回合奖励 = {ep_return:.2f}")
                print(f"\t回合成本 = {ep_cost:.2f}")
                print(f"\t被攻破主机 = {ep_info.get('compromised_hosts', 0)}")
                print(f"\t探索率 = {self.get_epsilon():.3f}")
        
        self.logger.close()
        if self.verbose:
            print("训练完成")
            print(f"\n最终统计:")
            print(f"\t总回合数 = {num_episodes}")
            print(f"\t平均奖励 = {np.mean(self.episode_rewards[-100:]):.2f}")
            print(f"\t平均成本 = {np.mean(self.episode_costs[-100:]):.2f}")
    
    def run_train_episode(self, step_limit):
        """运行训练回合"""
        obs = self.env.reset()
        done = False
        
        steps = 0
        episode_return = 0
        episode_cost = 0
        
        while not done and steps < step_limit:
            # 获取动作掩码
            action_mask = self.env.get_action_mask()
            
            # 选择动作
            action = self.get_action(obs, action_mask)
            
            # 执行动作
            next_obs, reward, done, info = self.env.step(action)
            
            # 存储经验
            self.replay.store(obs, action, next_obs, reward, done, action_mask)
            
            # 优化网络
            if self.replay.size >= self.batch_size:
                loss, mean_v = self.optimize()
                self.logger.add_scalar("train/loss", loss, self.steps_done)
                self.logger.add_scalar("train/mean_v", mean_v, self.steps_done)
            
            # 更新状态
            obs = next_obs
            episode_return += reward
            episode_cost += info.get('action_cost', 0)
            steps += 1
            self.steps_done += 1
        
        return episode_return, episode_cost, steps, info
    
    def run_eval_episode(self, env=None, render=False, eval_epsilon=0.0):
        """运行评估回合"""
        if env is None:
            env = self.env
        
        obs = env.reset()
        done = False
        
        steps = 0
        episode_return = 0
        episode_cost = 0
        
        if render:
            print("\n" + "="*60)
            print(f"运行评估，探索率 = {eval_epsilon:.4f}")
            print("="*60)
            env.render()
        
        while not done:
            action_mask = env.get_action_mask()
            action = self.get_action(obs, action_mask, eval_epsilon)
            
            next_obs, reward, done, info = env.step(action)
            
            obs = next_obs
            episode_return += reward
            episode_cost += info.get('action_cost', 0)
            steps += 1
            
            if render:
                print(f"\n步骤 {steps}")
                print(f"动作: {env.action_space.get_action(action).__class__.__name__}")
                print(f"奖励: {reward:.2f}")
                print(f"成本: {info.get('action_cost', 0):.2f}")
                env.render()
                
                if done:
                    print("\n" + "="*60)
                    print("回合结束")
                    print("="*60)
                    evaluation = env.get_total_evaluation()
                    for key, value in evaluation.items():
                        print(f"{key}: {value}")
        
        return episode_return, episode_cost, steps, env.get_total_evaluation()


def create_defender_agent(scenario_name, **kwargs):
    """创建防御者智能体的便捷函数"""
    import nasim
    
    # 创建攻击者环境（用于获取场景信息）
    attacker_env = nasim.make_benchmark(
        scenario_name,
        seed=kwargs.get('seed', 0),
        fully_obs=True,
        flat_actions=True,
        flat_obs=True
    )
    
    # 创建防御者环境
    defender_env = DefenderEnv(attacker_env.scenario, attacker_env)
    
    # 创建防御者智能体
    agent = DefenderDQNAgent(defender_env, **kwargs)
    
    return agent, defender_env


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_name", type=str, help="基准场景名称")
    parser.add_argument("--render_eval", action="store_true", help="渲染最终策略")
    parser.add_argument("--hidden_sizes", type=int, nargs="*", default=[256, 256, 128])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("-t", "--training_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--target_update_freq", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay_size", type=int, default=100000)
    parser.add_argument("--final_epsilon", type=float, default=0.05)
    parser.add_argument("--exploration_steps", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--quiet", action="store_false", dest="verbose")
    
    args = parser.parse_args()
    
    # 创建智能体和环境
    agent, env = create_defender_agent(args.scenario_name, **vars(args))
    
    # 训练
    agent.train()
    
    # 评估
    agent.run_eval_episode(render=args.render_eval)
