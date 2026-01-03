# run_hw1.py 参数详解

本文档详细解释 `run_hw1.py` 中所有命令行参数的含义、用途和推荐值。

---

## 📋 参数分类

### 1. 必需参数（Required Parameters）

#### `--expert_policy_file` / `-epf`
```python
parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)
```

**含义**：专家策略文件的路径

**作用**：
- 指定预训练的专家策略文件（通常是 `.pkl` 文件）
- 在 Behavior Cloning 中，这个文件用于加载专家策略，用于：
  - 评估：比较学习到的策略与专家的性能
  - DAgger：在重新标注时使用专家策略生成正确动作

**使用示例**：
```bash
--expert_policy_file ./cs224r/policies/experts/Ant.pkl
```

**代码中的使用**：
```python
loaded_expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])
```

---

#### `--expert_data` / `-ed`
```python
parser.add_argument('--expert_data', '-ed', type=str, required=True)
```

**含义**：专家数据文件的路径

**作用**：
- 指定预收集的专家轨迹数据（`.pkl` 文件）
- 包含专家在环境中执行时的观测-动作对
- 这是 Behavior Cloning 的训练数据来源

**数据结构**：
- 文件包含一个列表，每个元素是一个轨迹（path）
- 每个轨迹是字典，包含：
  - `observation`: 观测序列
  - `action`: 动作序列
  - `reward`: 奖励序列
  - 等

**使用示例**：
```bash
--expert_data ./cs224r/expert_data/expert_data_Ant-v4.pkl
```

**代码中的使用**：
```python
trainer.run_training_loop(
    initial_expertdata=params['expert_data'],  # 第一次迭代时加载
    ...
)
```

---

#### `--env_name` / `-env`
```python
parser.add_argument('--env_name', '-env', type=str,
    help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)
```

**含义**：要使用的强化学习环境名称

**可选值**：
- `Ant-v4`：蚂蚁环境（6条腿）
- `Walker2d-v4`：两足行走机器人
- `HalfCheetah-v4`：半猎豹（快速移动）
- `Hopper-v4`：单足跳跃机器人

**作用**：
- 指定训练和评估的环境
- 不同环境有不同的观测空间和动作空间

**使用示例**：
```bash
--env_name Ant-v4
```

**代码中的使用**：
```python
self.env = gym.make(self.params['env_name'], **self.params['env_kwargs'])
```

---

#### `--exp_name` / `-exp`
```python
parser.add_argument('--exp_name', '-exp', type=str,
    default='pick an experiment name', required=True)
```

**含义**：实验名称

**作用**：
- 用于创建日志目录，区分不同的实验
- 日志目录格式：`q1_<exp_name>_<env_name>_<timestamp>`
- 例如：`q1_bc_ant_Ant-v4_01-01-2026_10-00-01`

**使用示例**：
```bash
--exp_name my_bc_experiment
```

**代码中的使用**：
```python
logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + \
    time.strftime("%d-%m-%Y_%H-%M-%S")
```

---

### 2. 训练控制参数（Training Control）

#### `--do_dagger`
```python
parser.add_argument('--do_dagger', action='store_true')
```

**含义**：是否启用 DAgger 算法

**作用**：
- 如果提供此参数：使用 DAgger 算法（需要 `n_iter > 1`）
- 如果不提供：使用标准的 Behavior Cloning（`n_iter = 1`）

**DAgger vs BC**：
- **BC**：只用专家数据训练一次
- **DAgger**：迭代训练，每次用当前策略收集数据，用专家重新标注

**使用示例**：
```bash
# 启用 DAgger
--do_dagger

# 不启用（标准 BC）
# 不提供此参数即可
```

**代码中的使用**：
```python
if args.do_dagger:
    logdir_prefix = 'q2_'
    assert args.n_iter > 1  # DAgger 需要多次迭代
else:
    logdir_prefix = 'q1_'
    assert args.n_iter == 1  # BC 只需要一次迭代
```

---

#### `--n_iter` / `-n`
```python
parser.add_argument('--n_iter', '-n', type=int, default=1)
```

**含义**：训练迭代次数

**作用**：
- 控制训练循环执行多少次
- **BC**：通常为 1（只用专家数据训练一次）
- **DAgger**：需要 > 1（多次迭代收集新数据并训练）

**推荐值**：
- BC: `1`
- DAgger: `10-20`（根据任务复杂度调整）

**使用示例**：
```bash
--n_iter 1      # BC
--n_iter 10     # DAgger
```

**代码中的使用**：
```python
for itr in range(n_iter):  # 训练循环
    # 收集数据、训练、评估...
```

---

#### `--num_agent_train_steps_per_iter`
```python
parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=1000)
```

**含义**：每个迭代中，Agent 训练的梯度步数

**作用**：
- 控制每个迭代中执行多少次梯度更新
- 每次梯度更新会从经验回放缓冲区采样一个批次的数据

**理解**：
- 假设 `n_iter=10`，`num_agent_train_steps_per_iter=1000`
- 总共会执行 `10 × 1000 = 10,000` 次梯度更新

**推荐值**：
- 小任务：`500-1000`
- 大任务：`1000-2000`

**使用示例**：
```bash
--num_agent_train_steps_per_iter 1000
```

**代码中的使用**：
```python
for train_step in range(self.params['num_agent_train_steps_per_iter']):
    ob_batch, ac_batch, ... = self.agent.sample(...)
    train_log = self.agent.train(ob_batch, ac_batch)
```

---

### 3. 数据收集参数（Data Collection）

#### `--batch_size`
```python
parser.add_argument('--batch_size', type=int, default=1000)
```

**含义**：每次迭代收集的训练数据步数（环境交互步数）

**作用**：
- 控制每次迭代从环境中收集多少步数据
- 用于 DAgger：每次迭代用当前策略收集新数据
- 注意：这是**环境步数**，不是批次大小

**推荐值**：
- 开发/调试：`1000`（快速）
- 最终结果：`≥ 10,000`（更稳定，减少方差）

**使用示例**：
```bash
--batch_size 10000
```

**代码中的使用**：
```python
paths, envsteps_this_batch = utils.sample_trajectories(
    self.env, 
    collect_policy, 
    self.params['batch_size'],  # 收集这么多步
    self.params['ep_len']
)
```

---

#### `--eval_batch_size`
```python
parser.add_argument('--eval_batch_size', type=int, default=1000)
```

**含义**：评估时收集的数据步数

**作用**：
- 控制评估时收集多少步数据来计算性能指标
- 用于计算平均回报、标准差等统计量

**推荐值**：
- `1000-5000`（足够计算可靠的统计量）

**使用示例**：
```bash
--eval_batch_size 2000
```

**代码中的使用**：
```python
eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
    self.env, eval_policy, self.params['eval_batch_size'],
    self.params['ep_len']
)
```

---

#### `--train_batch_size`
```python
parser.add_argument('--train_batch_size', type=int, default=100)
```

**含义**：每次梯度更新时采样的数据点数量

**作用**：
- 控制每次梯度更新从经验回放缓冲区采样多少条数据
- 这是真正的"批次大小"（batch size）

**理解**：
- `batch_size=10000`：收集 10,000 步数据到缓冲区
- `train_batch_size=100`：每次梯度更新从缓冲区采样 100 条数据
- `num_agent_train_steps_per_iter=1000`：执行 1000 次梯度更新
- 总共使用：`100 × 1000 = 100,000` 条数据（可能重复采样）

**推荐值**：
- `32-256`（根据任务复杂度调整）

**使用示例**：
```bash
--train_batch_size 128
```

**代码中的使用**：
```python
ob_batch, ac_batch, ... = self.agent.sample(self.params['train_batch_size'])
```

---

#### `--ep_len`
```python
parser.add_argument('--ep_len', type=int)
```

**含义**：每个轨迹（episode）的最大长度

**作用**：
- 限制单个轨迹的最大步数
- 如果轨迹提前结束（done=True），则实际长度可能更短
- 如果不提供，使用环境的默认最大长度

**推荐值**：
- 通常使用环境默认值（不提供此参数）
- 如果需要限制：`200-1000`（根据环境调整）

**使用示例**：
```bash
--ep_len 500
```

**代码中的使用**：
```python
self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
# 如果不提供，使用环境的默认值
```

---

### 4. 网络架构参数（Network Architecture）

#### `--n_layers`
```python
parser.add_argument('--n_layers', type=int, default=2)
```

**含义**：策略网络的隐藏层数量

**作用**：
- 控制策略网络的深度
- 不包括输入层和输出层，只计算隐藏层

**网络结构**：
- `n_layers=2`：输入层 → 隐藏层1 → 隐藏层2 → 输出层
- 总共 4 层（2 个隐藏层 + 输入 + 输出）

**推荐值**：
- 简单任务：`2`
- 复杂任务：`3-4`

**使用示例**：
```bash
--n_layers 3
```

**代码中的使用**：
```python
self.actor = MLPPolicySL(
    ...
    n_layers=self.agent_params['n_layers'],
    ...
)
```

---

#### `--size`
```python
parser.add_argument('--size', type=int, default=64)
```

**含义**：每个隐藏层的神经元数量（宽度）

**作用**：
- 控制策略网络的宽度
- 所有隐藏层使用相同的宽度

**推荐值**：
- 小任务：`32-64`
- 中等任务：`64-128`
- 大任务：`128-256`

**使用示例**：
```bash
--size 128
```

**代码中的使用**：
```python
self.actor = MLPPolicySL(
    ...
    size=self.agent_params['size'],
    ...
)
```

---

### 5. 优化器参数（Optimizer）

#### `--learning_rate` / `-lr`
```python
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
```

**含义**：学习率

**作用**：
- 控制梯度更新的步长
- 太大：训练不稳定，可能发散
- 太小：训练慢，可能陷入局部最优

**推荐值**：
- 监督学习（BC）：`1e-3` 到 `1e-2`（0.001 到 0.01）
- 默认值 `5e-3`（0.005）是一个不错的起点

**使用示例**：
```bash
--learning_rate 0.001
# 或
-lr 1e-3
```

**代码中的使用**：
```python
self.actor = MLPPolicySL(
    ...
    learning_rate=self.agent_params['learning_rate'],
    ...
)
```

---

### 6. 日志参数（Logging）

#### `--video_log_freq`
```python
parser.add_argument('--video_log_freq', type=int, default=5)
```

**含义**：每隔多少次迭代记录一次视频

**作用**：
- 控制视频记录的频率
- 视频用于可视化策略的行为
- 设置为 `-1` 可以禁用视频记录

**推荐值**：
- 开发阶段：`5-10`（更频繁，方便观察）
- 最终训练：`10-20`（减少开销）
- 无显示环境：`-1`（禁用）

**使用示例**：
```bash
--video_log_freq 10
--video_log_freq -1  # 禁用视频
```

**代码中的使用**：
```python
if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
    self.log_video = True
```

---

#### `--scalar_log_freq`
```python
parser.add_argument('--scalar_log_freq', type=int, default=1)
```

**含义**：每隔多少次迭代记录一次标量指标

**作用**：
- 控制指标记录的频率
- 指标包括：平均回报、损失、训练步数等

**推荐值**：
- 通常设为 `1`（每次都记录）

**使用示例**：
```bash
--scalar_log_freq 1
```

**代码中的使用**：
```python
if itr % self.params['scalar_log_freq'] == 0:
    self.log_metrics = True
```

---

#### `--save_params`
```python
parser.add_argument('--save_params', action='store_true')
```

**含义**：是否保存模型参数

**作用**：
- 如果提供此参数，会在每次记录日志时保存模型
- 保存的模型可以用于后续评估或继续训练

**使用示例**：
```bash
--save_params
```

**代码中的使用**：
```python
if self.params['save_params']:
    self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))
```

---

### 7. 硬件参数（Hardware）

#### `--no_gpu` / `-ngpu`
```python
parser.add_argument('--no_gpu', '-ngpu', action='store_true')
```

**含义**：是否禁用 GPU

**作用**：
- 如果提供此参数，强制使用 CPU
- 否则尝试使用 GPU（如果可用）

**使用场景**：
- 调试时可能想用 CPU（更慢但更稳定）
- 没有 GPU 的机器

**使用示例**：
```bash
--no_gpu
```

**代码中的使用**：
```python
ptu.init_gpu(
    use_gpu=not self.params['no_gpu'],  # 如果 no_gpu=True，则 use_gpu=False
    gpu_id=self.params['which_gpu']
)
```

---

#### `--which_gpu`
```python
parser.add_argument('--which_gpu', type=int, default=0)
```

**含义**：使用哪个 GPU（在多 GPU 系统中）

**作用**：
- 在多 GPU 系统中指定使用哪个 GPU
- GPU 编号从 0 开始

**使用示例**：
```bash
--which_gpu 0  # 使用第一个 GPU
--which_gpu 1  # 使用第二个 GPU
```

**代码中的使用**：
```python
ptu.init_gpu(
    use_gpu=not self.params['no_gpu'],
    gpu_id=self.params['which_gpu']
)
```

---

### 8. 其他参数（Miscellaneous）

#### `--max_replay_buffer_size`
```python
parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
```

**含义**：经验回放缓冲区的最大容量

**作用**：
- 限制缓冲区能存储的最大数据量
- 超过容量时，旧数据会被新数据覆盖（FIFO）

**推荐值**：
- 小任务：`100,000 - 1,000,000`
- 大任务：`1,000,000 - 10,000,000`

**使用示例**：
```bash
--max_replay_buffer_size 2000000
```

**代码中的使用**：
```python
self.replay_buffer = ReplayBuffer(
    self.agent_params['max_replay_buffer_size']
)
```

---

#### `--seed`
```python
parser.add_argument('--seed', type=int, default=1)
```

**含义**：随机种子

**作用**：
- 设置随机数生成器的种子，确保实验可复现
- 相同的种子会产生相同的结果

**推荐值**：
- 开发：`1`
- 实验：使用不同的种子多次运行，取平均

**使用示例**：
```bash
--seed 42
```

**代码中的使用**：
```python
seed = self.params['seed']
np.random.seed(seed)
torch.manual_seed(seed)
self.env.reset(seed=seed)
```

---

## 📊 参数关系图

```
训练流程中的参数关系：

n_iter (迭代次数)
  └─> 每次迭代：
      ├─> batch_size (收集数据步数)
      │   └─> 收集到缓冲区
      │
      ├─> num_agent_train_steps_per_iter (梯度更新次数)
      │   └─> 每次更新：
      │       └─> train_batch_size (采样批次大小)
      │
      └─> eval_batch_size (评估数据步数)
          └─> 计算性能指标
```

---

## 🎯 常用参数组合示例

### Behavior Cloning（标准）
```bash
python run_hw1.py \
    --expert_policy_file ./cs224r/policies/experts/Ant.pkl \
    --expert_data ./cs224r/expert_data/expert_data_Ant-v4.pkl \
    --env_name Ant-v4 \
    --exp_name bc_ant \
    --n_iter 1 \
    --batch_size 10000 \
    --train_batch_size 128 \
    --learning_rate 0.001 \
    --n_layers 2 \
    --size 64
```

### DAgger
```bash
python run_hw1.py \
    --expert_policy_file ./cs224r/policies/experts/Ant.pkl \
    --expert_data ./cs224r/expert_data/expert_data_Ant-v4.pkl \
    --env_name Ant-v4 \
    --exp_name dagger_ant \
    --do_dagger \
    --n_iter 10 \
    --batch_size 10000 \
    --num_agent_train_steps_per_iter 1000 \
    --train_batch_size 128 \
    --learning_rate 0.001 \
    --n_layers 2 \
    --size 64
```

---

## 💡 参数调优建议

1. **先从小参数开始**：`batch_size=1000`, `train_batch_size=32`，快速验证代码
2. **逐步增大**：确认代码正确后，增大 `batch_size` 到 `10000+` 获得更好结果
3. **学习率**：从默认值开始，如果损失不下降，尝试 `1e-4` 或 `1e-2`
4. **网络大小**：简单任务用 `n_layers=2, size=64`，复杂任务用 `n_layers=3, size=128`
5. **多次运行**：使用不同 `seed` 多次运行，取平均结果

---

## 🔍 快速参考表

| 参数 | 类型 | 默认值 | 用途 |
|------|------|--------|------|
| `--expert_policy_file` | str | 必需 | 专家策略文件路径 |
| `--expert_data` | str | 必需 | 专家数据文件路径 |
| `--env_name` | str | 必需 | 环境名称 |
| `--exp_name` | str | 必需 | 实验名称 |
| `--do_dagger` | flag | False | 启用 DAgger |
| `--n_iter` | int | 1 | 迭代次数 |
| `--batch_size` | int | 1000 | 收集数据步数 |
| `--train_batch_size` | int | 100 | 训练批次大小 |
| `--n_layers` | int | 2 | 网络层数 |
| `--size` | int | 64 | 隐藏层宽度 |
| `--learning_rate` | float | 0.005 | 学习率 |
| `--seed` | int | 1 | 随机种子 |

