# Problem 2: 超参数实验使用说明

## 📋 概述

这个实验测试不同训练步数（`num_agent_train_steps_per_iter`）对 BC 性能的影响。

## 🚀 使用方法

### 方法1: 自动运行实验并生成报告（推荐）

```bash
# 步骤1: 运行超参数实验（需要一些时间）
python run_hyperparameter_experiment.py

# 步骤2: 提取结果并生成图表和报告
python extract_results_and_plot.py
```

### 方法2: 手动输入数据（如果已有实验结果）

1. 编辑 `create_plot_from_data.py`，修改 `RESULTS` 字典为你的实验结果
2. 运行：
```bash
python create_plot_from_data.py
```
3. 根据生成的数据更新 `problem2_report_template.md`

### 方法3: 使用 TensorBoard 手动查看结果

```bash
# 启动 TensorBoard
tensorboard --logdir data

# 在浏览器中查看 Eval_AverageReturn 和 Eval_StdReturn
# 手动记录不同训练步数的结果
```

## 📊 实验结果格式

实验结果应该是这样的格式：

```python
RESULTS = {
    250: (平均回报, 标准差),
    500: (平均回报, 标准差),
    1000: (平均回报, 标准差),
    2000: (平均回报, 标准差),
    5000: (平均回报, 标准差),
}
```

## 📝 生成的文件

- `hyperparameter_experiment.png` - 实验图表
- `problem2_report.md` - 完整报告（自动生成）
- 或 `problem2_report_template.md` - 报告模板（手动填写）

## ⚙️ 实验参数

- **环境**: Ant-v4
- **超参数值**: [250, 500, 1000, 2000, 5000]
- **其他参数保持不变**:
  - Network: 2 layers, 64 hidden units
  - Learning rate: 0.005
  - Eval batch size: 5000

## 📌 注意事项

1. 实验可能需要较长时间（每个实验约5-10分钟）
2. 确保有足够的磁盘空间
3. 如果遇到渲染错误，脚本会自动设置 `MUJOCO_GL=osmesa`

