#!/usr/bin/env python3
"""
从手动输入的数据创建超参数实验图
如果无法自动提取TensorBoard数据，可以使用这个脚本手动输入数据
"""

import matplotlib.pyplot as plt
import numpy as np

# 手动输入数据（格式：训练步数: (平均回报, 标准差)）
# 基于Ant环境的典型表现，训练步数越多性能越好
# 专家性能约为4713.65，5000步时BC达到4644.26（98.5%）
RESULTS = {
    250: (4123.45, 145.23),   # 较少训练步数，性能较低
    500: (4389.12, 128.56),   # 中等训练步数
    1000: (4512.78, 112.34),  # 标准训练步数（默认值）
    2000: (4598.34, 105.67),  # 较多训练步数
    5000: (4644.26, 98.19),   # 大量训练步数，接近专家性能
}

def plot_hyperparameter_experiment(results, save_path='hyperparameter_experiment.png'):
    """
    绘制超参数实验图
    
    Args:
        results: dict of {train_steps: (mean_return, std_return)}
        save_path: 保存路径
    """
    # 排序
    sorted_steps = sorted(results.keys())
    means = [results[s][0] for s in sorted_steps]
    stds = [results[s][1] for s in sorted_steps]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 绘制误差棒图
    ax.errorbar(sorted_steps, means, yerr=stds, 
                marker='o', markersize=8, linewidth=2, 
                capsize=5, capthick=2, elinewidth=1.5,
                label='BC Performance', color='#2E86AB')
    
    # 设置坐标轴
    ax.set_xlabel('Number of Training Steps (num_agent_train_steps_per_iter)', 
                  fontsize=14, fontweight='bold')
    ax.set_ylabel('Average Return', fontsize=14, fontweight='bold')
    ax.set_title('BC Agent Performance vs. Training Steps (Ant-v4)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='lower right')
    
    # 设置x轴为对数刻度（如果步数跨度大）
    if max(sorted_steps) / min(sorted_steps) > 10:
        ax.set_xscale('log')
        ax.set_xlabel('Number of Training Steps (log scale)', 
                     fontsize=14, fontweight='bold')
    
    # 添加数值标签（可选）
    for i, (step, mean, std) in enumerate(zip(sorted_steps, means, stds)):
        ax.annotate(f'{mean:.0f}', 
                   xy=(step, mean), 
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=9,
                   alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    plt.show()

def print_table(results):
    """打印结果表格"""
    print("\n" + "="*60)
    print("Results Table")
    print("="*60)
    print(f"{'Training Steps':<20} {'Average Return':<20} {'Std Return':<20}")
    print("-"*60)
    
    sorted_steps = sorted(results.keys())
    for steps in sorted_steps:
        mean, std = results[steps]
        print(f"{steps:<20} {mean:<20.2f} {std:<20.2f}")
    print("="*60)

if __name__ == '__main__':
    print("="*60)
    print("Creating Hyperparameter Experiment Plot")
    print("="*60)
    
    # 打印表格
    print_table(RESULTS)
    
    # 绘制图表
    print("\nGenerating plot...")
    plot_hyperparameter_experiment(RESULTS)
    
    print("\n" + "="*60)
    print("Done! Update RESULTS dictionary with your actual data.")
    print("="*60)

