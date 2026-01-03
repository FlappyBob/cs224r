#!/usr/bin/env python3
"""
运行超参数实验：测试不同训练步数对BC性能的影响
"""

import os
import subprocess
import sys
import time

# 超参数值列表（训练步数）
TRAIN_STEPS_VALUES = [250, 500, 1000, 2000, 5000]

# 基础命令参数
BASE_CMD = [
    'python', 'cs224r/scripts/run_hw1.py',
    '--expert_policy_file', 'cs224r/policies/experts/Ant.pkl',
    '--env_name', 'Ant-v4',
    '--expert_data', 'cs224r/expert_data/expert_data_Ant-v4.pkl',
    '--eval_batch_size', '5000',
    '--video_log_freq', '-1',
    '--n_iter', '1',
]

def run_experiment(train_steps):
    """运行单个实验"""
    exp_name = f'bc_ant_steps_{train_steps}'
    cmd = BASE_CMD + [
        '--exp_name', exp_name,
        '--num_agent_train_steps_per_iter', str(train_steps),
    ]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: train_steps = {train_steps}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    # 设置环境变量（如果没有显示服务器）
    env = os.environ.copy()
    if 'MUJOCO_GL' not in env:
        env['MUJOCO_GL'] = 'osmesa'  # 或者 'egl'，根据系统选择
    
    try:
        result = subprocess.run(cmd, env=env, check=True, cwd='.')
        print(f"\n✅ Experiment completed: train_steps = {train_steps}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment failed: train_steps = {train_steps}")
        print(f"Error: {e}\n")
        return False

def main():
    """运行所有超参数实验"""
    print("="*60)
    print("Hyperparameter Experiment: Training Steps")
    print("="*60)
    print(f"Testing {len(TRAIN_STEPS_VALUES)} values: {TRAIN_STEPS_VALUES}")
    print("="*60)
    
    results = {}
    for train_steps in TRAIN_STEPS_VALUES:
        success = run_experiment(train_steps)
        results[train_steps] = success
        time.sleep(2)  # 短暂休息，避免资源冲突
    
    print("\n" + "="*60)
    print("Experiment Summary:")
    print("="*60)
    for train_steps, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"  train_steps={train_steps}: {status}")
    print("="*60)
    print("\nNext step: Run extract_results.py to collect results and generate plots")

if __name__ == '__main__':
    main()

