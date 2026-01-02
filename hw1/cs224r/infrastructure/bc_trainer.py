"""
TO EDIT: Defines a trainer which updates a behavior cloning agent

Functions to edit:
    1. collect_training_trajectories (line 184)
    2. train_agent line(222)
    3. do_relabel_with_expert (line 242)

You will need to first implement `sample_n_trajectories` in utils.py
before running this file, as it is a dependency of this file
"""

from collections import OrderedDict

import pickle
import time
import torch
import gym

import numpy as np

from cs224r.infrastructure import pytorch_util as ptu
from cs224r.infrastructure.logger import Logger
from cs224r.infrastructure import utils

# The number of rollouts to save to videos in PyTorch
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # Constant for video length, we overwrite this in the code below


class BCTrainer:
    """
    A class which defines the training algorithm for the agent. Handles
    sampling data, updating the agent, and logging the results.

    ...

    Attributes
    ----------
    agent : BCAgent
        The agent we want to train

    Methods
    -------
    run_training_loop:
        Main training loop for the agent
    collect_training_trajectories:
        Collect data to be used for training
    train_agent
        Samples a batch and updates the agent
    do_relabel_with_expert
        Relabels trajectories with new actions for DAgger
    """

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get parameters, create logger, and create the TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])

        # Set random seeds
        seed = self.params['seed']
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(
            use_gpu=not self.params['no_gpu'],
            gpu_id=self.params['which_gpu']
        )

        # Set logger attributes
        self.log_video = True
        self.log_metrics = True

        #############
        ## ENV
        #############

        # Make the gym environment
        if self.params['video_log_freq'] == -1:
            self.params['env_kwargs']['render_mode'] = None
        self.env = gym.make(self.params['env_name'], **self.params['env_kwargs'])
        self.env.reset(seed=seed)

        # Set the maximum length for episodes and videos
        self.params['ep_len'] = self.params['ep_len'] or self.env.spec.max_episode_steps
        MAX_VIDEO_LEN = self.params['ep_len']

        # Set whether the environment is continuous or discrete. NOTE: All agents
        # in this assignment are continuous, so we have an assert to break the training
        # if for some reason discrete is set.
        assert not isinstance(self.env.action_space, gym.spaces.Discrete)

        # Set observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim

        # Define the simulation timestep, which will be used for video saving
        if 'model' in dir(self.env):
            self.fps = 1/self.env.model.opt.timestep
        else:
            self.fps = self.env.env.metadata['render_fps']

        #############
        ## AGENT
        #############

        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # Initialize variables at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # Decide if videos should be rendered/logged at this iteration
            if itr % self.params['video_log_freq'] == 0 and self.params['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # Decide if metrics should be logged
            if itr % self.params['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # Collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                initial_expertdata,
                collect_policy
            )  # HW1: implement this function below
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # Relabel the collected observations with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                # HW1: implement this function below
                paths = self.do_relabel_with_expert(expert_policy, paths)

            # Add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # Train agent (using sampled data from replay buffer)
            # HW1: implement this function below
            training_logs = self.train_agent()

            # Log and save videos and metrics
            if self.log_video or self.log_metrics:

                # Perform logging
                print('\nBeginning logging procedure...')
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)

                if self.params['save_params']:
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(self.params['logdir'], itr))

    ####################################
    ####################################

    def collect_training_trajectories(
            self,
            itr,
            load_initial_expertdata,
            collect_policy
    ):
        """
        :param itr:
        :param load_initial_expertdata: path to expert data pkl file
        :param collect_policy: the current policy using which we collect data
        :return:
            paths: a list trajectories                                                                                                  
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
        # HINT1: On the first iteration, do you need to collect training trajectories? You might
        # want to handle loading from expert data, and if the data doesn't exist, collect an appropriate
        # number of transitions.
        # HINT2: Loading from expert transitions can be done using pickle.load()
        # HINT3: To collect data, you might want to use pre-existing sample_trajectories code from utils
        # HINT4: You want each of these collected rollouts to be of length self.params['ep_len']

        print("\nCollecting data to be used for training...")
<<<<<<< Updated upstream
        if itr == 0: 
            with open(load_initial_expertdata, 'rb') as f:
                paths = pickle.load(f)
            envsteps_this_batch = sum(utils.get_pathlength(path) for path in paths)
        else:
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.env,
                collect_policy, 
                self.params['batch_size'], 
                self.params['ep_len']
            )
        
=======
        
        # 判断是第一次迭代还是后续迭代
        if itr == 0:
            # 第一次迭代：加载专家数据
            # Behavior Cloning 使用预收集的专家数据作为训练集
            # 这些数据包含了专家策略在环境中的轨迹（观测-动作对）
            print(f"Loading expert data from {load_initial_expertdata}")
            with open(load_initial_expertdata, 'rb') as f:
                paths = pickle.load(f)
            
            # 计算总的环境步数（用于日志记录）
            # 每个 path 是一个字典，包含 "reward" 数组，其长度就是该轨迹的步数
            envsteps_this_batch = sum([len(path["reward"]) for path in paths])
        else:
            # 后续迭代（主要用于 DAgger）：
            # 使用当前训练的策略（collect_policy）在环境中收集新的轨迹
            # 这样可以收集到当前策略可能犯错的区域的数据
            # batch_size 指定要收集的总步数，ep_len 是每个轨迹的最大长度
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.env, 
                collect_policy, 
                self.params['batch_size'],  # 要收集的总步数
                self.params['ep_len']       # 每个轨迹的最大长度
            )
>>>>>>> Stashed changes

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:            
            ## TODO look in utils and implement sample_n_trajectories
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = utils.sample_n_trajectories(self.env,
                collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        """
        Samples a batch of trajectories and updates the agent with the batch
        """
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        
        # 每个迭代中，我们进行多次梯度更新
        # num_agent_train_steps_per_iter 指定了每个迭代要更新多少次
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

<<<<<<< Updated upstream
            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = \
                self.agent.sample(self.params['train_batch_size'])

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
=======
            # 从经验回放缓冲区中随机采样一个批次的数据
            # 这是 Behavior Cloning 的核心：从专家数据中学习
            # agent.sample() 返回：
            #   - ob_batch: 观测批次 (batch_size x obs_dim)
            #   - ac_batch: 动作批次 (batch_size x ac_dim) - 这是专家动作，作为监督学习的标签
            #   - re_batch: 奖励批次（BC中不使用）
            #   - next_ob_batch: 下一个观测批次（BC中不使用）
            #   - terminal_batch: 终止标志批次（BC中不使用）
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = \
                self.agent.sample(self.params['train_batch_size'])

            # 使用采样的数据训练 Agent
            # Behavior Cloning 是监督学习：
            #   输入：观测 (ob_batch)
            #   输出：预测的动作
            #   标签：专家的真实动作 (ac_batch)
            #   损失：预测动作与专家动作之间的差异（通常是 MSE）
            # agent.train() 会调用策略网络的 update() 方法，执行反向传播和参数更新
>>>>>>> Stashed changes
            train_log = self.agent.train(ob_batch, ac_batch)
            all_logs.append(train_log)
        
        return all_logs

    def do_relabel_with_expert(self, expert_policy, paths):
        """
        Relabels collected trajectories with an expert policy

        :param expert_policy: the policy we want to relabel the paths with
        :param paths: paths to relabel
        """
        expert_policy.to(ptu.device)
        print("\nRelabelling collected observations with labels from an expert policy...")

<<<<<<< Updated upstream
        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        
        # Iterate through each path
        for path in paths:
            # Get observations from the path
            observations = path["observation"]  # shape: (path_length, obs_dim)
            
            # Get expert actions for all observations in this path
            expert_actions = []
            for obs in observations:
                # Query expert policy for action given this observation
                expert_action = expert_policy.get_action(obs)
                expert_actions.append(expert_action)
            
            # Replace the actions in the path with expert actions
            path["action"] = np.array(expert_actions, dtype=np.float32)
=======
        # DAgger 算法的核心步骤：重新标注
        # 
        # 问题背景：
        #   - 当前策略收集的轨迹中，动作可能不是最优的
        #   - 如果直接用这些"错误"的动作训练，会强化错误行为
        # 
        # DAgger 的解决方案：
        #   1. 用当前策略收集轨迹（获得真实的观测分布）
        #   2. 用专家策略为这些观测生成正确的动作标签
        #   3. 用重新标注的数据训练，这样既学到了真实分布，又学到了正确动作
        #
        # 这样做的优势：
        #   - 避免了分布偏移（distribution shift）问题
        #   - 训练数据来自当前策略的分布，但标签来自专家
        #   - 可以逐步纠正策略的错误
        
        # 遍历所有收集的轨迹
        for path in paths:
            # 获取该轨迹中的所有观测
            observations = path["observation"]  # shape: (trajectory_length, obs_dim)
            
            # 用专家策略为每个观测生成动作
            # expert_policy.get_action() 可以处理单个观测或批量观测
            # 返回的动作 shape: (trajectory_length, ac_dim)
            expert_actions = expert_policy.get_action(observations)
            
            # 替换原来的动作（当前策略的动作）为专家动作
            # 这样我们就有了：当前策略的观测分布 + 专家的正确动作
            path["action"] = expert_actions
>>>>>>> Stashed changes
        
        return paths

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):
        """
        Logs training trajectories and evals the provided policy to log
        evaluation trajectories and videos

        :param itr:
        :param paths: paths collected during training that we want to log
        :param eval_policy: policy to generate eval logs and videos
        :param train_video_paths: videos generated during training
        :param training_logs: additional logs generated during training
        """

        # Collect evaluation trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
            self.env, eval_policy, self.params['eval_batch_size'],
            self.params['ep_len'])


        eval_video_paths = None
        # Save evaluation rollouts as videos in tensorboard event file
        if self.log_video:
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(self.env,
                eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        # Save training and evaluation videos
        print('\nSaving rollouts as videos...')
        if train_video_paths is not None:
            self.logger.log_paths_as_videos(train_video_paths, itr,
                fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                video_title='train_rollouts')
        if eval_video_paths is not None:
            self.logger.log_paths_as_videos(eval_video_paths, itr,
                fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                video_title='eval_rollouts')

        # Save evaluation metrics
        if self.log_metrics:
            # Get the returns and episode lengths of all paths, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # Define logged metrics
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now from additional training logs
            logs.update(last_log)


            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # Perform the logging with tensorboard
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
