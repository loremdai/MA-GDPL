# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from utils import state_vectorize, state_vectorize_user, to_device
from evaluator import MultiWozEvaluator
from estimator import RewardEstimator
from rlmodule import MultiDiscretePolicy, HybridValue, Memory, Transition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 与环境交互，产生模拟经验。
def sampler(pid, queue, evt, env, policy, batchsz):
    """
    This is a sampler function, and it will be called by multiprocess.Process to sample data from environment by multiple
    processes.
    :param pid: process id
    :param queue: multiprocessing.Queue, to collect sampled data
    :param evt: multiprocessing.Event, to keep the process alive
    :param env: environment instance
    :param policy: policy network, to generate action from current policy
    :param batchsz: total sampled items
    :return:
    """
    buff = Memory()

    # we need to sample batchsz of (state, action, next_state, reward, mask)
    # each trajectory contains `trajectory_len` num of items, so we only need to sample
    # `batchsz//trajectory_len` num of trajectory totally
    # the final sampled number may be larger than batchsz.

    sampled_num = 0  # 累计的对话轮数
    sampled_traj_num = 0  # 对话session数
    traj_len = 40  # 最大对话轮数 L_max=40
    real_traj_len = 0  # 某个session中的对话轮数 (<= traj_len)

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):  # 若对话轮数>40则不记录在内

            # [s_dim] => [a_dim]
            s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db))
            a = policy.select_action(s_vec.to(device=DEVICE)).cpu()

            # interact with env
            next_s, done = env.step(s, a)

            # a flag indicates ending or not (mask = signal T, T=0=done, T=1=continue)
            mask = 0 if done else 1

            # get reward compared to demostrations
            next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db))

            # save to queue
            buff.push(s_vec.numpy(), a.numpy(), mask, next_s_vec.numpy())

            # update per step
            s = next_s
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1  # 对话session数+1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()

class Policy(object):
    def __init__(self, env_cls, args, manager, cfg, process_num, character, pre=False, pre_irl=False, infer=False):
        """
        :param env_cls: env class or function, not instance, as we need to create several instance in class.
        :param args:
        :param manager:
        :param cfg:
        :param process_num: process number
        :param character: user or system
        :param pre: set to pretrain mode
        :param infer: set to test mode
        """

        self.process_num = process_num
        self.character = character

        # 为每个进程初始化环境。
        self.env_list = []
        for _ in range(process_num):
            self.env_list.append(env_cls(args.data_dir, cfg))

        # 实例化策略和混合价值网络。
        self.policy = MultiDiscretePolicy(cfg, character).to(device=DEVICE)

        # 若为预训练模式：
        if pre:
            self.print_per_batch = args.print_per_batch
            from dbquery import DBQuery
            db = DBQuery(args.data_dir, cfg)
            self.data_train = manager.create_dataset_policy('train', args.batchsz, cfg, db, character)
            self.data_valid = manager.create_dataset_policy('valid', args.batchsz, cfg, db, character)
            self.data_test = manager.create_dataset_policy('test', args.batchsz, cfg, db, character)
            if character == 'sys':
                pos_weight = args.policy_weight_sys * torch.ones([cfg.a_dim]).to(device=DEVICE)
            elif character == 'usr':
                pos_weight = args.policy_weight_usr * torch.ones([cfg.a_dim_usr]).to(device=DEVICE)
            else:
                raise Exception('Unknown character')
            self.multi_entropy_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            # estimator.py已包含关于预训练模式的处理以及优化器的配置。
            self.rewarder = RewardEstimator(args, manager, cfg, character, pretrain=pre_irl, inference=infer)
            self.evaluator = MultiWozEvaluator(args.data_dir)

        # 超参数设置：
        self.save_dir = args.save_dir + '/' + character if pre else args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.gamma = args.gamma
        self.process_num = process_num

        # 优化器设置：
        self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=args.lr_policy)
        self.policy.eval()

        self.writer = SummaryWriter()

    """
    预训练系统/用户智能体模块
    """
    # 计算损失，该函数只在预训练模块中被调用
    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy(s)
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a
    # 预训练智能体
    def imitating(self, epoch):
        """
        pretrain the policy by simple imitation learning (behavioral cloning)
        """
        self.policy.train()
        a_loss = 0.
        for i, data in enumerate(self.data_train):
            self.policy_optim.zero_grad()
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            loss_a.backward()
            self.policy_optim.step()
            
            if (i+1) % self.print_per_batch == 0:
                a_loss /= self.print_per_batch
                logging.debug('<<dialog policy {}>> epoch {}, iter {}, loss_a:{}'.format(self.character, epoch, i, a_loss))
                a_loss = 0.
        
        if (epoch+1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)
        self.policy.eval()
    # 测试智能体
    def imit_test(self, epoch, best):
        """
        provide an unbiased evaluation of the policy fit on the training dataset
        """        
        a_loss = 0.
        for i, data in enumerate(self.data_valid):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_valid)
        logging.debug('<<dialog policy {}>> validation, epoch {}, loss_a:{}'.format(self.character, epoch, a_loss))
        if a_loss < best:
            logging.info('<<dialog policy {}>> best model saved'.format(self.character))
            best = a_loss
            self.save(self.save_dir, 'best', True)
            
        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy {}>> test, epoch {}, loss_a:{}'.format(self.character, epoch, a_loss))
        self.writer.add_scalar('pretrain/dialogue_policy_{}/test'.format(self.character), a_loss, epoch)
        return best


    """
    预训练RewardEstimator模块
    """
    # 预训练RE（逆强化学习）
    def train_irl(self, epoch, batchsz):
        batch = self.sample(batchsz)
        self.rewarder.train_irl(batch, epoch)
    # 测试RE
    def test_irl(self, epoch, batchsz, best):
        batch = self.sample(batchsz)
        best = self.rewarder.test_irl(batch, epoch, best)
        return best


    """
    辅助函数模块
    """
    # 采样，调用代码顶端的def sampler()
    def sample(self, batchsz):
        """
        Given batchsz number of task, the batchsz will be splited equally to each processes
        and when processes return, it merge all data and return
        :param batchsz:
        :return: batch
        """

        # batchsz will be splitted into each process,
        # final batchsz maybe larger than batchsz parameters
        process_batchsz = np.ceil(batchsz / self.process_num).astype(np.int32)
        # buffer to save all data
        queue = mp.Queue()

        # start processes for pid in range(1, processnum)
        # if processnum = 1, this part will be ignored.
        # when save tensor in Queue, the process should keep alive till Queue.get(),
        # please refer to : https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847
        # however still some problem on CUDA tensors on multiprocessing queue,
        # please refer to : https://discuss.pytorch.org/t/cuda-tensors-on-multiprocessing-queue/28626
        # so just transform tensors into numpy, then put them into queue.
        evt = mp.Event()
        processes = []
        for i in range(self.process_num):
            process_args = (i, queue, evt, self.env_list[i], self.policy, process_batchsz)
            processes.append(mp.Process(target=sampler, args=process_args))
        for p in processes:
            # set the process as daemon, and it will be killed once the main process is stoped.
            p.daemon = True
            p.start()

        # we need to get the first Memory object and then merge others Memory use its append function.
        pid0, buff0 = queue.get()
        for _ in range(1, self.process_num):
            pid, buff_ = queue.get()
            buff0.append(buff_) # merge current Memory into buff0
        evt.set()

        # now buff saves all the sampled data
        buff = buff0

        return buff.get_batch()
    # 保存
    def save(self, directory, epoch, rl_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 此处修改，追加保存RE
        if not rl_only:
            self.rewarder.save_irl(directory, epoch)    # 调用save_irl函数保存
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_pol.mdl')

        logging.info('<<dialog policy {}>> epoch {}: saved network to mdl'.format(self.character, epoch))
    # 载入
    def load(self, filename):
        # 此处修改，追加保存RE
        self.rewarder.load_irl(filename)
        policy_mdl = filename + '_pol.mdl'
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            logging.info('<<dialog policy {}>> loaded checkpoint from file: {}'.format(self.character, policy_mdl))