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

from utils import state_vectorize, to_device
from evaluator import MultiWozEvaluator
from estimator import RewardEstimator
from rlmodule import MultiDiscretePolicy, HybridValue, Memory, Transition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        # initialize envs for each process
        self.env_list = []
        for _ in range(process_num):
            self.env_list.append(env_cls())

        # 此处新增价值网络
        self.policy = MultiDiscretePolicy(cfg, character).to(device=DEVICE)
        self.vnet = HybridValue(cfg).to(device=DEVICE)

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
            # 此处添加reward estimator
            self.rewarder = RewardEstimator(args, manager, cfg, pretrain=pre_irl, inference=infer)
            self.evaluator = MultiWozEvaluator(args.data_dir)

        # 下面新增若干项（均在imit_value中被调用）
        self.save_dir = args.save_dir + '/' + character if pre else args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.update_round = args.update_round   # Epoch num for inner loop of PPO
        self.policy.eval()
        self.vnet.eval()

        self.tau = args.tau  # GAE广义优势估计超参数lambda（此处用tau）
        self.gamma = args.gamma
        self.policy_optim = optim.RMSprop(self.policy.parameters(), lr=args.lr_policy, weight_decay=args.weight_decay)
        self.vnet_optim = optim.Adam(self.vnet.parameters(), lr=args.lr_vnet)
        self.writer = SummaryWriter()

    """
    预训练（模仿学习-行为克隆）模块
    """
    # 计算损失，该函数只在预训练模块中被调用
    def policy_loop(self, data):
        s, target_a = to_device(data)
        a_weights = self.policy(s)
        
        loss_a = self.multi_entropy_loss(a_weights, target_a)
        return loss_a

    # 预训练智能体（以下2个函数只在main.py预训练区被调用）
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

    # 预训练RewardEstimator
    def imit_value(self, epoch, batchsz, best):
        """
        预训练价值函数，返回最好的value_loss
        """
        self.vnet.train()
        batch = self.sample(batchsz)
        s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        mask = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
        batchsz = s.size(0)

        v = self.vnet(s, 'usr').squeeze(-1).detach()
        log_pi_old_sa = self.policy.get_log_prob(s, a).detach()
        r = self.rewarder.estimate(s, a, next_s, log_pi_old_sa).detach()
        A_sa, v_target = self.est_adv(r, v, mask)

        for i in range(self.update_round):
            perm = torch.randperm(batchsz)
            v_target_shuf, s_shuf = v_target[perm], s[perm]
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            v_target_shuf, s_shuf = torch.chunk(v_target_shuf, optim_chunk_num), torch.chunk(s_shuf, optim_chunk_num)

            value_loss = 0.
            for v_target_b, s_b in zip(v_target_shuf, s_shuf):
                self.vnet_optim.zero_grad()
                v_b = self.vnet(s_b, 'usr').squeeze(-1)
                loss = (v_b - v_target_b).pow(2).mean()
                value_loss += loss.item()
                loss.backward()
                self.vnet_optim.step()

            value_loss /= optim_chunk_num
            logging.debug('<<dialog policy {}>> epoch {}, iteration {}, loss {}'.format("Reward Estimator", epoch, i, value_loss))

        if value_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = value_loss  # 记录该损失为best
            self.save(self.save_dir, 'best', True)  # 保存最佳模型
        # 每隔XX轮保存一次模型
        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)
        self.vnet.eval()  # 关闭训练模式

        return best  # 返回最佳（最小）损失
    def train_irl(self, epoch, batchsz):
        batch = self.sample(batchsz)
        self.rewarder.train_irl(batch, epoch)
    def test_irl(self, epoch, batchsz, best):
        batch = self.sample(batchsz)
        best = self.rewarder.test_irl(batch, epoch, best)
        return best


    """
    计算模块
    """
    # 计算价值函数V和优势函数A
    def est_adv(self, r, v, mask):
        """
        we save a trajectory in continuous space and it reaches the ending of current trajectory when mask=0.
        :param r: reward, Tensor, [b]
        :param v: estimated value, Tensor, [b]
        :param mask: indicates ending for 0 otherwise 1, Tensor, [b]
        :return: A(s, a), V-target(s), both Tensor

        此函数用于估计优势函数。
        """
        batchsz = v.size(0)

        # v_target is worked out by Bellman equation.
        v_target = torch.Tensor(batchsz).to(device=DEVICE)
        delta = torch.Tensor(batchsz).to(device=DEVICE) # TD-error
        A_sa = torch.Tensor(batchsz).to(device=DEVICE) # 优势函数

        # 上一时间步的V和A初始化为0
        prev_v_target = 0
        prev_v = 0
        prev_A_sa = 0
        for t in reversed(range(batchsz)):
            # mask here indicates a end of trajectory
            # this value will be treated as the target value of value network.
            # mask = 0 means the immediate reward is the real V(s) since it's end of trajectory.
            # formula: V(s_t) = r_t + gamma * V(s_t+1)
            v_target[t] = r[t] + self.gamma * prev_v_target * mask[t]

            # please refer to : https://arxiv.org/abs/1506.02438
            # for generalized adavantage estimation
            # formula: delta(s_t) = r_t + gamma * V(s_t+1) - V(s_t)
            delta[t] = r[t] + self.gamma * prev_v * mask[t] - v[t]

            # formula: A(s, a) = delta(s_t) + gamma * lamda * A(s_t+1, a_t+1)
            # here use symbol tau as lambda, but original paper uses symbol lambda.
            # 此处使用广义优势估计作为优势函数。
            A_sa[t] = delta[t] + self.gamma * self.tau * prev_A_sa * mask[t]

            # update previous
            prev_v_target = v_target[t]
            prev_v = v[t]
            prev_A_sa = A_sa[t]

        # normalize A_sa
        A_sa = (A_sa - A_sa.mean()) / A_sa.std()

        return A_sa, v_target


    """
    辅助函数模块
    """
    # 采样
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
    def save(self, directory, epoch, rl_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 此处修改，追加保存value和RE
        if not rl_only:
            self.rewarder.save_irl(directory, epoch)    # 调用save_irl函数保存
        torch.save(self.vnet.state_dict(), directory + '/' + str(epoch) + '_vnet.mdl')
        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_pol.mdl')

        logging.info('<<dialog policy {}>> epoch {}: saved network to mdl'.format(self.character, epoch))
    def load(self, filename):
        self.rewarder.load_irl(filename)
        vnet_mdl = filename + '_vnet.mdl'
        policy_mdl = filename + '_pol.mdl'
        if os.path.exists(vnet_mdl):
            self.vnet.load_state_dict(torch.load(vnet_mdl))
            logging.info('<<dialog policy>> loaded checkpoint from file: {}'.format(vnet_mdl))
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            logging.info('<<dialog policy {}>> loaded checkpoint from file: {}'.format(self.character, policy_mdl))
