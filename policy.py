# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import os
import logging

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from utils import state_vectorize, state_vectorize_user, to_device

from evaluator import MultiWozEvaluator
from rlmodule import MultiDiscretePolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy(object):
    def __init__(self, env_cls, args, manager, cfg, process_num, character, pre=False):
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
            self.env_list.append(env_cls())

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
            self.evaluator = MultiWozEvaluator(args.data_dir)

        # 超参数设置：
        self.save_dir = args.save_dir + '/' + character if pre else args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.gamma = args.gamma

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
            self.save(self.save_dir, epoch)
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
            self.save(self.save_dir, 'best')
            
        a_loss = 0.
        for i, data in enumerate(self.data_test):
            loss_a = self.policy_loop(data)
            a_loss += loss_a.item()
            
        a_loss /= len(self.data_test)
        logging.debug('<<dialog policy {}>> test, epoch {}, loss_a:{}'.format(self.character, epoch, a_loss))
        self.writer.add_scalar('pretrain/dialogue_policy_{}/test'.format(self.character), a_loss, epoch)
        return best


    """
    辅助函数模块
    """
    # 保存
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + '/sys')
            os.makedirs(directory + '/usr')
            os.makedirs(directory + '/vnet')

        torch.save(self.policy.state_dict(), directory + '/' + str(epoch) + '_pol.mdl')
        logging.info('<<dialog policy {}>> epoch {}: saved network to mdl'.format(self.character, epoch))
    # 载入
    def load(self, filename):
        policy_mdl = filename + '_pol.mdl'
        if os.path.exists(policy_mdl):
            self.policy.load_state_dict(torch.load(policy_mdl))
            logging.info('<<dialog policy {}>> loaded checkpoint from file: {}'.format(self.character, policy_mdl))