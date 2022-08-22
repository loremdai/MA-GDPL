import os
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from utils import to_device, reparameterize
from dbquery import DBQuery
from rlmodule import AIRL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardEstimator(object):
    def __init__(self, args, config, manager, character, pretrain=False, inference=False):

        self.character = character

        # 实例化IRL模型
        self.irl = AIRL(config, args.gamma, character=character).to(device=DEVICE)

        # 超参数设定区
        self.step = 0
        self.anneal = args.anneal
        self.optim_batchsz = args.batchsz
        self.weight_cliping_limit = args.clip
        self.save_per_epoch = args.save_per_epoch
        self.save_dir = args.save_dir

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.irl_params = self.irl.parameters()
        self.irl_optim = optim.RMSprop(self.irl_params, lr=args.lr_irl)
        self.irl.eval()

        db = DBQuery(args.data_dir, config)
        # 预训练模式：切分3个数据集 -> 放入迭代器中。
        if pretrain:
            self.print_per_batch = args.print_per_batch
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db, character)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db, character)
            self.data_test = manager.create_dataset_irl('test', args.batchsz, config, db, character)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)
            self.irl_iter_test = iter(self.data_test)
        # 训练模式：切分训练集和验证集 -> 放入迭代器中。
        elif not inference:
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db, character)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db, character)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)

    # 分别计算并返回真实经验、模拟经验的loss
    def irl_loop(self, data_real, data_gen):
        s_real, a_real, next_s_real = to_device(data_real)
        s, a, next_s = data_gen

        # train with real data
        weight_real = self.irl(s_real, a_real, next_s_real)
        loss_real = -weight_real.mean()  # 梯度上升，因此real为负，gen为正。

        # train with generated data
        weight = self.irl(s, a, next_s)
        loss_gen = weight.mean()
        return loss_real, loss_gen

    # 训练模型
    def train_irl(self, batch, epoch):
        self.irl.train()

        if self.character == 'sys':
            input_s = torch.from_numpy(np.stack(batch.state_sys)).to(device=DEVICE)
            input_a = torch.from_numpy(np.stack(batch.action_sys)).to(device=DEVICE)
            input_next_s = torch.from_numpy(np.stack(batch.state_sys_next)).to(device=DEVICE)
        elif self.character == 'usr':
            input_s = torch.from_numpy(np.stack(batch.state_usr)).to(device=DEVICE)
            input_a = torch.from_numpy(np.stack(batch.action_usr)).to(device=DEVICE)
            input_next_s = torch.from_numpy(np.stack(batch.state_usr_next)).to(device=DEVICE)
        else:
            raise NotImplementedError('Unknown character {}'.format(self.character))
        batchsz = input_s.size(0)

        # 将sampler()得到的数据分块
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        # 训练
        real_loss, gen_loss = 0., 0.
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = self.irl_iter.next()  # data为数据集的数据
            except StopIteration:
                self.irl_iter = iter(self.data_train)
                data = self.irl_iter.next()

            self.irl_optim.zero_grad()
            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
            loss = loss_real + loss_gen
            loss.backward()
            self.irl_optim.step()

            for p in self.irl_params:
                p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator {}>> epoch {}, loss_real:{}, loss_gen:{}'.format(
            self.character, epoch, real_loss, gen_loss))
        if (epoch + 1) % self.save_per_epoch == 0:
            self.save_irl(self.save_dir, epoch)
        self.irl.eval()

    # 验证和测试最优模型
    def test_irl(self, batch, epoch, best):
        if self.character == 'sys':
            input_s = torch.from_numpy(np.stack(batch.state_sys)).to(device=DEVICE)
            input_a = torch.from_numpy(np.stack(batch.action_sys)).to(device=DEVICE)
            input_next_s = torch.from_numpy(np.stack(batch.state_sys_next)).to(device=DEVICE)
        elif self.character == 'usr':
            input_s = torch.from_numpy(np.stack(batch.state_usr)).to(device=DEVICE)
            input_a = torch.from_numpy(np.stack(batch.action_usr)).to(device=DEVICE)
            input_next_s = torch.from_numpy(np.stack(batch.state_usr_next)).to(device=DEVICE)
        else:
            raise NotImplementedError('Unknown character {}'.format(self.character))
        batchsz = input_s.size(0)

        # 将sampler()得到的数据分块
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        # 在验证集上找出最优模型
        real_loss, gen_loss = 0., 0.
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = self.irl_iter_valid.next()
            except StopIteration:
                self.irl_iter_valid = iter(self.data_valid)
                data = self.irl_iter_valid.next()

            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator {}>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
            self.character, epoch, real_loss, gen_loss))
        loss = real_loss + gen_loss
        if loss < best:
            logging.info('<<reward estimator>> best model saved')
            best = loss
            self.save_irl(self.save_dir, 'best')

        # 在测试集上测试最优模型
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = self.irl_iter_test.next()
            except StopIteration:
                self.irl_iter_test = iter(self.data_test)
                data = self.irl_iter_test.next()

            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()
        real_loss /= turns
        gen_loss /= turns
        logging.debug('<<reward estimator {}>> test, epoch {}, loss_real:{}, loss_gen:{}'.format(
            self.character, epoch, real_loss, gen_loss))
        return best

    # 更新模型
    def update_irl(self, inputs, batchsz, epoch, best=None):
        """
        train the reward estimator (together with encoder) using cross entropy loss (real, mixed, generated)
        Args:
            inputs: (s, a, next_s)
        """
        backward = True if best is None else False
        if backward:
            self.irl.train()
        input_s, input_a, input_next_s = inputs

        # 分块
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        real_loss, gen_loss = 0., 0.

        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            # 训练模式
            if backward:
                try:
                    data = self.irl_iter.next()
                except StopIteration:
                    self.irl_iter = iter(self.data_train)
                    data = self.irl_iter.next()
            # 测试模式
            else:
                try:
                    data = self.irl_iter_valid.next()
                except StopIteration:
                    self.irl_iter_valid = iter(self.data_valid)
                    data = self.irl_iter_valid.next()

            if backward:
                self.irl_optim.zero_grad()

            loss_real, loss_gen = self.irl_loop(data, (s, a, next_s))
            real_loss += loss_real.item()
            gen_loss += loss_gen.item()

            if backward:
                loss = loss_real + loss_gen
                loss.backward()
                self.irl_optim.step()

                for p in self.irl_params:
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)

        real_loss /= turns
        gen_loss /= turns

        # 训练模式：记录训练结果
        if backward:
            logging.debug('<<reward estimator {}>> epoch {}, loss_real:{}, loss_gen:{}'.format(
                self.character, epoch, real_loss, gen_loss))
            self.irl.eval()
        # 否则即为测试模式：记录验证集结果，保存最佳模型
        else:
            logging.debug('<<reward estimator {}>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
                self.character, epoch, real_loss, gen_loss))
            loss = real_loss + gen_loss
            if loss < best:
                logging.info('<<reward estimator {}>> best model saved'.format(self.character))
                best = loss
                self.save_irl(self.save_dir, 'best')
            return best

    # 推断reward
    def estimate(self, s, a, next_s, log_pi):
        """
        infer the reward of state action pair with the estimator
        """
        weight = self.irl(s, a.float(), next_s)  # weight = f(s, a, s')
        logging.debug('<<reward estimator {}>> weight {}'.format(self.character, weight.mean().item()))
        logging.debug('<<reward estimator {}>> log pi {}'.format(self.character, log_pi.mean().item()))
        # see AIRL paper
        # r = f(s, a, s') - log_p(a|s)
        reward = (weight - log_pi).squeeze(-1)
        return reward

    # 保存模型
    def save_irl(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + '/sys')
            os.makedirs(directory + '/usr')
            os.makedirs(directory + '/vnet')

        torch.save(self.irl.state_dict(), directory + '/' + self.character + '/' + str(epoch) + '_estimator.mdl')
        logging.info('<<reward estimator {}>> epoch {}: saved network to mdl'.format(self.character, epoch))

    # 载入模型
    def load_irl(self, filename):
        directory, epoch = filename.rsplit('/', 1)
        irl_mdl = directory + '/' + self.character + '/' + epoch + '_estimator.mdl'
        if os.path.exists(irl_mdl):
            self.irl.load_state_dict(torch.load(irl_mdl))
            logging.info('<<reward estimator {}>> loaded checkpoint from file: {}'.format(self.character, irl_mdl))
