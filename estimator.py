import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils import to_device, reparameterize
from dbquery import DBQuery

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RewardEstimator(object):
    def __init__(self, args, manager, config, pretrain=False, inference=False):

        # initialize IRL model
        self.irl = AIRL(config, args.gamma).to(device=DEVICE)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.step = 0
        self.anneal = args.anneal
        self.irl_params = self.irl.parameters()
        self.irl_optim = optim.RMSprop(self.irl_params, lr=args.lr_irl)
        self.weight_cliping_limit = args.clip

        self.save_dir = args.save_dir
        self.save_per_epoch = args.save_per_epoch
        self.optim_batchsz = args.batchsz
        self.irl.eval()

        db = DBQuery(args.data_dir, config)

        # 预训练模式，切分3个数据集后放入迭代器中。
        if pretrain:
            self.print_per_batch = args.print_per_batch
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db)
            self.data_test = manager.create_dataset_irl('test', args.batchsz, config, db)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)
            self.irl_iter_test = iter(self.data_test)
        # 训练模式，切分2个数据集后放入迭代器中。
        elif not inference:
            self.data_train = manager.create_dataset_irl('train', args.batchsz, config, db)
            self.data_valid = manager.create_dataset_irl('valid', args.batchsz, config, db)
            self.irl_iter = iter(self.data_train)
            self.irl_iter_valid = iter(self.data_valid)

    # 计算KL散度
    def kl_divergence(self, mu, logvar, istrain):
        klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        beta = min(self.step / self.anneal, 1) if istrain else 1
        return beta * klds

    # 计算并返回真实经验与模拟经验的loss
    def irl_loop(self, data_real, data_gen):
        s_real, a_real, next_s_real = to_device(data_real)
        s, a, next_s = data_gen

        # train with real data
        weight_real = self.irl(s_real, a_real, next_s_real)
        loss_real = -weight_real.mean()

        # train with generated data
        weight = self.irl(s, a, next_s)
        loss_gen = weight.mean()
        return loss_real, loss_gen

    # 训练模型
    def train_irl(self, batch, epoch):
        self.irl.train()
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        batchsz = input_s.size(0)

        # 将数据按照batch_size分块
        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        real_loss, gen_loss = 0., 0.

        # 在训练集上训练
        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            try:
                data = self.irl_iter.next()
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
        logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
            epoch, real_loss, gen_loss))
        if (epoch + 1) % self.save_per_epoch == 0:
            self.save_irl(self.save_dir, epoch)
        self.irl.eval()

    # 验证和测试最优模型
    def test_irl(self, batch, epoch, best):
        input_s = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
        input_a = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
        input_next_s = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
        batchsz = input_s.size(0)

        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        real_loss, gen_loss = 0., 0.

        # 在验证集上找出最优模型
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
        logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
            epoch, real_loss, gen_loss))
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
        logging.debug('<<reward estimator>> test, epoch {}, loss_real:{}, loss_gen:{}'.format(
            epoch, real_loss, gen_loss))
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

        turns = batchsz // self.optim_batchsz
        s_chunk = torch.chunk(input_s, turns)
        a_chunk = torch.chunk(input_a.float(), turns)
        next_s_chunk = torch.chunk(input_next_s, turns)

        real_loss, gen_loss = 0., 0.

        for s, a, next_s in zip(s_chunk, a_chunk, next_s_chunk):
            # 如果为训练模式，则进行训练
            if backward:
                try:
                    data = self.irl_iter.next()
                except StopIteration:
                    self.irl_iter = iter(self.data_train)
                    data = self.irl_iter.next()
            # 否则即为测试模式，在验证集上进行验证。
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

        # 如果为训练模式，则记录训练结果
        if backward:
            logging.debug('<<reward estimator>> epoch {}, loss_real:{}, loss_gen:{}'.format(
                epoch, real_loss, gen_loss))
            self.irl.eval()
        # 否则即为测试模式，记录验证集结果，保存最佳模型
        else:
            logging.debug('<<reward estimator>> validation, epoch {}, loss_real:{}, loss_gen:{}'.format(
                epoch, real_loss, gen_loss))
            loss = real_loss + gen_loss
            if loss < best:
                logging.info('<<reward estimator>> best model saved')
                best = loss
                self.save_irl(self.save_dir, 'best')
            return best

    # 使用reward estimator推断（输出）reward
    def estimate(self, s, a, next_s, log_pi):
        """
        infer the reward of state action pair with the estimator
        """
        weight = self.irl(s, a.float(), next_s)  # weight = f(s, a, s')
        logging.debug('<<reward estimator>> weight {}'.format(weight.mean().item()))
        logging.debug('<<reward estimator>> log pi {}'.format(log_pi.mean().item()))
        # see AIRL paper
        # r = f(s, a, s') - log_p(a|s)
        reward = (weight - log_pi).squeeze(-1)
        return reward

    # 保存模型
    def save_irl(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.irl.state_dict(), directory + '/' + str(epoch) + '_estimator.mdl')
        logging.info('<<reward estimator>> epoch {}: saved network to mdl'.format(epoch))

    # 载入模型
    def load_irl(self, filename):
        irl_mdl = filename + '_estimator.mdl'
        if os.path.exists(irl_mdl):
            self.irl.load_state_dict(torch.load(irl_mdl))
            logging.info('<<reward estimator>> loaded checkpoint from file: {}'.format(irl_mdl))


"""
下面定义逆强化学习网络结构，将reward estimator f(s,a)分为g(s,a)和h(s)。其中g()和h()为单层感知机结构。
"""
class AIRL(nn.Module):
    """
    label: 1 for real, 0 for generated
    """

    def __init__(self, cfg, gamma):
        super(AIRL, self).__init__()

        self.gamma = gamma
        self.g = nn.Sequential(nn.Linear(cfg.s_dim + cfg.a_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))
        self.h = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hi_dim),
                               nn.ReLU(),
                               nn.Linear(cfg.hi_dim, 1))

    def forward(self, s, a, next_s):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :param next_s: [b, s_dim]
        :return:  [b, 1]
        """
        # g(s,a) + γ * h(s') - h(s)
        weights = self.g(torch.cat([s, a], -1)) + self.gamma * self.h(next_s) - self.h(s)
        return weights