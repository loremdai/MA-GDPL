import torch
import torch.nn as nn
import numpy as np

from collections import namedtuple
import random

"""
下面定义策略(Policy)的网络结构，采用2层感知机实现，激活函数为relu。
"""
class MultiDiscretePolicy(nn.Module):
    def __init__(self, cfg, character='sys'):
        super(MultiDiscretePolicy, self).__init__()

        if character == 'sys':
            self.net = nn.Sequential(nn.Linear(cfg.s_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.a_dim))
        elif character == 'usr':
            self.net = nn.Sequential(nn.Linear(cfg.s_dim_usr, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, cfg.a_dim_usr))
        else:
            raise NotImplementedError('Unknown character {}'.format(character))

    def forward(self, s):
        # [b, s_dim] => [b, a_dim]
        a_weights = self.net(s)

        return a_weights

    def select_action(self, s, sample=True):
        """
        :param s: [s_dim]
        :return: [a_dim]
        """
        # forward to get action probs
        # [s_dim] => [a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)

        # [a_dim] => [a_dim, 2]
        a_probs = a_probs.unsqueeze(1)
        a_probs = torch.cat([1 - a_probs, a_probs], 1)

        # [a_dim, 2] => [a_dim]
        a = a_probs.multinomial(1).squeeze(1) if sample else a_probs.argmax(1)

        return a

    def batch_select_action(self, s, sample=False):
        """
        :param s: [b, s_dim]
        :return: [b, a_dim]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)

        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(2)
        a_probs = torch.cat([1 - a_probs, a_probs], 2)

        # [b, a_dim, 2] => [b*a_dim, 2] => [b*a_dim, 1] => [b*a_dim] => [b, a_dim]
        a = a_probs.reshape(-1, 2).multinomial(1).squeeze(1).reshape(a_weights.shape) if sample else a_probs.argmax(2)

        return a

    def get_log_prob(self, s, a):
        """
        :param s: [b, s_dim]
        :param a: [b, a_dim]
        :return: [b, 1]
        """
        # forward to get action probs
        # [b, s_dim] => [b, a_dim]
        a_weights = self.forward(s)
        a_probs = torch.sigmoid(a_weights)

        # [b, a_dim] => [b, a_dim, 2]
        a_probs = a_probs.unsqueeze(-1)
        a_probs = torch.cat([1 - a_probs, a_probs], -1)

        # [b, a_dim, 2] => [b, a_dim]
        trg_a_probs = a_probs.gather(-1, a.unsqueeze(-1)).squeeze(-1)
        log_prob = torch.log(trg_a_probs)

        return log_prob.sum(-1, keepdim=True)

"""
下面定义价值(Value)的网络结构
"""
class HybridValue(nn.Module):
    def __init__(self, cfg):
        super(HybridValue, self).__init__()

        self.net_sys_s = nn.Sequential(nn.Linear(cfg.s_dim, cfg.hs_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.hs_dim, cfg.hs_dim),
                                       nn.Tanh())
        self.net_usr_s = nn.Sequential(nn.Linear(cfg.s_dim_usr, cfg.hs_dim),
                                       nn.ReLU(),
                                       nn.Linear(cfg.hs_dim, cfg.hs_dim),
                                       nn.Tanh())

        self.net_sys = nn.Sequential(nn.Linear(cfg.hs_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, 1))
        self.net_usr = nn.Sequential(nn.Linear(cfg.hs_dim, cfg.h_dim),
                                     nn.ReLU(),
                                     nn.Linear(cfg.h_dim, 1))
        self.net_global = nn.Sequential(nn.Linear(cfg.hs_dim + cfg.hs_dim, cfg.h_dim),
                                        nn.ReLU(),
                                        nn.Linear(cfg.h_dim, 1))

    def forward(self, s, character):
        if character == 'sys':
            h_s_sys = self.net_sys_s(s)
            v = self.net_sys(h_s_sys)
        elif character == 'usr':
            h_s_usr = self.net_usr_s(s)
            v = self.net_usr(h_s_usr)
        elif character == 'global':
            h_s_usr = self.net_usr_s(s[0])
            h_s_sys = self.net_sys_s(s[1])
            h = torch.cat([h_s_usr, h_s_sys], -1)
            v = self.net_global(h)
        else:
            raise NotImplementedError('Unknown character {}'.format(character))
        return v.squeeze(-1)

"""
下面定义记忆（回放缓存），经验采用元组存储，缓存采用列表结构
"""
Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state'))

class Memory(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def get_batch(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def __len__(self):
        return len(self.memory)