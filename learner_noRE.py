# -*- coding: utf-8 -*-
import os
import pickle
import logging

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

from rlmodule import MultiDiscretePolicy, HybridValue, Memory, Transition
from estimator import RewardEstimator
from utils import state_vectorize, state_vectorize_user

try:
    mp = mp.get_context('spawn')
except RuntimeError:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sampler(pid, queue, evt, env, policy_usr, policy_sys, batchsz):
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

    sampled_num = 0
    sampled_traj_num = 0
    traj_len = 40
    real_traj_len = 0

    while sampled_num < batchsz:
        # for each trajectory, we reset the env and get initial state
        s = env.reset()

        for t in range(traj_len):

            # [s_dim_usr] => [a_dim_usr]
            s_vec = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain))
            # 根据当前状态S^u，做出动作a^u
            a = policy_usr.select_action(s_vec.to(device=DEVICE)).cpu()

            # interact with env, done is a flag indicates ending or not
            next_s, done = env.step_usr(s, a)  # 与环境交互得到状态S^s

            # [s_dim] => [a_dim]
            next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db))
            # 根据当前状态S^s，做出动作a^s
            next_a = policy_sys.select_action(next_s_vec.to(device=DEVICE)).cpu()

            # interact with env
            s = env.step_sys(next_s, next_a)  # 与环境交互得到状态S'^u

            # get reward compared to demonstrations
            if done:
                env.set_rollout(True)
                s_vec_next = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain))
                a_next = torch.zeros_like(a)
                next_s_next, _ = env.step_usr(s, a_next)  # 与环境交互得到状态S'^s
                next_s_vec_next = torch.Tensor(state_vectorize(next_s_next, env.cfg, env.db))
                env.set_rollout(False)

                r_global = 20 if env.evaluator.task_success() else -5
            else:
                # one step roll out
                env.set_rollout(True)
                s_vec_next = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain))
                # 根据当前状态S'^u，做出动作a'^u（后续未用到a'^u）
                a_next = policy_usr.select_action(s_vec_next.to(device=DEVICE)).cpu()
                next_s_next, _ = env.step_usr(s, a_next)  # 与环境交互得到状态S'^s
                next_s_vec_next = torch.Tensor(state_vectorize(next_s_next, env.cfg, env.db))
                env.set_rollout(False)

                r_global = 5 if env.evaluator.cur_domain and env.evaluator.domain_success(
                    env.evaluator.cur_domain) else -1

            # save to queue
            # s^u, a^u, r^u, s'^u, s^s, a^s, r^s, s'^s, t, r_glo
            buff.push(s_vec.numpy(), a.numpy(), s_vec_next.numpy(), next_s_vec.numpy(), next_a.numpy(),
                      next_s_vec_next.numpy(), done, r_global)

            # update per step
            real_traj_len = t

            if done:
                break

        # this is end of one trajectory
        sampled_num += real_traj_len
        sampled_traj_num += 1
        # t indicates the valid trajectory length

    # this is end of sampling all batchsz of items.
    # when sampling is over, push all buff data into queue
    queue.put([pid, buff])
    evt.wait()


class Learner():

    def __init__(self, env_cls, args, cfg, process_num, manager, pre_irl=False, infer=False):
        self.policy_sys = MultiDiscretePolicy(cfg).to(device=DEVICE)
        self.policy_usr = MultiDiscretePolicy(cfg, 'usr').to(device=DEVICE)
        self.rewarder_sys = RewardEstimator(args, cfg, manager, character='sys', pretrain=pre_irl, inference=infer)
        self.rewarder_usr = RewardEstimator(args, cfg, manager, character='usr', pretrain=pre_irl, inference=infer)
        self.vnet = HybridValue(cfg).to(device=DEVICE)

        self.policy_sys.eval()
        self.policy_usr.eval()
        self.vnet.eval()
        self.infer = infer
        if not infer:
            self.target_vnet = HybridValue(cfg).to(device=DEVICE)
            self.episode_num = 0
            self.last_target_update_episode = 0
            self.target_update_interval = args.interval

            self.l2_loss = nn.MSELoss()
            self.multi_entropy_loss = nn.BCEWithLogitsLoss()

            self.policy_sys_optim = optim.RMSprop(self.policy_sys.parameters(), lr=args.lr_policy)
            self.policy_usr_optim = optim.RMSprop(self.policy_usr.parameters(), lr=args.lr_policy)
            self.vnet_optim = optim.RMSprop(self.vnet.parameters(), lr=args.lr_vnet, weight_decay=args.weight_decay)

        # initialize envs for each process
        self.env_list = []
        for _ in range(process_num):
            self.env_list.append(env_cls(args.data_dir, cfg))

        self.gamma = args.gamma
        self.tau = args.tau  # GAE广义优势估计超参数lambda（此处用tau）
        self.epsilon = args.epsilon
        self.update_round = args.update_round  # Epoch num for inner loop of PPO
        self.grad_norm_clip = args.clip
        self.optim_batchsz = args.batchsz
        self.save_per_epoch = args.save_per_epoch
        self.save_dir = args.save_dir
        self.process_num = process_num
        self.writer = SummaryWriter()

    """
    预训练RewardEstimator模块
    """

    # 预训练RE（逆强化学习）
    def train_irl(self, epoch, batchsz):
        batch = self.sample(batchsz)
        self.rewarder_sys.train_irl(batch, epoch)
        self.rewarder_usr.train_irl(batch, epoch)

    # 测试RE
    def test_irl(self, epoch, batchsz, best0, best1):
        batch = self.sample(batchsz)
        best_sys = self.rewarder_sys.test_irl(batch, epoch, best0)  # best = float('inf')
        best_usr = self.rewarder_usr.test_irl(batch, epoch, best1)

        return best_sys, best_usr

    # 预训练价值网络
    def imit_value(self, epoch, batchsz, best):
        self.vnet.train()

        batch = self.sample(batchsz)
        s_sys = torch.from_numpy(np.stack(batch.state_sys)).to(device=DEVICE)
        a_sys = torch.from_numpy(np.stack(batch.action_sys)).to(device=DEVICE)
        next_s_sys = torch.from_numpy(np.stack(batch.state_sys_next)).to(device=DEVICE)

        s_usr = torch.from_numpy(np.stack(batch.state_usr)).to(device=DEVICE)
        a_usr = torch.from_numpy(np.stack(batch.action_usr)).to(device=DEVICE)
        next_s_usr = torch.from_numpy(np.stack(batch.state_usr_next)).to(device=DEVICE)

        r_glo = torch.Tensor(np.stack(batch.reward_global)).to(device=DEVICE)
        ternimal = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)

        # sys part
        log_pi_old_sa_sys = self.policy_sys.get_log_prob(s_sys, a_sys).detach()
        r_sys = self.rewarder_sys.estimate(s_sys, a_sys, next_s_sys, log_pi_old_sa_sys).detach()
        v_target_sys = r_sys + self.gamma * (1 - ternimal) * self.target_vnet(next_s_sys, 'sys').detach()

        # usr part
        log_pi_old_sa_usr = self.policy_usr.get_log_prob(s_usr, a_usr).detach()
        r_usr = self.rewarder_usr.estimate(s_usr, a_usr, next_s_usr, log_pi_old_sa_usr).detach()
        v_target_usr = r_usr + self.gamma * (1 - ternimal) * self.target_vnet(next_s_usr, 'usr').detach()

        # glo part
        v_target_glo = r_glo + self.gamma * (1 - ternimal) * self.target_vnet((next_s_usr, next_s_sys),
                                                                              'global').detach()

        for i in range(self.update_round):
            perm = torch.randperm(batchsz)
            v_target_sys_shuf, s_sys_shuf, v_target_usr_shuf, s_usr_shuf, v_target_glo_shuf = v_target_sys[perm], s_sys[
                perm], \
                                                                                              v_target_usr[perm], s_usr[
                                                                                                  perm], v_target_glo[
                                                                                                  perm]
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            v_target_sys_shuf, s_sys_shuf, v_target_usr_shuf, s_usr_shuf, v_target_glo_shuf = torch.chunk(
                v_target_sys_shuf,
                optim_chunk_num), \
                                                                                              torch.chunk(s_sys_shuf,
                                                                                                          optim_chunk_num), \
                                                                                              torch.chunk(
                                                                                                  v_target_usr_shuf,
                                                                                                  optim_chunk_num), \
                                                                                              torch.chunk(s_usr_shuf,
                                                                                                          optim_chunk_num), \
                                                                                              torch.chunk(
                                                                                                  v_target_glo_shuf,
                                                                                                  optim_chunk_num)

            vnet_sys_loss, vnet_usr_loss, vnet_glo_loss, value_loss = 0., 0., 0., 0.
            for v_target_sys_b, s_sys_b, v_target_usr_b, s_usr_b, v_target_glo_b in zip(v_target_sys_shuf, s_sys_shuf,
                                                                                        v_target_usr_shuf, s_usr_shuf,
                                                                                        v_target_glo_shuf):
                # update vnet sys
                v_sys_b = self.vnet(s_sys_b, 'sys').squeeze(-1)
                loss_sys = self.l2_loss(v_sys_b, v_target_sys_b)
                vnet_sys_loss += loss_sys.item()

                # update vnet usr
                v_usr_b = self.vnet(s_usr_b, 'usr').squeeze(-1)
                loss_usr = self.l2_loss(v_usr_b, v_target_sys_b)
                vnet_usr_loss += loss_usr.item()

                # update vnet global
                v_glo_b = self.vnet((s_usr_b, s_sys_b), 'global').squeeze(-1)
                loss_glo = self.l2_loss(v_glo_b, v_target_glo_b)
                vnet_glo_loss += loss_glo.item()

                self.vnet_optim.zero_grad()
                value_loss = loss_usr + loss_sys + loss_glo
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vnet.parameters(), self.grad_norm_clip)
                self.vnet_optim.step()

            vnet_usr_loss /= optim_chunk_num
            vnet_sys_loss /= optim_chunk_num
            vnet_glo_loss /= optim_chunk_num
            value_loss /= optim_chunk_num
            logging.debug('<<Hybrid Vnet> epoch {}, iteration {}, value network: usr {}, sys {}, global {}, total {}' \
                          .format(epoch, i, vnet_usr_loss, vnet_sys_loss, vnet_glo_loss, value_loss))

        if value_loss < best:
            logging.info('<<dialog policy>> best model saved')
            best = value_loss  # 记录该损失为best
            self.save(self.save_dir, 'best', True)  # 保存最佳模型
        # 每隔XX轮保存一次模型
        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch, True)
        self.vnet.eval()  # 关闭训练模式

        return best  # 返回最佳（最小）损失ds

    """
    测试模块
    """

    # 测试：用户vs系统
    def evaluate(self, N):
        logging.info('eval: user 2 system')
        env = self.env_list[0]
        traj_len = 40
        turn_tot, inform_tot, match_tot, success_tot = [], [], [], []
        for seed in range(N):
            s = env.reset(seed)
            print('seed', seed)
            print('origin goal', env.goal)
            print('goal', env.evaluator.goal)
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy_usr.select_action(s_vec, False)
                next_s, done = env.step_usr(s, a)

                next_s_vec = torch.Tensor(state_vectorize(next_s, env.cfg, env.db)).to(device=DEVICE)
                next_a = self.policy_sys.select_action(next_s_vec, False)
                s = env.step_sys(next_s, next_a)

                print('usr', s['user_action'])
                print('sys', s['sys_action'])

                if done:
                    break

            turn_tot.append(env.time_step // 2)
            match_tot += env.evaluator.match_rate(aggregate=False)
            inform_tot.append(env.evaluator.inform_F1(aggregate=False))
            print('turn', env.time_step // 2)
            match_session = env.evaluator.match_rate()
            print('match', match_session)
            inform_session = env.evaluator.inform_F1()
            print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
                    or (match_session == 1 and inform_session[1] is None) \
                    or (match_session is None and inform_session[1] == 1):
                print('success', 1)
                success_tot.append(1)
            else:
                print('success', 0)
                success_tot.append(0)

        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))

    # 测试系统智能体：使用基于日程的用户模拟器
    def evaluate_with_agenda(self, env, N):
        logging.info('eval: agenda 2 system')
        traj_len = 40
        turn_tot, inform_tot, match_tot, success_tot = [], [], [], []
        for seed in range(N):
            s = env.reset(seed)
            print('seed', seed)
            print('goal', env.goal.domain_goals)
            print('usr', s['user_action'])
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy_sys.select_action(s_vec, False)
                next_s, done = env.step(s, a.cpu())
                s = next_s
                print('sys', s['sys_action'])
                print('usr', s['user_action'])
                if done:
                    break
            s_vec = torch.Tensor(state_vectorize(s, env.cfg, env.db)).to(device=DEVICE)
            # mode with policy during evaluation
            a = self.policy_sys.select_action(s_vec, False)
            s = env.update_belief_sys(s, a.cpu())
            print('sys', s['sys_action'])

            assert (env.time_step % 2 == 0)
            turn_tot.append(env.time_step // 2)
            match_tot += env.evaluator.match_rate(aggregate=False)
            inform_tot.append(env.evaluator.inform_F1(aggregate=False))
            print('turn', env.time_step // 2)
            match_session = env.evaluator.match_rate()
            print('match', match_session)
            inform_session = env.evaluator.inform_F1()
            print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
                    or (match_session == 1 and inform_session[1] is None) \
                    or (match_session is None and inform_session[1] == 1):
                print('success', 1)
                success_tot.append(1)
            else:
                print('success', 0)
                success_tot.append(0)

        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))

    # 测试用户智能体：使用基于规则的系统智能体
    def evaluate_with_rule(self, env, N):
        logging.info('eval: user 2 rule')
        traj_len = 40
        turn_tot, inform_tot, match_tot, success_tot = [], [], [], []
        for seed in range(N):
            s = env.reset(seed)
            print('seed', seed)
            print('goal', env.evaluator.goal)
            for t in range(traj_len):
                s_vec = torch.Tensor(state_vectorize_user(s, env.cfg, env.evaluator.cur_domain)).to(device=DEVICE)
                # mode with policy during evaluation
                a = self.policy_usr.select_action(s_vec, False)
                next_s = env.step(s, a.cpu())
                s = next_s
                print('usr', s['user_action'])
                print('sys', s['sys_action'])
                done = s['others']['terminal']
                if done:
                    break

            assert (env.time_step % 2 == 0)
            turn_tot.append(env.time_step // 2)
            match_tot += env.evaluator.match_rate(aggregate=False)
            inform_tot.append(env.evaluator.inform_F1(aggregate=False))
            print('turn', env.time_step // 2)
            match_session = env.evaluator.match_rate()
            print('match', match_session)
            inform_session = env.evaluator.inform_F1()
            print('inform', inform_session)
            if (match_session == 1 and inform_session[1] == 1) \
                    or (match_session == 1 and inform_session[1] is None) \
                    or (match_session is None and inform_session[1] == 1):
                print('success', 1)
                success_tot.append(1)
            else:
                print('success', 0)
                success_tot.append(0)

        logging.info('turn {}'.format(np.mean(turn_tot)))
        logging.info('match {}'.format(np.mean(match_tot)))
        TP, FP, FN = np.sum(inform_tot, 0)
        prec = TP / (TP + FP)
        rec = TP / (TP + FN)
        F1 = 2 * prec * rec / (prec + rec)
        logging.info('inform rec {}, F1 {}'.format(rec, F1))
        logging.info('success {}'.format(np.mean(success_tot)))

    """
    训练模块
    """

    # 计算价值函数V和优势函数A
    def _update_targets(self):
        self.target_vnet.load_state_dict(self.vnet.state_dict())
        logging.info('Updated target network')

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
        delta = torch.Tensor(batchsz).to(device=DEVICE)  # TD-error
        A_sa = torch.Tensor(batchsz).to(device=DEVICE)  # 优势函数

        # 初始化为0
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

    # 算法总流程
    def update(self, batchsz, epoch, best=None):
        """
        firstly sample batchsz items and then perform optimize algorithms.
        :param batchsz:
        :param epoch:
        :param best:
        :return:
        """
        backward = True if best is None else False
        if backward:
            self.policy_sys.train()
            self.policy_usr.train()
            self.vnet.train()

        # 1. sample data asynchronously
        batch = self.sample(batchsz)

        s_sys = torch.from_numpy(np.stack(batch.state_sys)).to(device=DEVICE)
        a_sys = torch.from_numpy(np.stack(batch.action_sys)).to(device=DEVICE)
        s_sys_next = torch.from_numpy(np.stack(batch.state_sys_next)).to(device=DEVICE)
        r_sys = torch.Tensor(np.stack(batch.reward_sys)).to(device=DEVICE)
        batchsz_sys = s_sys.size(0)

        s_usr = torch.from_numpy(np.stack(batch.state_usr)).to(device=DEVICE)
        a_usr = torch.from_numpy(np.stack(batch.action_usr)).to(device=DEVICE)
        s_usr_next = torch.from_numpy(np.stack(batch.state_usr_next)).to(device=DEVICE)
        r_usr = torch.Tensor(np.stack(batch.reward_usr)).to(device=DEVICE)
        batchsz_usr = s_usr.size(0)

        ternimal = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
        r_glo = torch.Tensor(np.stack(batch.reward_global)).to(device=DEVICE)

        # 2. update reward estimator
        inputs_sys = (s_sys, a_sys, s_sys_next)
        inputs_usr = (s_usr, a_usr, s_usr_next)
        if backward:  # 若为训练模式
            self.rewarder_sys.update_irl(inputs_sys, batchsz_sys, epoch)
            self.rewarder_usr.update_irl(inputs_usr, batchsz_usr, epoch)
        else:
            best[1] = self.rewarder_sys.update_irl(inputs_sys, batchsz_sys, epoch, best[1])
            best[2] = self.rewarder_usr.update_irl(inputs_usr, batchsz_usr, epoch, best[2])

        # 3. compute rewards
        log_pi_old_sa_sys = self.policy_sys.get_log_prob(s_sys, a_sys).detach()
        log_pi_old_sa_usr = self.policy_usr.get_log_prob(s_usr, a_usr).detach()

        # r_sys = self.rewarder_sys.estimate(s_sys, a_sys, s_sys_next, log_pi_old_sa_sys).detach()
        # r_usr = self.rewarder_usr.estimate(s_usr, a_usr, s_usr_next, log_pi_old_sa_usr).detach()

        # 4. estimate A and V_td-target
        A_sys = r_sys + self.gamma * (1 - ternimal) * self.vnet(s_sys_next, 'sys').detach() - self.vnet(s_sys,
                                                                                                        'sys').detach()
        v_target_sys = r_sys + self.gamma * (1 - ternimal) * self.target_vnet(s_sys_next, 'sys').detach()

        A_usr = r_usr + self.gamma * (1 - ternimal) * self.vnet(s_usr_next, 'usr').detach() - self.vnet(s_usr,
                                                                                                        'usr').detach()
        v_target_usr = r_usr + self.gamma * (1 - ternimal) * self.target_vnet(s_usr_next, 'usr').detach()

        A_glo = r_glo + self.gamma * (1 - ternimal) * self.vnet((s_usr_next, s_sys_next),
                                                                'global').detach() - self.vnet((s_usr, s_sys),
                                                                                               'global').detach()
        v_target_glo = r_glo + self.gamma * (1 - ternimal) * self.target_vnet((s_usr_next, s_sys_next),
                                                                              'global').detach()
        if not backward:
            reward = r_sys.mean().item() + r_usr.mean().item() + r_glo.mean().item()
            logging.debug('validation, epoch {}, reward {}'.format(epoch, reward))
            self.writer.add_scalar('train/reward', reward, epoch)
            if reward > best[3]:
                logging.info('best model saved')
                best[3] = reward
                self.save(self.save_dir, 'best')
            with open(self.save_dir + '/best.pkl', 'wb') as f:
                pickle.dump(best, f)
            return best
        else:
            logging.debug(
                'epoch {}, reward: sys {}, usr {}, global {}'.format(epoch, r_sys.mean().item(), r_usr.mean().item(),
                                                                     r_glo.mean().item()))

        # 6. update dialog policy
        for i in range(self.update_round):

            # 1. shuffle current batch
            perm = torch.randperm(batchsz)

            v_target_usr_shuf, A_usr_shuf, s_usr_shuf, a_usr_shuf, log_pi_old_sa_usr_shuf, v_target_sys_shuf, A_sys_shuf, s_sys_shuf, a_sys_shuf, log_pi_old_sa_sys_shuf, v_target_glo_shuf, A_glo_shuf, r_glo_shuf = \
                v_target_usr[perm], A_usr[perm], s_usr[perm], a_usr[perm], log_pi_old_sa_usr[perm], \
                v_target_sys[perm], A_sys[perm], s_sys[perm], a_sys[perm], log_pi_old_sa_sys[perm], \
                v_target_glo[perm], A_glo[perm], r_glo[perm]

            # 2. get mini-batch for optimizing
            optim_chunk_num = int(np.ceil(batchsz / self.optim_batchsz))
            # chunk the optim_batch for total batch
            v_target_usr_shuf, A_usr_shuf, s_usr_shuf, a_usr_shuf, log_pi_old_sa_usr_shuf, v_target_sys_shuf, A_sys_shuf, s_sys_shuf, a_sys_shuf, log_pi_old_sa_sys_shuf, v_target_glo_shuf, A_glo_shuf, r_glo_shuf = \
                torch.chunk(v_target_usr_shuf, optim_chunk_num), torch.chunk(A_usr_shuf, optim_chunk_num), torch.chunk(
                    s_usr_shuf, optim_chunk_num), \
                torch.chunk(a_usr_shuf, optim_chunk_num), torch.chunk(log_pi_old_sa_usr_shuf, optim_chunk_num), \
                torch.chunk(v_target_sys_shuf, optim_chunk_num), torch.chunk(A_sys_shuf, optim_chunk_num), torch.chunk(
                    s_sys_shuf, optim_chunk_num), \
                torch.chunk(a_sys_shuf, optim_chunk_num), torch.chunk(log_pi_old_sa_sys_shuf, optim_chunk_num), \
                torch.chunk(v_target_glo_shuf, optim_chunk_num), torch.chunk(A_glo_shuf, optim_chunk_num), torch.chunk(
                    r_glo_shuf, optim_chunk_num)

            # 3. iterate all mini-batch to optimize
            policy_usr_loss, policy_sys_loss, vnet_usr_loss, vnet_sys_loss, vnet_glo_loss = 0., 0., 0., 0., 0.

            for v_target_usr_b, A_usr_b, s_usr_b, a_usr_b, log_pi_old_sa_usr_b, v_target_sys_b, A_sys_b, s_sys_b, a_sys_b, log_pi_old_sa_sys_b, v_target_glo_b, A_glo_b, r_glo_b in \
                    zip(v_target_usr_shuf, A_usr_shuf, s_usr_shuf, a_usr_shuf, log_pi_old_sa_usr_shuf,
                        v_target_sys_shuf, A_sys_shuf, s_sys_shuf, a_sys_shuf, log_pi_old_sa_sys_shuf,
                        v_target_glo_shuf, A_glo_shuf, r_glo_shuf):
                # 1. update value network
                # update sys vnet
                v_sys_b = self.vnet(s_sys_b, 'sys')
                loss_sys = self.l2_loss(v_sys_b, v_target_sys_b)
                vnet_sys_loss += loss_sys.item()

                # update usr vnet
                v_usr_b = self.vnet(s_usr_b, 'usr')
                loss_usr = self.l2_loss(v_usr_b, v_target_usr_b)
                vnet_usr_loss += loss_usr.item()

                # update global vnet
                v_glo_b = self.vnet((s_usr_b, s_sys_b), 'global')
                loss_glo = self.l2_loss(v_glo_b, v_target_glo_b)
                vnet_glo_loss += loss_glo.item()

                self.vnet_optim.zero_grad()
                loss = loss_usr + loss_sys + loss_glo
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vnet.parameters(), self.grad_norm_clip)
                self.vnet_optim.step()

                self.episode_num += 1
                if (self.episode_num - self.last_target_update_episode) / self.target_update_interval >= 1.0:
                    self._update_targets()
                    self.last_target_update_episode = self.episode_num

                # 2. update policy by PPO
                # update sys policy
                self.policy_sys_optim.zero_grad()
                log_pi_sa_sys = self.policy_sys.get_log_prob(s_sys_b, a_sys_b)
                ratio_sys = (log_pi_sa_sys - log_pi_old_sa_sys_b).exp().squeeze(-1)
                surrogate1_sys = ratio_sys * (A_sys_b + A_glo_b)
                surrogate2_sys = torch.clamp(ratio_sys, 1 - self.epsilon, 1 + self.epsilon) * (A_sys_b + A_glo_b)
                surrogate_sys = - torch.min(surrogate1_sys, surrogate2_sys).mean()
                policy_sys_loss += surrogate_sys.item()

                surrogate_sys.backward()
                torch.nn.utils.clip_grad_norm(self.policy_sys.parameters(), self.grad_norm_clip)
                self.policy_sys_optim.step()

                # update usr policy
                self.policy_usr_optim.zero_grad()
                log_pi_sa_usr = self.policy_usr.get_log_prob(s_usr_b, a_usr_b)  # [b, 1]
                ratio_usr = (log_pi_sa_usr - log_pi_old_sa_usr_b).exp().squeeze(-1)  # [b, 1] => [b]
                surrogate1_usr = ratio_usr * (A_usr_b + A_glo_b)
                surrogate2_usr = torch.clamp(ratio_usr, 1 - self.epsilon, 1 + self.epsilon) * (A_usr_b + A_glo_b)
                surrogate_usr = - torch.min(surrogate1_usr, surrogate2_usr).mean()
                policy_usr_loss += surrogate_usr.item()

                surrogate_usr.backward()  # backprop
                torch.nn.utils.clip_grad_norm(self.policy_usr.parameters(),
                                              self.grad_norm_clip)  # gradient clipping, for stability
                self.policy_usr_optim.step()

            vnet_usr_loss /= optim_chunk_num
            vnet_sys_loss /= optim_chunk_num
            vnet_glo_loss /= optim_chunk_num
            policy_usr_loss /= optim_chunk_num
            policy_sys_loss /= optim_chunk_num

            # 记录loss信息
            logging.debug('epoch {}, iteration {}, policy: usr {}, sys {}'.format
                          (epoch, i, policy_usr_loss, policy_sys_loss))
            logging.debug('epoch {}, iteration {}, vnet: usr {}, sys {}, global {}'.format
                          (epoch, i, vnet_usr_loss, vnet_sys_loss, vnet_glo_loss))
            self.writer.add_scalar('train/usr_policy_loss', policy_usr_loss, epoch)
            self.writer.add_scalar('train/sys_policy_loss', policy_sys_loss, epoch)
            self.writer.add_scalar('train/vnet_usr_loss', vnet_usr_loss, epoch)
            self.writer.add_scalar('train/vnet_sys_loss', vnet_sys_loss, epoch)
            self.writer.add_scalar('train/vnet_glo_loss', vnet_glo_loss, epoch)

        # 保存训练模型
        if (epoch + 1) % self.save_per_epoch == 0:
            self.save(self.save_dir, epoch)
            with open(self.save_dir + '/' + str(epoch) + '.pkl', 'wb') as f:
                pickle.dump(best, f)
        self.policy_sys.eval()
        self.policy_usr.eval()
        self.vnet.eval()

    """
    辅助函数模块
    """

    # 采样
    def sample(self, batchsz):
        """
        Given batchsz number of task, the batchsz will be split equally to each processes
        and when processes return, it merge all data and return
        :param batchsz:
        :return: batch
        """

        # batchsz will be split into each process,
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
            process_args = (i, queue, evt, self.env_list[i], self.policy_usr, self.policy_sys, process_batchsz)
            processes.append(mp.Process(target=sampler, args=process_args))
        for p in processes:
            # set the process as daemon, and it will be killed once the main process is stoped.
            p.daemon = True
            p.start()

        # we need to get the first Memory object and then merge others Memory use its append function.
        pid0, buff0 = queue.get()
        for _ in range(1, self.process_num):
            pid, buff_ = queue.get()
            buff0.append(buff_)  # merge current Memory into buff0
        evt.set()

        # now buff saves all the sampled data
        buff = buff0

        return buff.get_batch()

    # 保存模型
    def save(self, directory, epoch, rl_only=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
            os.makedirs(directory + '/sys')
            os.makedirs(directory + '/usr')
            os.makedirs(directory + '/vnet')

        if not rl_only:
            self.rewarder_sys.save_irl(directory, epoch)
            self.rewarder_usr.save_irl(directory, epoch)

        torch.save(self.policy_sys.state_dict(), directory + '/sys/' + str(epoch) + '_pol.mdl')
        torch.save(self.policy_usr.state_dict(), directory + '/usr/' + str(epoch) + '_pol.mdl')
        torch.save(self.vnet.state_dict(), directory + '/vnet/' + str(epoch) + '_vnet.mdl')

        logging.info('<<multi agent learner>> epoch {}: saved network to mdl'.format(epoch))

    # 加载模型
    def load(self, filename):
        directory, epoch = filename.rsplit('/', 1)

        self.rewarder_sys.load_irl(filename)
        self.rewarder_usr.load_irl(filename)

        policy_sys_mdl = directory + '/sys/' + epoch + '_pol.mdl'
        if os.path.exists(policy_sys_mdl):
            self.policy_sys.load_state_dict(torch.load(policy_sys_mdl))
            logging.info('<<dialog policy sys>> loaded checkpoint from file: {}'.format(policy_sys_mdl))

        policy_usr_mdl = directory + '/usr/' + epoch + '_pol.mdl'
        if os.path.exists(policy_usr_mdl):
            self.policy_usr.load_state_dict(torch.load(policy_usr_mdl))
            logging.info('<<dialog policy usr>> loaded checkpoint from file: {}'.format(policy_usr_mdl))

        if not self.infer:
            self._update_targets()

        best_pkl = filename + '.pkl'
        if os.path.exists(best_pkl):
            with open(best_pkl, 'rb') as f:
                best = pickle.load(f)
        else:
            # unknown, sys_reward_loss, usr_reward_loss, total_reward
            best = [float('inf'), float('inf'), float('inf'), float('-inf')]

        return best
