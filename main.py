# -*- coding: utf-8 -*-
"""
@author: ryuichi takanobu
"""
import sys
import time
import logging
from utils import get_parser, init_logging_handler
from torch import multiprocessing as mp

from policy import Policy
from learner import Learner

from config import MultiWozConfig
from controller import Controller

from agenda import UserAgenda
from rule import SystemRule

from datamanager import DataManager


"""
预训练区
"""
def worker_policy_sys(args, manager, config):
    init_logging_handler(args.log_dir, '_policy_sys')
    agent = Policy(None, args, manager, config, 0, 'sys', True)
    
    best = float('inf')
    for e in range(args.epoch):
        agent.imitating(e)
        best = agent.imit_test(e, best)
def worker_policy_usr(args, manager, config):
    init_logging_handler(args.log_dir, '_policy_usr')
    agent = Policy(None, args, manager, config, 0, 'usr', True)
    
    best = float('inf')
    for e in range(args.epoch):
        agent.imitating(e)
        best = agent.imit_test(e, best)
# 新增项：预训练RE
def worker_estimator(args, manager, config, make_env):
    init_logging_handler(args.log_dir, '_estimator')
    agent = Learner(make_env, args, config, args.process, manager, pre_irl=True)
    agent.load(args.save_dir+'/best')
    best0, best1, best2 = float('inf'), float('inf'), float('inf')
    for e in range(args.epoch):
        agent.train_irl(e, args.batchsz_traj)
        best0, best1 = agent.test_irl(e, args.batchsz, best0, best1)
        best2 = agent.imit_value(e, args.batchsz_traj, best2)

"""
环境区
"""
# 在训练模式和测试模式中被调用
def make_env(data_dir, config):
    controller = Controller(data_dir, config)
    return controller
# 以下2个函数只在测试模式中被调用
def make_env_rule(data_dir, config):
    env = SystemRule(data_dir, config)
    return env
def make_env_agenda(data_dir, config):
    env = UserAgenda(data_dir, config)
    return env

"""
主函数区
"""
if __name__ == '__main__':
    parser = get_parser()
    argv = sys.argv[1:]
    args, _ = parser.parse_known_args(argv)
    
    if args.config == 'multiwoz':
        config = MultiWozConfig()
    else:
        raise NotImplementedError('Config of the dataset {} not implemented'.format(args.config))

    manager = DataManager(args.data_dir, config)

    init_logging_handler(args.log_dir)
    logging.debug(str(args))
    
    try:
        mp = mp.get_context('spawn')
    except RuntimeError:
        pass

    # 预训练模式a
    if args.pretrain:
        logging.debug('pretrain')

        processes = []
        process_args = (args, manager, config)

        # 预训练：系统智能体
        processes.append(mp.Process(target=worker_policy_sys, args=process_args))
        # 预训练：用户智能体
        processes.append(mp.Process(target=worker_policy_usr, args=process_args))
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        # 预训练：系统/用户端智能体
        worker_estimator(args, manager, config, make_env)
    # 测试模式
    elif args.test:
        logging.debug('test')
        logging.disable(logging.DEBUG)
    
        agent = Learner(make_env, args, config, 1, manager, infer=True)
        agent.load(args.load)

        # 测试：用户vs系统
        # agent.evaluate(args.test_case)
        
        # # 测试系统智能体：使用Agenda-based用户模拟器
        # env = make_env_agenda(args.data_dir, config)
        # agent.evaluate_with_agenda(env, args.test_case)

        # 测试用户智能体：使用Rule-based系统智能体
        env = make_env_rule(args.data_dir, config)
        agent.evaluate_with_rule(env, args.test_case)
    # 训练模式
    else:
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        logging.debug('train {}'.format(current_time))
    
        agent = Learner(make_env, args, config, args.process, manager)
        best = agent.load(args.load)

        for i in range(args.epoch):
            # 训练
            agent.update(args.batchsz_traj, i)
            # 验证
            best = agent.update(args.batchsz, i, best)

            current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            logging.debug('epoch {} {}'.format(i, current_time))