import numpy as np
from core.sampler.evolution import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize
from core.sampler.base_sampler import BaseSampler
from torch import distributed as dist
import torch
import math
import random
from core.search_space.ops import Conv2d, InvertedResidual, FC
import torch.nn as nn
import time
import logging
import os 
import yaml

class EvolutionSampler(BaseSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flops_constraint = float(self.flops_constraint)
        self.scaling_topk = {}
        self.basenet_topk = {}
        self.scaling_stage = self.cfg_sampler['kwargs'].get('max_scaling_stage', 3)
        self.ratio = self.cfg_sampler['kwargs'].get('ratio', 1.)
        self.weights = {}
        self.weights[self.scaling_stage] = self.ratio
        for i in range(self.scaling_stage - 1, 0, -1):
            self.weights[i] = self.weights[i + 1] * self.ratio

        self.start_step = 'start-1'
        self.arch_topk_path = self.cfg_stg_sample['save_path'] + '/' + self.cfg_stg_sample['resume_name']
        if os.path.exists(self.arch_topk_path):
            arch_topk = yaml.load(open(self.arch_topk_path, 'r'), Loader=yaml.FullLoader)
            if type(arch_topk['scaling_topk']) == str:
                arch_topk['scaling_topk'] = eval(arch_topk['scaling_topk'])
            self.scaling_topk = arch_topk['scaling_topk']
            if type(arch_topk['basenet_topk']) == str:
                arch_topk['basenet_topk'] = eval(arch_topk['basenet_topk'])
            self.basenet_topk = arch_topk['basenet_topk']
            self.admm_iter = arch_topk['admm_iter']
            self.start_step = arch_topk['start_step']
        self.err_base = 25e6
        self.err_scale = 50e6

    def generate_subnet(self):
        return None
    
    def check_basenet(self, basenet):
        # basenet: list, [op, ...]
        depth_stage = self.model.module.depth_stage# [[base_max_depth, max_depth], ...]

        # Divide basenet into stages
        basenet_stage = []# [stage1: [op1, op2, ...], ...]
        idx = 0
        for i in depth_stage:
            basenet_stage.append(basenet[idx: (idx + i[1])])
            idx += i[1]
        
        # Check each stage
        for i in range(len(basenet_stage)):
            if type(basenet_stage[i]) == list and depth_stage[i][0] > 1:
                # Check the first operation
                if basenet_stage[i][0] == 0:
                    basenet_stage[i][0] = 1
                # Check the thermometer code
                id_flag = -1 # first identity opereation index
                for j in range(depth_stage[i][0]):# Only consider the basenet, ignore the scaled parts
                    if basenet_stage[i][j] > 0 and id_flag != -1:# Swap the non-id with id
                        basenet_stage[i][id_flag] = basenet_stage[i][j]
                        basenet_stage[i][j] = 0
                        id_flag = -1
                    elif basenet_stage[i][j] == 0 and id_flag == -1:
                        id_flag = j

        # Gather the checked basenet
        basenet = []
        for i in basenet_stage:
            if type(i) == list:
                basenet += i
            else:
                basenet.append(i)
        
        # Check identity
        id = []# base->1, scaled parts->0
        for i in depth_stage:
            if i[0] == i[1]:
                id.append(1)
            else:
                id += [1] * i[0] + [0] * (i[1] - i[0])
        assert len(id) == len(basenet)
        basenet = list(map(lambda x, y: x * y, basenet, id))

        return basenet

    def generate_subnet_(self, basenet, scaling, scaling_stage=0):
        # Generate a subnet from a base model and a scaling strategy
        # basenet: list, [op, ...]
        # scaling: list of index, [depth, width, resolution]
        # scaling_stage: int
        basenet = self.check_basenet(basenet)
        depth_stage = self.model.module.depth_stage# [[base_max_depth, max_depth], ...]
        if scaling_stage == 0:
            scaling_value = [1., 1., 224]
        else:
            scaling_value = [self.model.module.depth_multiplier[scaling_stage][scaling[scaling_stage][0]],
                            self.model.module.channel_multiplier[scaling_stage][scaling[scaling_stage][1]],
                            self.model.module.resolution_multiplier[scaling_stage][scaling[scaling_stage][2]]]

        # Divide basenet into stages
        basenet_stage = []# [stage1: [op1, op2, ...], ...]
        idx = 0
        for i in depth_stage:
            basenet_stage.append(basenet[idx: (idx + i[1])])
            idx += i[1]

        # Scaling depth
        for i in range(len(basenet_stage)):
            if type(basenet_stage[i]) == list and len(basenet_stage[i]) > 1:
                # Compute the depth of basenet in this stage
                n_base = 0
                for j in range(depth_stage[i][0]):# Only consider the basenet, ignore the scaled parts
                    if basenet_stage[i][j] > 0:
                        n_base += 1
                    else:
                        break
                # Compute the scaled depth of basenet in this stage
                n = int(math.ceil(n_base * scaling_value[0]))
                # Get the scaled stage. n <= len(basenet_stage) is guaranteed
                basenet_stage[i][n_base: n] = [basenet_stage[i][n_base - 1]] * (n - n_base)

        # Get subnet
        subnet = []
        for i in basenet_stage:
            if type(i) == list:
                subnet += i
            else:
                subnet.append(i)
        return subnet + scaling_value[1:]# [op, ... , c_m, r]
    
    def eval_subnet(self, basenet, scaling, scaling_stage, err):
        # basenet: list, [op, ...]
        # scaling: dict of list of index, [depth, width, resolution]
        # scaling_stage: list
        assert len(self.model.module.net) == len(basenet)
        flops_constraint = getattr(self, 'flops_constraint', 400e6)

        score = []
        weights_used = []
        flops_ = self.count_flops(basenet + [1., 224])
        if abs(flops_constraint - flops_) <= err:
            for i in scaling_stage:
                weights_used.append(self.weights[i])
                subnet = self.generate_subnet_(basenet, scaling, scaling_stage=i)
                flops = self.count_flops(subnet)
                flops_constraint_ = flops_constraint * (2 ** i)
                self.logger.info('==subnet: {}, FLOPs: {}'.format(str(subnet), flops))

                if not self.check_flops([basenet], scaling, i, err):
                    score.append(3. + (flops_constraint_ - flops) / flops_constraint_)
                else:
                    self.model.module.reset_bn()
                    self.model.train()
                    time.sleep(2)# process may be stuck in dataloader
                    score.append(self.tester.test(subnet=subnet))
                    time.sleep(2)# process may be stuck in dataloader
            self.logger.info(score)
            score = sum([s * w for s, w in zip(score, weights_used)]) / sum(weights_used)

            if self.logger is not None:
                self.logger.info('{}-{}-{}-{}-{}\n'.format(str(basenet), str(scaling), 
                                                        self.scaling_stage, score, flops))
        else:
            score = 3. + (flops_constraint - flops_) / flops_constraint

            if self.logger is not None:
                self.logger.info('{}-{}-{}-{}-{}\n'.format(str(basenet), str(scaling), 
                                                        self.scaling_stage, score, flops_))
        return score

    def eval_subnet_host(self, basenet, scaling, scaling_stage, err):
        # basenet: list or list of list
        # scaling: dict
        # scaling_stage: list
        if isinstance(basenet[0], list):
            score = []
            for basenet_ in basenet:
                score.append(self.eval_subnet(basenet_, scaling, scaling_stage, err))
            score = sum(score) / len(score)
        else:
            score = self.eval_subnet(basenet, scaling, scaling_stage, err)
        return score

    def sample_basenet(self, fixed_scaling, pop_size, n_gens, sample_num):
        # Optimize basenet, fix scaling index
        # fixed_scaling: dict
        # print('==rank{}=={}'.format(self.rank, 2))
        depth_stage = self.model.module.depth_stage# [[base_max_depth, max_depth], ...]
        for s in fixed_scaling.keys():
            assert len(fixed_scaling[s]) == 3
        
        basenet_eval_dict = {}
        
        n_offspring = None #40
        # print('==rank{}=={}'.format(self.rank, 3))
        # setup NAS search problem
        n_var = len(self.model.module.net) # operation index
        lb = np.zeros(n_var)  # left index of each block
        ub = np.zeros(n_var) # right index of each block

        for i, block in enumerate(self.model.module.net):# Generate for the whole supernet
            ub[i] = len(block) - 1

        idx = 0
        for i in depth_stage:
            for j in range(i[0]):
                if i[0] == 1 and i[1] == 1:# conv_stem, conv_out, pooling, FC
                    ub[idx + j] = 0
                elif j == 0:# Downsampling operation in the stage
                    lb[idx + j] = 1
            for j in range(i[0], i[1]):# The scaled operations are not contained in the basenet
                ub[idx + j] = 0
            idx += i[1]

        # print('==rank{}=={}'.format(self.rank, 4))
        scaling_stage = [i for i in range(1, self.scaling_stage + 1)]
        nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub,
                eval_func=lambda basenet: self.eval_subnet_host(basenet, fixed_scaling, scaling_stage, self.err_base),
                result_dict=basenet_eval_dict, rank=self.rank, world_size=self.world_size)

        # print('==rank{}=={}'.format(self.rank, 5))
        # configure the nsga-net method
        init_sampling = []
        for _ in range(pop_size):
            init_sampling.append(self.get_init_basenet(flops=self.flops_constraint))
        # print('==rank{}=={}'.format(self.rank, 6))
        method = engine.nsganet(pop_size=pop_size,
                                        n_offsprings=n_offspring,
                                        eliminate_duplicates=True,
                                        sampling=np.array(init_sampling, dtype=np.int32))
                                        
        # print('==rank{}=={}'.format(self.rank, 7))
        res = minimize(nas_problem,
                           method,
                           callback=lambda algorithm: self.generation_callback(algorithm),
                           termination=('n_gen', n_gens))

        if self.rank == 0:
            sorted_basenet = sorted(basenet_eval_dict.items(), key=lambda i: i[1]['acc'], reverse=True)
            sorted_basenet_key = [x[1]['arch'] for x in sorted_basenet]
            basenet_topk = sorted_basenet_key[:sample_num]
            self.logger.info('== search result ==')
            self.logger.info([[list(x[1]['arch']), x[1]['acc']] for x in sorted_basenet])
            self.logger.info('== best basenet ==')
            self.logger.info([list(x) for x in basenet_topk])
            self.basenet_topk = basenet_topk
            basenet_topk = torch.IntTensor(basenet_topk).cuda()
            dist.broadcast(basenet_topk, 0)
        else:
            basenet_topk = torch.IntTensor([[0] * n_var for _ in range(sample_num)]).cuda()
            dist.broadcast(basenet_topk, 0)
            self.basenet_topk = basenet_topk.cpu().tolist()
    
    def sample_scaling(self, fixed_basenet=None, scaling_stage=1):
        # Optimize scaling index, fix basenet
        assert len(fixed_basenet) == len(self.model.module.net)

        scaling_eval_dict = {}
        
        n_offspring = None #40

        # setup NAS search problem
        n_var = 3 # depth, width, resolution
        lb = np.zeros(n_var)  # left index of scaling multiplier
        ub = np.array([len(self.model.module.depth_multiplier[scaling_stage]) - 1, # right index of scaling multiplier
                        len(self.model.module.channel_multiplier[scaling_stage]) - 1,
                        len(self.model.module.resolution_multiplier[scaling_stage]) - 1], dtype=float) 

        nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub,
            eval_func=lambda scaling: self.eval_subnet_host(fixed_basenet, scaling, [scaling_stage], self.err_scale),
            result_dict=scaling_eval_dict, rank=self.rank, world_size=self.world_size)

        # configure the nsga-net method
        init_sampling = []
        for _ in range(self.pop_size_scaling):
            init_sampling.append(self.get_init_scaling(scaling_stage, [fixed_basenet]))
        method = engine.nsganet(pop_size=self.pop_size_scaling,
                                        n_offsprings=n_offspring,
                                        eliminate_duplicates=True,
                                        sampling=np.array(init_sampling, dtype=np.int32))

        res = minimize(nas_problem,
                           method,
                           callback=lambda algorithm: self.generation_callback(algorithm),
                           termination=('n_gen', self.n_gens_scaling))

        if self.rank == 0:
            sorted_scaling = sorted(scaling_eval_dict.items(), key=lambda i: i[1]['acc'], reverse=True)
            sorted_scaling_key = [x[1]['arch'] for x in sorted_scaling]
            scaling_topk = sorted_scaling_key[:self.sample_num_scaling]
            self.logger.info('== search result ==')
            self.logger.info([[list(x[1]['arch']), x[1]['acc']] for x in sorted_scaling])
            self.logger.info('== best scaling ==')
            self.logger.info([list(x) for x in scaling_topk])
            self.scaling_topk[scaling_stage] = scaling_topk
            scaling_topk = torch.IntTensor(scaling_topk).cuda()
            dist.broadcast(scaling_topk, 0)
        else:
            scaling_topk = torch.IntTensor([[0] * n_var for _ in range(self.sample_num_scaling)]).cuda()
            dist.broadcast(scaling_topk, 0)
            self.scaling_topk[scaling_stage] = scaling_topk.cpu().tolist()
    
    def grid_search_scaling(self, fixed_basenet=None, scaling_stage=1):
        # Optimize scaling index, fix basenet
        assert len(fixed_basenet) == len(self.model.module.net)

        scaling_eval_dict = {}
        
        scaling_list = []
        for d in range(len(self.model.module.depth_multiplier[scaling_stage])):
            for w in range(len(self.model.module.channel_multiplier[scaling_stage])):
                for r in range(len(self.model.module.resolution_multiplier[scaling_stage])):
                    if self.check_flops([fixed_basenet], {scaling_stage: [d, w, r]}, scaling_stage, self.err_scale):
                        scaling_list.append([d, w, r])
        len_scaling_list_fit = len(scaling_list)
        self.logger.info(scaling_list)

        score = torch.zeros(len(scaling_list))
        for i in range(self.rank, len(scaling_list), self.world_size):
            arch_str = str(scaling_list[i]).replace('\n', '')
            acc = self.eval_subnet_host(fixed_basenet, {scaling_stage: scaling_list[i]}, [scaling_stage], self.err_scale)
            logging.info('==rank{}== [{}/{}] evaluation basenet/scaling:{} prec@1:{}\n\n'.format(self.rank, i, 
                                                                                                len(scaling_list), 
                                                                                                arch_str, acc))
            score[i] = acc

        score = score.cuda()
        dist.all_reduce(score)
        score = score.cpu().tolist()

        for i in range(len_scaling_list_fit):
            arch_str = str(scaling_list[i]).replace('\n', '')
            if scaling_eval_dict.get(arch_str) is None:
                scaling_eval_dict[arch_str] = {'acc': score[i], 'arch': scaling_list[i]}

        if self.rank == 0:
            sorted_scaling = sorted(scaling_eval_dict.items(), key=lambda i: i[1]['acc'], reverse=True)
            scaling_topk = [x[1]['arch'] for x in sorted_scaling]
            self.logger.info('== search result ==')
            self.logger.info([[list(x[1]['arch']), x[1]['acc']] for x in sorted_scaling])
            self.logger.info('== best scaling ==')
            self.logger.info([list(x) for x in scaling_topk])
            self.scaling_topk[scaling_stage] = scaling_topk
            scaling_topk = torch.IntTensor(scaling_topk).cuda()
            dist.broadcast(scaling_topk, 0)
        else:
            scaling_topk = torch.IntTensor([[0] * 3 for _ in range(len_scaling_list_fit)]).cuda()
            dist.broadcast(scaling_topk, 0)
            self.scaling_topk[scaling_stage] = scaling_topk.cpu().tolist()

    def sample_scaling_(self, scaling_stage=1):
        # Optimize scaling index for all (sampling a fixed subset of) basenets

        scaling_eval_dict = {}
        n_offspring = None #40
        basenet_list = []
        for _ in range(self.n_basenet):
            basenet_list.append(self.get_init_basenet(flops=self.flops_constraint))


        basenet_list = torch.IntTensor(basenet_list).cuda()
        dist.broadcast(basenet_list, 0)
        basenet_list = basenet_list.cpu().tolist()

        # setup NAS search problem
        n_var = 3 # depth, width, resolution
        lb = np.zeros(n_var)  # left index of scaling multiplier
        ub = np.array([len(self.model.module.depth_multiplier[scaling_stage]) - 1, # right index of scaling multiplier
                        len(self.model.module.channel_multiplier[scaling_stage]) - 1,
                        len(self.model.module.resolution_multiplier[scaling_stage]) - 1], dtype=float) 

        nas_problem = NAS(n_var=n_var, n_obj=1, n_constr=0, lb=lb, ub=ub,
                eval_func=lambda scaling: self.eval_subnet_host(basenet_list, scaling, [scaling_stage], self.err_scale),
                result_dict=scaling_eval_dict, rank=self.rank, world_size=self.world_size)

        # configure the nsga-net method
        init_sampling = []
        for _ in range(self.pop_size_scaling):
            init_sampling.append(self.get_init_scaling(scaling_stage, basenet_list))
        method = engine.nsganet(pop_size=self.pop_size_scaling,
                                        n_offsprings=n_offspring,
                                        eliminate_duplicates=True,
                                        sampling=np.array(init_sampling, dtype=np.int32))

        res = minimize(nas_problem,
                           method,
                           callback=lambda algorithm: self.generation_callback(algorithm),
                           termination=('n_gen', self.n_gens_scaling))

        if self.rank == 0:
            sorted_scaling = sorted(scaling_eval_dict.items(), key=lambda i: i[1]['acc'], reverse=True)
            sorted_scaling_key = [x[1]['arch'] for x in sorted_scaling]
            scaling_topk = sorted_scaling_key[:self.sample_num_scaling]
            self.logger.info('== search result ==')
            self.logger.info([[list(x[1]['arch']), x[1]['acc']] for x in sorted_scaling])
            self.logger.info('== best scaling ==')
            self.logger.info([list(x) for x in scaling_topk])
            self.scaling_topk[scaling_stage] = scaling_topk
            scaling_topk = torch.IntTensor(scaling_topk).cuda()
            dist.broadcast(scaling_topk, 0)
        else:
            scaling_topk = torch.IntTensor([[0] * n_var for _ in range(self.sample_num_scaling)]).cuda()
            dist.broadcast(scaling_topk, 0)
            self.scaling_topk[scaling_stage] = scaling_topk.cpu().tolist()
    
    def grid_search_scaling_(self, scaling_stage=1):
        # Optimize scaling index, fix basenet
        scaling_eval_dict = {}
        basenet_list = []
        for _ in range(self.n_basenet):
            basenet_list.append(self.get_init_basenet(flops=self.flops_constraint))
        basenet_list = torch.IntTensor(basenet_list).cuda()
        dist.broadcast(basenet_list, 0)
        basenet_list = basenet_list.cpu().tolist()
        scaling_list = []
        for d in range(len(self.model.module.depth_multiplier[scaling_stage])):
            for w in range(len(self.model.module.channel_multiplier[scaling_stage])):
                for r in range(len(self.model.module.resolution_multiplier[scaling_stage])):
                    if self.check_flops(basenet_list, {scaling_stage: [d, w, r]}, scaling_stage, self.err_scale):
                        scaling_list.append([d, w, r])
        len_scaling_list_fit = len(scaling_list)
        self.logger.info(scaling_list)
        score = torch.zeros(len(scaling_list))
        for i in range(self.rank, len(scaling_list), self.world_size):
            arch_str = str(scaling_list[i]).replace('\n', '')
            acc = self.eval_subnet_host(basenet_list, {scaling_stage: scaling_list[i]}, [scaling_stage], self.err_scale)
            logging.info('==rank{}== [{}/{}] evaluation basenet/scaling:{} prec@1:{}\n\n'.format(self.rank, i, 
                                                                                                len(scaling_list), 
                                                                                                arch_str, acc))
            score[i] = acc

        score = score.cuda()
        dist.all_reduce(score)
        score = score.cpu().tolist()

        for i in range(len_scaling_list_fit):
            arch_str = str(scaling_list[i]).replace('\n', '')
            if scaling_eval_dict.get(arch_str) is None:
                scaling_eval_dict[arch_str] = {'acc': score[i], 'arch': scaling_list[i]}

        if self.rank == 0:
            sorted_scaling = sorted(scaling_eval_dict.items(), key=lambda i: i[1]['acc'], reverse=True)
            scaling_topk = [x[1]['arch'] for x in sorted_scaling]
            self.logger.info('== search result ==')
            self.logger.info([[list(x[1]['arch']), x[1]['acc']] for x in sorted_scaling])
            self.logger.info('== best scaling ==')
            self.logger.info([list(x) for x in scaling_topk])
            self.scaling_topk[scaling_stage] = scaling_topk
            scaling_topk = torch.IntTensor(scaling_topk).cuda()
            dist.broadcast(scaling_topk, 0)
        else:
            scaling_topk = torch.IntTensor([[0] * 3 for _ in range(len_scaling_list_fit)]).cuda()
            dist.broadcast(scaling_topk, 0)
            self.scaling_topk[scaling_stage] = scaling_topk.cpu().tolist()

    def check_flops(self, basenet_list, scaling, scaling_stage, err):
        flops = []
        for basenet in basenet_list:
            subnet = self.generate_subnet_(basenet, scaling, scaling_stage=scaling_stage)
            flops_base = self.count_flops(basenet + [1., 224]) * 2 ** scaling_stage
            flops_scaled = self.count_flops(subnet)
            err_ = abs(flops_base - flops_scaled)
            flops.append(err_)
        err = err * 2 ** scaling_stage
        if sum(flops) / len(flops) <= err:
            return True
        else:
            return False

    def get_init_basenet(self, flops=400e6):
        depth_stage = self.model.module.depth_stage# [[base_max_depth, max_depth], ...]
        flag = True
        while flag:
            subnet = []
            for block in self.model.module.net:
                idx = random.randint(1, len(block) - 1) if len(block) > 1 else 0
                subnet.append(idx)# only op now
            
            id = []# base->1, id->0
            for i in depth_stage:
                if i[0] == i[1]:
                    id.append(1)
                else:
                    n_base = random.randint(1, i[0])
                    id += [1] * n_base + [0] * (i[1] - n_base)
            assert len(id) == len(subnet)
            subnet = list(map(lambda x, y: x * y, subnet, id))
            flops_ = self.count_flops(subnet + [1., 224])
            # print(flops_)
            if abs(flops - flops_) <= self.err_base:
                flag = False
        return subnet
    
    def get_init_scaling(self, scaling_stage, basenet_list):
        flops_constraint_ = self.flops_constraint * (2 ** scaling_stage)
        flag = True
        while flag:
            scaling = [random.randint(0, len(self.model.module.depth_multiplier[scaling_stage]) - 1), 
                        random.randint(0, len(self.model.module.channel_multiplier[scaling_stage]) - 1), 
                        random.randint(0, len(self.model.module.resolution_multiplier[scaling_stage]) - 1)]

            # flops = []
            # for basenet in basenet_list:
            #     subnet = self.generate_subnet_(basenet, scaling, scaling_stage=scaling_stage)
            #     flops.append(self.count_flops(subnet))
            # flops = sum(flops) / len(flops)
            # # print(flops)
            # err = self.err_base * 2 ** scaling_stage
            # if -err <= (flops_constraint_ - flops) <= err:
            if self.check_flops(basenet_list, {scaling_stage: scaling}, scaling_stage, self.err_scale):
                flag = False
        return scaling # [depth, width, resolution]

    def sample(self):
        admm_iter = self.admm_iter
        if self.rank == 0:
            self.logger.info(self.start_step)
            self.logger.info('Depth multipliers: '      + str(self.model.module.depth_multiplier))
            self.logger.info('Channel multipliers: '    + str(self.model.module.channel_multiplier))
            self.logger.info('Resolution multipliers: ' + str(self.model.module.resolution_multiplier))

        if self.start_step.split('-')[0] == 'start':
            # initially sample scaling strategy with evaluating all (a fixed subset of) basenets
            for s in range(int(self.start_step.split('-')[1]), self.scaling_stage + 1):
                # self.sample_scaling_(scaling_stage=s)
                # print('==rank{}=={}'.format(self.rank, 1))
                self.grid_search_scaling_(scaling_stage=s)
                if s < self.scaling_stage:
                    self.save_checkpoint('start-' + str(s + 1), admm_iter)
                else:
                    self.save_checkpoint('basenet', admm_iter)
        elif self.start_step.split('-')[0] == 'scaling':
            # load checkpoint from sampling scaling
            for s in range(int(self.start_step.split('-')[1]), self.scaling_stage + 1):
                # self.sample_scaling(fixed_basenet=[int(x) for x in self.basenet_topk[0]], scaling_stage=s)
                self.grid_search_scaling(fixed_basenet=[int(x) for x in self.basenet_topk[0]], scaling_stage=s)
                if s < self.scaling_stage:
                    self.save_checkpoint('scaling-' + str(s + 1), admm_iter)
                else:
                    self.save_checkpoint('basenet', admm_iter)

        # iteratively sample basenet and scaling strategy
        for _ in range(self.admm_iter):
            admm_iter -= 1
            scaling_top1 = {}
            for s in range(1, self.scaling_stage + 1):
                scaling_top1[s] = [int(x) for x in self.scaling_topk[s][0]]
            self.sample_basenet(scaling_top1, self.pop_size, self.n_gens, self.sample_num)
            self.save_checkpoint('scaling-1', admm_iter)

            for s in range(1, self.scaling_stage + 1):
                # self.sample_scaling(fixed_basenet=[int(x) for x in self.basenet_topk[0]], scaling_stage=s)
                self.grid_search_scaling(fixed_basenet=[int(x) for x in self.basenet_topk[0]], scaling_stage=s)
                if s < self.scaling_stage:
                    self.save_checkpoint('scaling-' + str(s + 1), admm_iter)
                else:
                    self.save_checkpoint('basenet', admm_iter)

        # finished
        if self.rank == 0:
            self.logger.info('\n\n\n')
            self.logger.info('========== final best scaling ==========')
            for s in range(1, self.scaling_stage + 1):
                self.logger.info('==scaling stage {}=={}'.format(s, [list(x) for x in self.scaling_topk[s]]))

            self.logger.info('========== final best basenet ==========')
            self.logger.info([list(x) for x in self.basenet_topk])

            self.logger.info('[END] Finish sampling.')
    
    def save_checkpoint(self, start_step, admm_iter):
        arch_topk = {'basenet_topk': str(self.basenet_topk), 
                        'scaling_topk': str(self.scaling_topk), 
                        'start_step': start_step,
                        'admm_iter': admm_iter}
        if self.rank == 0:
            with open(self.arch_topk_path, 'w') as f:
                yaml.dump(arch_topk, f)

    def generation_callback(self, algorithm):
        gen = algorithm.n_gen
        pop_var = algorithm.pop.get("X")
        pop_obj = algorithm.pop.get("F")
        self.logger.info(f'==Finished generation: {gen}')
    
    def count_flops_conv(self, inp, oup, kernel_size, padding, stride, groups, input_shape):
        c, w, h = input_shape
        c = oup
        w = (w + padding * 2 - kernel_size + 1) // stride
        h = (h + padding * 2 - kernel_size + 1) // stride
        flops = inp * oup * w * h // groups * kernel_size * kernel_size
        output_shape = [c, w, h]
        return flops, output_shape
    
    def count_flops_fc(self, inp, oup):
        return inp * oup

    def count_flops_op(self, op, c_m, input_shape):
        # Compute FLOPs of an operation
        flops = []
        if isinstance(op, InvertedResidual):
            inp = int(math.ceil(op.inp_base * c_m))
            hid = int(round(inp * op.t))
            oup = int(math.ceil(op.oup_base * c_m))
            if op.t == 1:
                #dw
                flops_, input_shape = self.count_flops_conv(hid, hid, op.k, op.k // 2, op.stride, 
                                                            hid, input_shape)
                flops.append(flops_)
                #se
                if op.use_se:
                    flops_ = self.count_flops_fc(hid, hid // op.se.reduction)
                    flops.append(flops_)
                    flops_ = self.count_flops_fc(hid // op.se.reduction, hid)
                    flops.append(flops_)
                #pw
                flops_, input_shape = self.count_flops_conv(hid, oup, 1, 0, 1, 1, input_shape)
                flops.append(flops_)
            else:
                #pw
                flops_, input_shape = self.count_flops_conv(inp, hid, 1, 0, 1, 1, input_shape)
                flops.append(flops_)
                #dw
                flops_, input_shape = self.count_flops_conv(hid, hid, op.k, op.k // 2, op.stride, hid, 
                                                            input_shape)
                flops.append(flops_)
                #se
                if op.use_se:
                    flops_ = self.count_flops_fc(hid, hid // op.se.reduction)
                    flops.append(flops_)
                    flops_ = self.count_flops_fc(hid // op.se.reduction, hid)
                    flops.append(flops_)
                #pw
                flops_, input_shape = self.count_flops_conv(hid, oup, 1, 0, 1, 1, input_shape)
                flops.append(flops_)
        elif isinstance(op, Conv2d):
            if op.k == 3:
                inp = op.inp_base
                oup = int(math.ceil(op.oup_base * c_m))
            elif op.k == 1:
                oup = op.oup_base
                inp = int(math.ceil(op.inp_base * c_m))
            flops_, input_shape = self.count_flops_conv(inp, oup, op.k, op.k//2, op.stride, 1, 
                                                        input_shape)
            flops.append(flops_)
        elif isinstance(op, FC):
            flops_ = self.count_flops_fc(op.inp, op.oup)
            flops.append(flops_)
        else:
            flops.append(0.)

        return int(sum(flops)), input_shape
    

    def count_flops(self, subnet, input_size=[3, 224, 224]):
        # subnet: list, [op, ... , c_m, r]
        # print(subnet)
        subnet_op = subnet[:-2]
        c_m = subnet[-2]
        input_size = [input_size[0], subnet[-1], subnet[-1]]
        subnet_flops = []
        for op, layer in zip(subnet_op, self.model.module.net):
            layer_flops, input_size = self.count_flops_op(layer[op], c_m, input_size)
            subnet_flops.append(layer_flops)
        return sum(subnet_flops)


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, n_var=20, n_obj=1, n_constr=0, lb=None, ub=None, eval_func=None, result_dict=None,
                    rank=0, world_size=1):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.eval_func = eval_func
        self.result_dict = result_dict
        self.rank = rank
        self.world_size = world_size

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.IntTensor(x).cuda()
        dist.broadcast(x, 0)
        x = x.cpu().numpy()

        objs = torch.zeros(x.shape[0], self.n_obj)
        for i in range(self.rank, x.shape[0], self.world_size):
            # all objectives assume to be MINIMIZED !!!!!
            arch_str = str(x[i].tolist()).replace('\n', ' ')
            if self.result_dict.get(arch_str) is not None:
                acc = self.result_dict[arch_str]['acc']
            else:
                acc = self.eval_func(x[i].tolist())

            logging.info('==rank{}== [{}/{}] evaluation basenet/scaling:{} prec@1:{}\n\n'.format(self.rank, i, x.shape[0],  
                                                                                                            arch_str, acc))
            objs[i, 0] = 100 - acc  # performance['valid_acc']
            # objs[i, 1] = 10  # performance['flops']
            self._n_evaluated += 1
        
        objs = objs.cuda()
        dist.all_reduce(objs)
        objs = objs.cpu().numpy().astype(np.float64)
        out["F"] = objs

        for i in range(x.shape[0]):
            arch_str = str(x[i].tolist()).replace('\n', '')
            if self.result_dict.get(arch_str) is None:
                self.result_dict[arch_str] = {'acc': 100 - objs[i, 0], 'arch': x[i].tolist()}
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints
