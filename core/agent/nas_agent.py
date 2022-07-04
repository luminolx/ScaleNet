from core.model.net import Net
from core.searcher import build_searcher
from core.sampler import build_sampler
from core.utils.lr_scheduler import IterLRScheduler
from core.utils.optimizer import build_optimizer
from core.dataset.build_dataloader import build_dataloader
from tools.eval.build_tester import build_tester
from tools.trainer import build_trainer
import torch.nn as nn

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
    has_apex = True
except ImportError:
    from torch.nn.parallel import DistributedDataParallel as DDP
    has_apex = False
import time
from os.path import join, exists
import os
from core.utils.flops import count_flops
import logging
import torch
from tools.dist_init import dist_init

class FormatterNoInfo(logging.Formatter):
    def __init__(self, fmt='%(levelname)s: %(message)s'):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        if record.levelno == logging.INFO:
            return str(record.getMessage())
        return logging.Formatter.format(self, record)

class NASAgent:
    def __init__(self, config):
        self.cfg_net = config.pop('model')
        self.cfg_search = config.pop('search')
        self.cfg_sample = config.pop('sample')

    def run(self):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(FormatterNoInfo())
        logging.root.addHandler(console_handler)
        logging.root.setLevel(logging.INFO)

        #search config
        cfg_search_searcher = self.cfg_search.pop('searcher')
        cfg_search_stg = self.cfg_search.pop('strategy')
        cfg_search_data = self.cfg_search.pop('data')
        #sample config
        cfg_sample_sampler = self.cfg_sample.pop('sampler')
        cfg_sample_stg = self.cfg_sample.pop('strategy')
        cfg_sample_data = self.cfg_sample.pop('data')

        self.rank, self.local_rank, self.world_size = dist_init()
        print('==rank{}==local rank{}==world size{}'.format(self.rank, self.local_rank, self.world_size))

        torch.manual_seed(42 + self.rank)

        # build model
        self._build_model(self.cfg_net, cfg_search_searcher)
        # search
        if self.cfg_search['flag']:
            if self.rank == 0:
                if not exists(join(cfg_search_stg['save_path'], 'checkpoint')):
                    os.makedirs(join(cfg_search_stg['save_path'], 'checkpoint'))
                if not exists(join(cfg_search_stg['save_path'], 'events')):
                    os.makedirs(join(cfg_search_stg['save_path'], 'events'))
                if not exists(join(cfg_search_stg['save_path'], 'log')):
                    os.makedirs(join(cfg_search_stg['save_path'], 'log'))

            self._build_searcher(cfg_search_searcher, cfg_search_data, cfg_search_stg)
            self.search()

        # sample
        if self.cfg_sample['flag']:
            if not exists(cfg_sample_stg['save_path']):
                os.makedirs(cfg_sample_stg['save_path'])
            self._build_sampler(cfg_sample_sampler, cfg_sample_data, cfg_sample_stg, self.cfg_net, 
                                cfg_search_searcher)
            self.sample()
            self.subnet_candidates = self.sampler.generate_subnet()

    def _build_model(self, cfg_net, cfg_sample_sampler):
        self.model = Net(cfg_net, cfg_sample_sampler).cuda()

    def _build_searcher(self, cfg_searcher, cfg_data_search, cfg_stg_search):
        self.search_dataloader = build_dataloader(cfg_data_search, cfg_searcher)

        opt = build_optimizer(self.model, cfg_stg_search['optimizer'])
        if has_apex:
            self.model, opt = amp.initialize(self.model, opt, opt_level='O1', min_loss_scale=2.**10)
            if self.local_rank == 0:
                logging.info('NVIDIA APEX installed. AMP on.')
            self.model = DDP(self.model, delay_allreduce=True)
        else:
            if self.local_rank == 0:
                logging.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        lr_scheduler = IterLRScheduler(opt, **cfg_stg_search['lr_scheduler'])

        for searcher_type, start_iter in zip(cfg_searcher['type'], cfg_searcher['start_iter']):
            searcher = build_searcher(searcher_type, cfg_searcher, 
                                        **cfg_searcher.get(searcher_type, {}))
            self.model.module.add_searcher(searcher, start_iter)

        self.search_trainer = build_trainer(cfg_stg_search, self.search_dataloader, self.model,
                                            opt, lr_scheduler, 
                                            time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        load_path = join(cfg_stg_search['save_path'], 'checkpoint', cfg_stg_search['load_name'])
        if cfg_stg_search.get('resume', False) and os.path.exists(load_path):
            self.search_trainer.load(load_path)

    def _build_sampler(self, cfg_sampler, cfg_data_sample, cfg_stg_sample, cfg_net, cfg_search_searcher):
        if has_apex:
            if self.local_rank == 0:
                logging.info('NVIDIA APEX installed. AMP on.')
            self.model = DDP(self.model, delay_allreduce=True)
        else:
            if self.local_rank == 0:
                logging.info("Using torch DistributedDataParallel. Install NVIDIA Apex for Apex DDP.")
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.tester = build_tester(cfg_stg_sample, cfg_data_sample, self.model, cfg_search_searcher)
        self.sampler = build_sampler(cfg_sampler, self.tester, cfg_net, cfg_stg_sample)

    def search(self):
        self.search_trainer.train()
        if hasattr(self.model.module.searcher, 'get_best_arch'):
            self.subnet_candidates = self.model.module.searcher.get_best_arch()
        self.model.module.remove_searcher()

    def sample(self):
        self.sampler.sample()

    def statistic_flops(self):
        sampler = build_sampler({'type': 'random'}, self.model, None, None)
        logger = open('./statistic_flops.txt', 'a')
        for _ in range(12500):  # x8
            subnet = sampler.generate_subnet()
            flops = count_flops(self.model.module.net, subnet)
            print('{},{}'.format(subnet, flops))
            logger.write('{},{}\n'.format(subnet, flops))
            logger.flush()
