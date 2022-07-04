from abc import abstractmethod
from core.model.net import Net
from tools.eval.base_tester import BaseTester
from torch import distributed as dist
import os
from core.utils.logger import create_logger
import time

class BaseSampler:
    def __init__(self, tester: BaseTester, **kwargs):
        self.tester = tester
        self.model = self.tester.model
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.flops_min = kwargs.pop('flops_min', 0)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        
        # build logger
        now = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        if now != '':
            now = '_' + now
        self.logger = create_logger('', 
            '{}/log/'.format(self.cfg_stg_sample['save_path']) + '/log_sample{}.txt'.format(now))

    def forward_subnet(self, input):
        """
        run one step
        :return:
        """
        pass
    
    @abstractmethod
    def eval_subnet(self):
        """
        Do eval for a model family with basenet and scaling.
        :return: a score for the family
        """

    @abstractmethod
    def generate_subnet(self):
        """
        generate one subnet
        :return: block indexes for each choice block
        """

    @abstractmethod
    def sample(self):
        """
        sample basenet and scaling
        :return: None
        """

