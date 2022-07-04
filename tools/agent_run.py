import os
import sys
sys.path.append(os.getcwd())
import yaml
from core.agent.nas_agent import NASAgent
import torch
import numpy as np


if __name__ == '__main__':
    # manual seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    config = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader)
    agent = NASAgent(config)
    agent.run()
