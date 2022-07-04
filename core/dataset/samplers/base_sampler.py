from torch.utils.data.sampler import Sampler


class BaseIterSampler(Sampler):
    def __init__(self, dataset, total_iter, batch_size, latest_iter=-1):
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.latest_iter = latest_iter
        self.total_size = self.total_iter * self.batch_size
        self.call = 0

    def __iter__(self):
        raise RuntimeError('unable to get iterator from BaseIterSampler')

    def __len__(self):
        return self.total_size
