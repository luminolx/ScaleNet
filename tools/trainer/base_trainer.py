
class BaseTrainer(object):
    ''' Base Trainer class '''

    def __init__(self, dataloader, model, optimizer,
                 lr_scheduler, print_freq,
                 save_path, snapshot, logger):
        self.dataloader = dataloader
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.print_freq = print_freq
        self.snapshot_freq = snapshot
        self.save_path = save_path
        self.logger = logger
        self.cur_iter = 1
        self.cur_epoch = 1

    def train(self):
        raise RuntimeError('BaseTrainer cannot train')

    def save(self):
        raise RuntimeError('BaseTrainer cannot save search_space')

    def load(self, ckpt_path):
        raise RuntimeError('BaseTrainer cannot load search_space and optimizer')

    def show_attr(self):
        for name, value in vars(self).items():
            print(name, value)

    def show_task_key(self):
        print(self.task_key)
