import time
from os.path import join, exists

import torch
from torch import distributed as dist
import core.dataset.build_dataloader as BD
from tools.trainer.base_trainer import BaseTrainer

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class ImagenetTrainer(BaseTrainer):
    ''' Imagenet Trainer
    requires attrs:
        - in Base Trainer
        (train) search_space, optimizer, lr_scheduler, dataloader, cur_iter
        (log) logger
        (save) print_freq, snapshot_freq, save_path

        - in Customized Trainer
        (dist) rank, world_size
        (train) max_iter
        (time) data_time, forw_time, batch_time
        (loss&acc) <task_name>_disp_loss, <task_name>_disp_acc
        (task) task_key, task_training_shapes
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        for k, v in kwargs.items():
            setattr(self, k, v)
        # check customized trainer has all required attrs
        self.required_atts = ('rank', 'world_size', 'max_iter',
                              'data_time', 'forw_time', 'batch_time', 'bckw_time', 'step_time')
        for att in self.required_atts:
            if not hasattr(self, att):
                raise RuntimeError(f'ImagenetTrainer must has attr: {att}')
        self.task_key = 'classification'
        self.logger.info("task key: %s" % (self.task_key))
        if not hasattr(self, 'disp_acc_top1'):
            raise RuntimeError(f'ImagenetTrainer must has attr: disp_acc_top1')
        if not hasattr(self, 'disp_acc_top5'):
            raise RuntimeError(f'ImagenetTrainer must has attr: disp_acc_top5')
        self.logger.info(f'[rank{self.rank}]ImagenetTrainer build done.')
        if self.rank == 0:
            self.logger.info(self.model)

    def train(self):
        self.model.train()
        if self.rank == 0:
            self.logger.info('Start training...')
            self.logger.info(f'Loading classification data')
        loader_iter = iter(BD.DataPrefetcher(self.dataloader))
        input_all = {}
        end_time = time.time()
        self.optimizer.zero_grad()
        while self.cur_iter <= self.max_iter:
            self.lr_scheduler.step(self.cur_iter)
            tmp_time = time.time()
            images, target = next(loader_iter)
            flag_epoch_end = False
            if images is not None and isinstance(images, dict):
                for idx in images.keys():
                    if images[idx] is None:
                        flag_epoch_end = True
                        break
            if images is None or flag_epoch_end:
                epoch = int(self.cur_iter / len(self.dataloader))
                if self.rank == 0:
                    self.logger.info('classification epoch-{} done at iter-{}'.format(epoch, 
                                                                                    self.cur_iter))
                self.dataloader.sampler.set_epoch(epoch)
                loader_iter = iter(BD.DataPrefetcher(self.dataloader))
                images, target = loader_iter.next()

            input_all['images'] = images
            input_all['labels'] = target

            self.data_time.update(time.time() - tmp_time)
            tmp_time = time.time()
            output = self.model(input_all, c_iter=self.cur_iter)
            self.forw_time.update(time.time() - tmp_time)

            loss = output['loss']
            if not self.model.module.asyn and self.model.module.n > 0:
                loss /= self.model.module.n
            reduced_loss = loss.data.clone() / self.world_size
            dist.all_reduce(reduced_loss)
            self.disp_loss.update(reduced_loss.item())
            if self.task_has_accuracy:
                prec1, prec5 = output['accuracy']
                reduced_prec1 = prec1.clone() / self.world_size
                dist.all_reduce(reduced_prec1)
                reduced_prec5 = prec5.clone() / self.world_size
                dist.all_reduce(reduced_prec5)
                self.disp_acc_top1.update(reduced_prec1.item())
                self.disp_acc_top5.update(reduced_prec5.item())

            tmp_time = time.time()
            if has_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.bckw_time.update(time.time() - tmp_time)
            # torch.cuda.empty_cache()
            if not self.model.module.asyn and self.model.module.n > 0:
                for _ in range(1, self.model.module.n):
                    output = self.model(input_all, subnet=output['subnet'], c_iter=self.cur_iter)
                    loss = output['loss'] / self.model.module.n
                    reduced_loss = loss.data.clone() / self.world_size
                    dist.all_reduce(reduced_loss)
                    self.disp_loss.update(reduced_loss.item())
                    if has_apex:
                        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
            
            tmp_time = time.time()
            self.optimizer.step()
            self.step_time.update(time.time() - tmp_time)
            self.optimizer.zero_grad()
            self.batch_time.update(time.time() - end_time)
            end_time = time.time()
            # vis loss
            if self.cur_iter % self.print_freq == 0 and self.rank == 0:
                self.logger.info('Iter: [{0}/{1}] '
                                 'BatchTime {batch_time.avg:.4f} | '
                                 'DataTime {data_time.avg:.4f} | '
                                 'ForwardTime {forw_time.avg:.4f} | '
                                 'BackwardTime {bckw_time.avg:.4f} | '
                                 'StepTime {step_time.avg:.4f} | '
                                 'Total {batch_time.all:.2f} hrs | '
                                 'Loss {loss.avg:.4f} | '
                                 'Prec@1 {top1.avg:.4f} | '
                                 'Prec@5 {top5.avg:.4f} | '
                                 'LR {lr:.6f} | ETA {eta:.2f} hrs'.format(
                    self.cur_iter, self.max_iter,
                    batch_time=self.batch_time,
                    data_time=self.data_time,
                    forw_time=self.forw_time,
                    bckw_time=self.bckw_time,
                    step_time=self.step_time,
                    loss=self.disp_loss,
                    top1=self.disp_acc_top1,
                    top5=self.disp_acc_top5,
                    lr=self.lr_scheduler.get_lr()[0],
                    eta=self.batch_time.avg * (self.max_iter - self.cur_iter) / 3600))

            # save search_space
            if self.cur_iter % self.snapshot_freq == 0:
                if self.rank == 0:
                    self.save()
            self.cur_iter += 1

        # finish training
        self.logger.info('Finish training {} iterations.'.format(self.cur_iter))

    def save(self):
        ''' save search_space '''
        path = join(self.save_path, 'iter_{}_ckpt.pth.tar'.format(self.cur_iter))
        latest_path = join(self.save_path, 'latest.pth.tar')
        if has_apex:
            torch.save({'step': self.cur_iter, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'amp': amp.state_dict()}, latest_path)
        else:
            torch.save({'step': self.cur_iter, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, latest_path)
        self.logger.info('[rank{}]Saved search_space to {}.'.format(self.rank, latest_path))

        if self.cur_iter % 10000 == 0:
            if has_apex:
                torch.save({'step': self.cur_iter, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), 'amp': amp.state_dict()}, path)
            else:
                torch.save({'step': self.cur_iter, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}, path)
            self.logger.info('[rank{}]Saved search_space to {}.'.format(self.rank, path))

    def load(self, ckpt_path):
        ''' load search_space and optimizer '''

        def map_func(storage, location):
            return storage.cuda()

        assert exists(ckpt_path), f'{ckpt_path} not exist.'
        ckpt = torch.load(ckpt_path, map_location=map_func)
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        ckpt_keys = set(ckpt['state_dict'].keys())
        own_keys = set(self.model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            if self.rank == 0:
                self.logger.info(f'**missing key while loading search_space**: {k}')
                raise RuntimeError(f'**missing key while loading search_space**: {k}')

        # load optimizer
        self.cur_iter = ckpt['step'] + 1
        epoch = int(self.cur_iter / len(self.dataloader))
        self.dataloader.sampler.set_epoch(epoch)
        if self.rank == 0:
            self.logger.info('load [resume] search_space done, '
                             f'current iter is {self.cur_iter}')
        
        # load amp
        if has_apex and 'amp' in ckpt:
            amp.load_state_dict(ckpt['amp'])
