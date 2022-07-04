import mc

if __name__ == '__main__':
    from base_mc_dataset import BaseMcDataset, img_loader
else:
    from .base_mc_dataset import BaseMcDataset, img_loader
import os
import json


class NormalDataset(BaseMcDataset):
    def __init__(self, cfg, transform=None, preprocessor='cv'):
        super().__init__(preprocessor)
        self.prefix = cfg['prefix']
        self.transform = transform
        self.cfg = cfg
        self.parse_json_()

    def parse_json_(self):
        # print('loading json file: {}'.format(self.cfg['json_path']))
        jdata = json.load(open(self.cfg['json_path'], 'r'))
        self.key = list(jdata.keys())[0]
        self.num = len(jdata[self.key])
        # print('building dataset from %s: %d images' % (self.prefix, self.num))

        self.metas = []
        for i in range(self.num):
            path = jdata[self.key][i]['img_info']['filename']
            label = jdata[self.key][i]['annos'][self.cfg.get('task', 'classification')][
                self.cfg.get('type', 'imagenet')]
            self.metas.append((path, int(label)))

    def __getitem__(self, idx):
        filename = self.prefix + '/' + self.metas[idx][0]
        cls = self.metas[idx][1]
        # memcached
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        img = img_loader(value_str, self.preprocessor)
        # transform
        if self.transform is not None:
            if isinstance(self.transform, dict):
                img_dict = {}
                for t in self.transform.keys():
                    img_dict[t] = self.transform[t](**{'image': img})['image']
                img = img_dict
            else:
                img = self.transform(**{'image': img})
                img = img['image']
        return img, cls

