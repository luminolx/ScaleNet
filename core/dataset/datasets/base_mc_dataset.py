import mc
import numpy as np
from torch.utils.data import Dataset
import io
import cv2
from PIL import Image


def img_loader(img_str, preprocessor='cv'):
    if preprocessor == 'pil':
        buff = io.BytesIO(img_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
    elif preprocessor == 'cv':
        img_array = np.frombuffer(img_str, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        raise ValueError('no such processor')
    return img

class BaseMcDataset(Dataset):
    def __init__(self, preprocessor='cv'):
        self.initialized = False
        self.num = 0
        self.preprocessor = preprocessor

    def __len__(self):
        return self.num

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)
            self.initialized = True

    def __getitem__(self, idx):
        raise RuntimeError("BaseMcDataset is unabled to be indexed")
