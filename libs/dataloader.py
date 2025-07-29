import torch

from torchvision.datasets import DatasetFolder
from libs.utils import unlimited

class addNoise(object):
    def __init__(self, std=0):
        self.std = std

    def __call__(self, img):

        img = img / torch.max(img)

        if isinstance(self.std, tuple):
            std = torch.randint(self.std[0], self.std[1], (1,)).item()
        else:
            std = self.std

        noise = torch.randn_like(img) * std / 255.0
        y = img + noise
        return (img, y, std)


class Modulo(object):
    def __init__(self, bit_depth=10, std=0, train=False):

        self.bit_depth = bit_depth
        self.std = std
        self.train = train
        self.relu = lambda x: torch.maximum(x, torch.zeros_like(x))

    def __call__(self, img):

        mx = 256

        if isinstance(self.std, tuple):
            std = torch.randint(self.std[0], self.std[1], (1,)).item()
        else:
            std = self.std

        max_val = 2**self.bit_depth - 1

        data_range = (max_val + 1) / mx

        img = img - img.min(dim=(-1), keepdim=True)[0].min(dim=(-2), keepdim=True)[0]
        img = img / img.max()

        if self.train:
            satfact = 0.2
            sat     = torch.rand(1, device=img.device) * satfact - (satfact/2)
            img = img * max_val * (1 + sat)
        else:
            img = img * max_val
            

        std_scaled = std / mx


        img = img  / mx
        
        img_noisy = img + torch.randn_like(img) * std_scaled
        img_noisy = self.relu(img_noisy)
        modulo = unlimited(img_noisy, 1.0)

        modulo = modulo.float()
        std = torch.tensor(std, dtype=torch.float32, device=img.device)

        return (img, modulo, std)


class ModuloDataset(DatasetFolder):
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        super().__init__(root, loader, extensions, transform, target_transform)

        self.data = None
        self.pre_load()

    def pre_load(self):
        self.data = [self.loader(path) for path, _ in self.samples]

    def __getitem__(self, index):

        _, target = self.samples[index]

        sample = self.data[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
