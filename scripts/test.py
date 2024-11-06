import torch
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from quant_scripts.ddim.quant_dataset import IntermediateInputDataset
    from torch.utils.data import DataLoader
    from quant_scripts.ddim.quant_dataset import get_train_samples
    # from quant_scripts.ddim.dntc_quant_sample import dntc_sample

    import torch

    data = torch.load('/home/zq/EfficientDM/reproduce/ldmi/data/DiffusionInput_250steps.pth')

    data = torch.load('/home/zq/EfficientDM/reproduce/ddim/data/cifar10_20step_uniform_generalized_inputdata.pth', map_location='cpu')
    # data_list = torch.load(data_path, map_location='cpu') ## its a list of tuples of tensors
    # print(data_list.keys())
    # print(data_list)
    xs = data['x']
    ts = data['t']

    all_samples = []

    for i in range(4):
        ls = []
        for j in range(5):
            x_samples_ddim = xs[i*5+j]
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
            ls.append(x_samples_ddim)

        ls = torch.cat(ls, dim=0)
        all_samples.append(ls)

    # # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=5)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    image_to_save.save('reproduce/ldmi/sample/test.jpg')
