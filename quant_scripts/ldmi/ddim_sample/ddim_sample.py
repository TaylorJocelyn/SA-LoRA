import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# from taming.models import vqgan

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# seed = 2000
# torch.cuda.manual_seed(seed)
# torch.manual_seed(seed)

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if __name__ == '__main__':
    torch.cuda.set_device(torch.device('cuda:0'))
    model = get_model()
    sampler = DDIMSampler(model)

    # classes = [25, 187, 448, 992]   # define classes to be sampled here
    # classes = [88, 301, 139, 871]
    # classes = [867, 187, 946, 11]   # define classes to be sampled here
    classes = [279, 139, 387, 195, 301, 985, 417, 975]

    n_samples_per_class = 6

    # ddim_steps = 20
    # ddim_eta = 0.0
    # scale = 3.0
    # torch.manual_seed(3343)

    ddim_steps = 12
    ddim_eta = 0.0
    scale = 1.0

    # from thop import profile
    # xt = torch.randn(1, 3, 64, 64).cuda()
    # t = torch.randn(1).cuda()
    # contex = torch.randn(1, 1, 512).cuda()
    # flops, params = profile(model.model.diffusion_model, (xt, t, contex))
    # params = torchprofile.profile_params(model)
    # flops = flops * 32 * 32 / 10**12
    # print(f"FLOPs: {flops}")
    # print(f"Parameters: {params}")

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples_per_class,
                                                shape=[3, 64, 64],
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc, 
                                                eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    image_to_save.save("reproduce/ldmi/sample/ddim_sample_scale{}_eta{}_{}steps.png".format(scale, ddim_eta, ddim_steps))
    # torch.save(all_samples, "reproduce/scale{}_eta{}_step{}/imagenet/save_data/cwi_fp_{}.pth".format(scale, ddim_eta, ddim_steps, seed))
