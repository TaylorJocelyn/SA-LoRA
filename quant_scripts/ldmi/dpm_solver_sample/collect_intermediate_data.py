'''
To collect input data, remember to uncomment line 987-988 in ldm/models/diffusion/ddpm.py and comment them after finish collecting.
'''
import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.dpm_solver_pytorch import DpmSolverSampler
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
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
    device = torch.device('cuda:0')
    model = get_model()
    sampler = DpmSolverSampler(model, algorithm_type='dpmsolver', model_type='noise', device=device)

    batch_size = 8

    sample_steps = 250
    scale = 1.5

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(batch_size*[1000]).to(model.device)}
                )
            xc = torch.randint(0,1000,(batch_size,)).to(model.device)
            c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
            
            x_interms, t_interms = sampler.collect_intermediate_data(S=sample_steps,
                                            conditioning=c,
                                            batch_size=batch_size,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc)

            # x_samples_ddim = model.decode_first_stage(samples_ddim)
            # x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
            #                             min=0.0, max=1.0)
            # all_samples.append(x_samples_ddim)

    ## save diffusion input data
    import ldm.globalvar as globalvar   
    input_list = globalvar.getInputList()
    save_dir = "reproduce/ldmi/data"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'DiffusionSolverPlusInput_{}steps.pth'.format(sample_steps))
    torch.save(input_list, save_path)
    sys.exit(0)
