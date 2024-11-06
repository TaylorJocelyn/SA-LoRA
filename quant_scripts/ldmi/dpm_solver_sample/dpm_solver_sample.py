import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
# from taming.models import vqgan

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver_pytorch import DPM_Solver, NoiseScheduleVP

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
# seed = 2000
# torch.cuda.manual_seed(seed)
# torch.manual_seed(seed)

from torch.utils.tensorboard import SummaryWriter

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

if __name__ == '__main__':
   
    device = torch.device('cuda:0')
    model = get_model()
    model.to(device)
    

    # classes = [25, 187, 448, 992]   # define classes to be sampled here
    # classes = [88, 301, 139, 871]
    # classes = [867, 187, 946, 11]   # define classes to be sampled here
    classes = [279, 139, 387, 195, 301, 985, 417, 975]

    n_samples_per_class = 6

    # sample_steps = 40
    # ddim_eta = 0.0
    # scale = 3.0
    # torch.manual_seed(3343)

    sample_steps = 200
    # ddim_eta = 0.0
    scale = 2.0
    solver_order = 1

    from ldm.models.diffusion.dpm_solver_pytorch import DpmSolverSampler
    from ldm.models.diffusion.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper
    noise_schedule = NoiseScheduleVP(schedule='discrete', betas=model.betas)
    
    sampler = DpmSolverSampler(model, algorithm_type='dpmsolver', model_type='noise', device=device)

    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )
            
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {sample_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                samples_ddim, _ = sampler.sample(
                                    S=sample_steps, # sample timestep
                                    batch_size=n_samples_per_class,
                                    shape=[3, 64, 64],
                                    conditioning=c,
                                    order=solver_order,
                                    skip_type='quad',
                                    method='singlestep',
                                    verbose=False,
                                    x_T=None,
                                    log_every_t=100,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc,
                                    return_intermediate=True
                )

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                            min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)

                # from torchviz import make_dot

                # # 假设你的模型是 model，输入是 input
                # dot = make_dot(samples_ddim, params=dict(model.model.diffusion_model.named_parameters()))
                # dot.render("reproduce/ldmi/pic/graph", format="png")  

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)
    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    image_to_save.save("reproduce/ldmi/sample/dpm_solver_{}_coupled_{}steps_scale{}.png".format(solver_order, sample_steps, scale))
    # torch.save(all_samples, "reproduce/scale{}_eta{}_step{}/imagenet/save_data/cwi_fp_{}.pth".format(scale, ddim_eta, ddim_steps, seed))