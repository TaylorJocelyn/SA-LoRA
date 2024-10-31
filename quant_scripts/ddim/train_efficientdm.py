'''
For training, remember to set self.total_steps = 100 in quant_scripts/quant_layer.py/TemporalActivationQuantizer
'''
import sys, time, datetime
sys.path.append(".")
sys.path.append('./taming-transformers')
# from taming.models import vqgan

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_trainer
import ldm.globalvar as globalvar

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

import argparse, yaml
import logging
from pytorch_lightning import seed_everything
from quant_scripts.ddim.quant_model import QuantModel_intnlora
from quant_scripts.ddim.quant_layer import QuantModule_intnlora, SimpleDequantizer

from ddim.models.diffusion import Model, Diffusion, DiffusionTrainer
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from torch.cuda import amp
import torchvision.utils as tvu

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        # print(_, ':', param.numel())
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "--cond", action="store_true",
        help="whether to use conditional guidance"
    )
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="quad",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--cali_sample_num",
        type=int,
        default=1024,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--solver_order",
        type=int,
        default=2,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--algorithm_type",
        type=str,
        default="dpmsolver",
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--weight_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--act_bit",
        type=int,
        default=8,
        help="int bit for activation quantization",
    )
    parser.add_argument(
        "--cali_ckpt_path",
        type=str,
        required=True,
        help="calibration dataset path",
    ),
    parser.add_argument(
        "--quant_init_ckpt",
        type=str,
        required=True,
        help="calibration dataset path",
    ),
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="train batch size",
    ),
    parser.add_argument(
        "--verbose", action="store_true",
        help="print out info like quantized model arch"
    )
    return parser
    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    parser = get_parser()
    args = parser.parse_args()

    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # fix random seed
    seed_everything(args.seed)

    # setup logger
    logdir = os.path.join(args.logdir, "samples", now)
    os.makedirs(logdir)
    args.logdir = logdir
    log_path = os.path.join(logdir, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    device = torch.device('cuda:0') 

    # initialize FP32 model 
    fp_model = Model(config)
    name = "cifar10"

    from ddim.functions.ckpt_util import get_ckpt_path
    ckpt = get_ckpt_path(f"ema_{name}")
    logger.info("Loading checkpoint {}".format(ckpt))
    fp_model.load_state_dict(torch.load(ckpt, map_location=device))
    
    fp_model.to(device)
    fp_model.eval()

    # initialize quant_model
    model = Model(config)
    name = "cifar10"

    from ddim.functions.ckpt_util import get_ckpt_path
    ckpt = get_ckpt_path(f"ema_{name}")
    logger.info("Loading checkpoint {}".format(ckpt))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    model.to(device)
    model.eval()

    from quant_scripts.ddim.quant_dataset import IntermediateInputDataset, get_train_samples
    from torch.utils.data import DataLoader

    n_bits_w, n_bits_a = args.weight_bit, args.act_bit

    dataset = IntermediateInputDataset(args.cali_ckpt_path)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    cali_images, cali_t = get_train_samples(data_loader, num_samples=args.cali_sample_num)
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel_intnlora(model=model, weight_quant_params=wq_params, act_quant_params=aq_params, num_steps=args.timesteps)
    qnn.to(device)
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    cali_images, cali_t = get_train_samples(data_loader, num_samples=args.cali_sample_num)
    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight, device=device)

    ckpt = torch.load(args.quant_init_ckpt, map_location='cpu')
    qnn.load_state_dict(ckpt, strict=False) ## no lora weight in ckpt

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.weight.data = module.weight.data.byte()
            module.intn_dequantizer.delta.data = module.weight_quantizer.delta ## share the same delta and zp
            module.intn_dequantizer.zero_point.data = module.weight_quantizer.zero_point
    
    print('First run to init model...') ## need run to init temporal act quantizer
    with torch.no_grad():
        _ = qnn(cali_images[:4].to(device), cali_t[:4].to(device))

    model = qnn

    for name, param in model.named_parameters():
        if 'lora' in name or 'delta' in name or 'zp_list' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in fp_model.named_parameters():
        param.requires_grad = False

    print_trainable_parameters(model)
    
    avg_delta_list = []
    from transformers import get_linear_schedule_with_warmup
    NUM_EPOCHS = 160
    firstone = True
    for name, module in model.named_modules():
        if isinstance(module, QuantModule_intnlora):
            if len(module.act_quantizer.delta_list) != args.timesteps:
                raise ValueError('Wrong act_quantizer.delta_list length')
            avg_delta = torch.sum(module.weight_quantizer.delta) / torch.numel(module.weight_quantizer.delta)

            params = [param for name, param in module.named_parameters() if 'lora' in name]
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=avg_delta / 2500, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': avg_delta / 2500})

            params = [param for name, param in module.named_parameters() if 'delta' in name and 'list' not in name] ## weight quantizer
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=1e-6, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': 1e-6})

            params = [param for name, param in module.named_parameters() if 'delta_list' in name or 'zp_list' in name] ## act quantizer
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=5e-4, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': 5e-4})

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(NUM_EPOCHS * args.timesteps),
    )

    # Saving optimizer state for each step, which has little effect on performance. Uncomment line 368-370 & 405 in ddim.py if use it.
    # globalvar.init_state_list(ddim_steps) 

    model.eval()
    diffusion_trainer = DiffusionTrainer(args, config, device, fp_model=fp_model, quant_model=model, lr_scheduler=lr_scheduler, optimizer=optimizer)
    eval_out_dir = os.path.join('experiments_log', str(datetime.datetime.now()))
    os.makedirs(eval_out_dir)
    all_samples = list()

    batch_size = args.batch_size
    all_samples = list()
    st_time = time.time()
    # for epoch in range(NUM_EPOCHS):
    #     print(f'{epoch=}')
    #     torch.cuda.manual_seed(3407+epoch)
    #     samples_ddim = diffusion_trainer.sample(batch_size, train_mode=True).to(device)
    #     # print('---- samples_ddim -------')
    #     # print('samples_ddim shape, ', samples_ddim.shape)
    #     # print('min ', torch.min(samples_ddim))
    #     # print('max ', torch.max(samples_ddim))

    #     # x_samples_ddim = torch.clamp((samples_ddim+1.0)/2.0, 
    #     #                             min=0.0, max=1.0)

    #     # x_samples_ddim = torch.clamp(samples_ddim, min=0.0, max=1.0)
    #     # x_samples_ddim = (x_samples_ddim * 255.0).clamp(0, 255).to(torch.uint8)

    #     # print('x_samples_ddim min: ', torch.min(x_samples_ddim))
    #     # print('x_samples_ddim max: ', torch.max(x_samples_ddim))
    #     # x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
    #     # samples = x_samples_ddim.contiguous()
        
    #     # generate samples for visual evaluation
    #     if epoch % 16 == 0:
    #         print('Evaluating...')
    #         torch.cuda.manual_seed(3407)
    #         model.eval()
    #         for _ in range(4):
    #             samples_ddim = diffusion_trainer.sample(batch_size=6, train_mode=False).to(device)
    #             samples_ddim = torch.clamp((samples_ddim+1.0)/2.0, 
    #                                         min=0.0, max=1.0)
    #             all_samples.append(samples_ddim)
            
    #         # display as grid
    #         grid = torch.stack(all_samples, 0)
    #         grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    #         grid = make_grid(grid, nrow=6)

    #         # to image
    #         grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    #         image_to_save = Image.fromarray(grid.astype(np.uint8))
    #         image_to_save.save(os.path.join(eval_out_dir, "epoch{}.jpg".format(epoch)))
    #         all_samples.clear()

    torch.save(model.state_dict(), 'reproduce/ddim/weight/quantw{}a{}_{}steps_efficientdm.pth'.format(n_bits_w, n_bits_a, args.timesteps))

    ed_time = time.time()
    print(f'qlora took {ed_time - st_time:.5f} seconds')

