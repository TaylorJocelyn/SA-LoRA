import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
torch.cuda.manual_seed(3407)
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

import numpy as np 

from quant_scripts.ddim.quant_model import QuantModel
from quant_scripts.ddim.quant_layer import QuantModule

import argparse
from tqdm import tqdm
import argparse, os, gc, glob, datetime, yaml
import logging
import math

import numpy as np
from torch.cuda import amp
from pytorch_lightning import seed_everything

from ddim.models.diffusion import Model, Diffusion
from ddim.datasets import inverse_data_transform
from ddim.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from ddim.functions.ckpt_util import get_ckpt_path
from torch.cuda import amp
import torchvision.utils as tvu

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
        "--quant_ckpt",
        type=str,
        required=True,
        help="calibration dataset path",
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
    diffusion = Diffusion(args, config, device)

    model = Model(config)
    name = "cifar10"

    from ddim.functions.ckpt_util import get_ckpt_path
    ckpt = get_ckpt_path(f"ema_{name}")
    logger.info("Loading checkpoint {}".format(ckpt))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    model.to(device)
    model.eval()

    n_bits_w = args.weight_bit
    n_bits_a = args.act_bit
    
    wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params, need_init=False)
    qnn.cuda()
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    ckpt = torch.load(args.quant_ckpt, map_location='cpu')
    qnn.load_state_dict(ckpt)
    qnn.cuda()
    qnn.eval()

    for name, param in qnn.named_parameters():
        param.requires_grad = False

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule) and module.ignore_reconstruction is False: 
            x = module.weight

            x_int = torch.round(x / module.weight_quantizer.delta) + module.weight_quantizer.zero_point
            x_quant = torch.clamp(x_int, 0, module.weight_quantizer.n_levels - 1) 

            ## pack to int
            ori_shape = x_quant.shape
            if module.fwd_func is F.conv2d:
                x_quant = x_quant.flatten(1)
            i = 0
            row = 0
            intweight = x_quant.int().cpu().numpy().astype(np.uint8)
            qweight = np.zeros(
                (intweight.shape[0] // 8 * n_bits_w, intweight.shape[1]), dtype=np.uint8
            )
            while row < qweight.shape[0]:
                if n_bits_w in [2, 4, 8]:
                    for j in range(i, i + (8 // n_bits_w)):
                        qweight[row] |= intweight[j] << (n_bits_w * (j - i))
                    i += 8 // n_bits_w
                    row += 1      

            qweight = torch.tensor(qweight).cuda()
            qweight = qweight.reshape([qweight.shape[0]]+list(ori_shape[1:]))

            module.weight.data = qweight
    
    qnn_sd = qnn.state_dict()
    torch.save(qnn_sd, 'reproduce/ddim/weight/quantw{}a{}_naiveQ_intsaved.pth'.format(n_bits_w, n_bits_a))
    