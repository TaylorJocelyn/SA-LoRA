import sys
sys.path.append(".")
sys.path.append('./taming-transformers')
import os, time
os.environ['CUDA_VISIBLE_DEVICES']='0'

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import ResBlock
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

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
    from quant_scripts.ddim.quant_dataset import IntermediateInputDataset
    from torch.utils.data import DataLoader
    from quant_scripts.ddim.quant_dataset import get_train_samples
    # from quant_scripts.ddim.dntc_quant_sample import dntc_sample

    dataset = IntermediateInputDataset(args.cali_ckpt_path)
    data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
    cali_images, cali_t = get_train_samples(data_loader, num_samples=args.cali_sample_num)
    # cali_images, cali_t = dntc_sample(args.cali_ckpt_path, num_samples=args.cali_samples_num)
    
    wq_params = {'n_bits': args.weight_bit, 'channel_wise': True, 'scale_method': 'max'}
    aq_params = {'n_bits': args.act_bit, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': True}
    qnn = QuantModel(model=model, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.to(device)
    qnn.eval()

    print('Setting the first and the last layer to 8-bit')
    qnn.set_first_last_layer_to_8bit()

    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    print('First run to init model...')
    with torch.no_grad():
        _ = qnn(cali_images[:32].to(device), cali_t[:32].to(device))

    save_dir = 'reproduce/ddim/weight'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'quantw{}a{}_naiveQ.pth'.format(args.weight_bit, args.act_bit))
    torch.save(qnn.state_dict(), save_path)
    pass