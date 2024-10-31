import argparse, os, gc, glob, datetime, yaml
import logging
import math

import numpy as np
import tqdm
import torch
import torch.nn as nn
from torch.cuda import amp
from pytorch_lightning import seed_everything
import sys
sys.path.append('.')
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
    # parser.add_argument(
    #     "--quant_act", action="store_true", 
    #     help="if to quantize activations when ptq==True"
    # )
    parser.add_argument(
        "--cali_sample_num",
        type=int,
        default=8,
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
        "--max_images", type=int, default=50000, help="number of images to sample"
    )
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


if __name__ == "__main__":
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

    if args.sample_type == "generalized":
        xt_list, t_list = diffusion.collect_intermediate_data_ddim_sampler()
    elif args.sample_type == "dpm_solver":
        xt_list, t_list = diffusion.collect_intermediate_data_dpm_solver()

    save_dir = 'reproduce/ddim/data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "cifar10_{}step_{}_{}_inputdata.pth".format(args.timesteps, args.skip_type, args.sample_type))
    torch.save({'x': xt_list, 't': t_list}, save_path)