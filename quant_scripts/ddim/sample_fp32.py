import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
# from taming.models import vqgan
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
torch.set_grad_enabled(False)
torch.cuda.manual_seed(3407)
import torch.nn as nn
from omegaconf import OmegaConf
import sys, time, datetime
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import argparse, yaml
import logging
from pytorch_lightning import seed_everything

from quant_scripts.ddim.quant_model import QuantModel_intnlora
from quant_scripts.ddim.quant_layer import QuantModule_intnlora, SimpleDequantizer

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
        "--quant_ckpt",
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

    from ddim.models.diffusion import Model
    # initialize FP32 model 
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

    device = next(qnn.parameters()).device
    # Initialize weight quantization parameters
    qnn.set_quant_state(True, True)

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight)

    for name, module in qnn.named_modules():
        if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
            module.weight.data = module.weight.data.byte() ## for running the model
            # module.intn_dequantizer.delta.data = module.weight_quantizer.delta ## share the same delta and zp
            # module.intn_dequantizer.zero_point.data = module.weight_quantizer.zero_point

    print('First run to init model...') ## need run to init act quantizer (delta_list)
    with torch.no_grad():
        _ = qnn(cali_images[:4].to(device),cali_t[:4].to(device))

    model = qnn
    
    ckpt = torch.load(args.quant_ckpt, map_location='cpu')
    model.load_state_dict(ckpt, strict=False) ## no lora weight in ckpt
    
    model.to(device)
    model.eval()

    all_samples = list()
    from ddim.models.diffusion import DiffusionTrainer
    diffusion_trainer = DiffusionTrainer(config)
    with torch.no_grad():
        for i in range(4):
            samples_ddim = diffusion_trainer.sample(batch_size=6, train_mode=False).to(device)
            samples_ddim = torch.clamp(samples_ddim, min=0.0, max=1.0)
            all_samples.append(samples_ddim)

    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=6)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    image_to_save = Image.fromarray(grid.astype(np.uint8))
    save_dir = 'reproduce/ddim/sample'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_to_save.save(os.path.join(save_dir, 'ddim_w{}a{}_{}steps.jpg'.format(n_bits_w, n_bits_a, args.timesteps)))
