"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys, time
sys.path.append(".")
sys.path.append('./taming-transformers')
import argparse
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import time
import logging
import numpy as np
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
# from ldm.models.diffusion.ddim import DDIMSampler, DDIMSampler_quantCorrection_imagenet, DDIMSampler_implicit_gaussian_quantCorrection_imagenet, DDIMSampler_integral_gaussian_quantCorrection_imagenet, DDIMSampler_gaussian_quantCorrection_imagenet, DDIMSampler_variance_shrinkage_gaussian_quantCorrection_imagenet, DDIMSampler_channel_wise_implicit_gaussian_quantCorrection_imagenet, DDIMSampler_improved_gaussian_quantCorrection_imagenet, DDIMSampler_channel_wise_explicit_gaussian_quantCorrection_imagenet
from ldm.models.diffusion.ddim import DDIMSampler
from quant_scripts.ldmi.quant_dataset import DiffusionInputDataset, get_calibration_set, get_train_samples
from quant_scripts.ldmi.quant_model import QuantModel
from quant_scripts.ldmi.quant_layer import QuantModule
# from quant_scripts.brecq_adaptive_rounding import AdaRoundQuantizer

import torch.nn as nn
from quant_scripts.ldmi.quant_dataset import DiffusionInputDataset
from torch.utils.data import DataLoader
from quant_scripts.ldmi.quant_model import QuantModel_intnlora
from quant_scripts.ldmi.quant_layer import QuantModule_intnlora, SimpleDequantizer

n_bits_w = 4
n_bits_a = 4

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

def get_train_samples(train_loader, num_samples):
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    return torch.cat(image_data, dim=0)[:num_samples], torch.cat(t_data, dim=0)[:num_samples], torch.cat(y_data, dim=0)[:num_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=25000)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--quant_type', default='fp32')
    parser.add_argument('--local-rank', help="local device id on current node", type=int)
    parser.add_argument('--nproc_per_node', default=2, type=int)
    args = parser.parse_args()
    print(args)

    print('quant type: ', args.quant_type)

    ddim_steps = 50
    ddim_eta = 0.4
    scale = 3.0  # for  guidance

    # ddim_steps = 250
    # ddim_eta = 1.0
    # scale = 1.5

    os.makedirs('evaluate_data/scale{}_eta{}_steps{}'.format(scale, ddim_eta, ddim_steps), exist_ok=True)

    # init ddp
    local_rank = args.local_rank
    device = torch.device("cuda", local_rank)
    
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.nproc_per_node, rank=local_rank)
    rank = torch.distributed.get_rank()
    
    # seed = int(time.time())
    seed = 100
    torch.manual_seed(seed + rank)
    torch.cuda.set_device(local_rank)
    torch.set_grad_enabled(False)

    # Load model:
    model = get_model()
    dmodel = model.model.diffusion_model
    dmodel.cuda(rank)
    model.cuda(rank)
    dmodel.eval()
    
    if args.quant_type != 'fp32':
        print('quant type: ', args.quant_type)

        dataset = DiffusionInputDataset('/home/zq/EfficientDM/reproduce/ldmi/data/DiffusionInput_250steps.pth')
        data_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True) ## each sample is (16,4,32,32)
        
        wq_params = {'n_bits': n_bits_w, 'channel_wise': True, 'scale_method': 'mse'}
        aq_params = {'n_bits': n_bits_a, 'channel_wise': False, 'scale_method': 'max', 'leaf_param': True}
        qnn = QuantModel_intnlora(model=dmodel, weight_quant_params=wq_params, act_quant_params=aq_params, num_steps=ddim_steps)

        qnn.cuda(rank)
        qnn.eval()

        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

        cali_images, cali_t, cali_y = get_train_samples(data_loader, num_samples=1024)
        device = next(qnn.parameters()).device
        # Initialize weight quantization parameters
        qnn.set_quant_state(True, True)

        for name, module in qnn.named_modules():
            if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
                module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight, device=device)

        for name, module in qnn.named_modules():
            if isinstance(module, QuantModule_intnlora) and module.ignore_reconstruction is False:
                module.weight.data = module.weight.data.byte() ## for running the model
            
        print('First run to init model...') ## need run to init act quantizer (delta_list)
        with torch.no_grad():
            _ = qnn(cali_images[:4].to(device),cali_t[:4].to(device),cali_y[:4].to(device))

        setattr(model.model, 'diffusion_model', qnn)
        
        ckpt = torch.load('reproduce/ldmi/weight/quantw4a4_100steps_efficientdm_sample50.pth')
        model.load_state_dict(ckpt)
        
        model.cuda(rank)
        model.eval()

    model=nn.parallel.DistributedDataParallel(model.cuda(rank), device_ids=[rank])

    sampler = DDIMSampler(model.module)

    out_path = os.path.join('evaluate_data/scale{}_eta{}_steps{}'.format(scale, ddim_eta, ddim_steps), f"{args.quant_type}_{args.num_samples}_steps{ddim_steps}_eta{ddim_eta}_scale{scale}_1.npz")
    print("out_path ", out_path)
    logging.info("sampling...")
    generated_num = torch.tensor(0, device=device)
    if rank == 0:
        all_images = []
        all_labels = []
        
        generated_num = torch.tensor(0, device=device)
    dist.barrier()
    dist.broadcast(generated_num, 0)
    n_samples_per_class = args.batch_size

    label_idx = 0
    while generated_num.item() < args.num_samples:

        if label_idx < 1000:
            class_labels = torch.tensor([min(label_idx+i, args.num_classes-1)for i in range(args.batch_size)], device=device)
            label_idx += args.batch_size
        else:
            class_labels = torch.randint(low=0,
                                        high=args.num_classes,
                                        size=(args.batch_size,),
                                        device=device)

        # class_labels = torch.randint(low=0,
        #                              high=args.num_classes,
        #                              size=(args.batch_size,),
        #                              device=device)

        print('class_labels ', class_labels)
        
        uc = model.module.get_learned_conditioning(
            {model.module.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.module.device)}
            )
        
        for class_label in class_labels:
            t0 = time.time()
            xc = torch.tensor(n_samples_per_class*[class_label]).to(model.module.device)
            c = model.module.get_learned_conditioning({model.module.cond_stage_key: xc.to(model.module.device)})
            
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=c,
                                            batch_size=n_samples_per_class,
                                            shape=[3, 64, 64],
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc, 
                                            eta=ddim_eta)

            x_samples_ddim = model.module.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                        min=0.0, max=1.0)
            
            x_samples_ddim = (x_samples_ddim * 255.0).clamp(0, 255).to(torch.uint8)
            x_samples_ddim = x_samples_ddim.permute(0, 2, 3, 1)
            samples = x_samples_ddim.contiguous()

            t1 = time.time()
            print('throughput : {}'.format((t1 - t0) / x_samples_ddim.shape[0]))
            
            # print('world_size: ', dist.get_world_size())
            gathered_samples = [torch.zeros_like(samples) for _ in range(dist.get_world_size())]
            # print('gathered_samples[0] shape: ', gathered_samples[0].shape)
            # print('gathered_samples len: ', len(gathered_samples))
            dist.all_gather(gathered_samples, samples)  

            gathered_labels = [
                torch.zeros_like(xc) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, xc)

            if rank == 0:
                all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
                # print('all_images len', len(all_images))
                # print('all_images[0] shape', all_images[0].shape)
                # print('dtype ', gathered_samples[0].cpu().numpy().dtype)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
                logging.info(f"created {len(all_images) * n_samples_per_class} samples")

                # save image
                # idx = 0
                # for i in range(len(gathered_samples)):
                #     bs = gathered_samples[i].shape[0]
                #     for j in range(bs):
                #         img_id = generated_num + idx
                #         img = gathered_samples[i][j].cpu().numpy()
                #         image_to_save = Image.fromarray(img.astype(np.uint8))
                #         image_to_save.save(out_dir + f"/img_{img_id :05d}.png")
                #         idx += 1

                generated_num = torch.tensor(len(all_images) * n_samples_per_class, device=device)
                print("generated_num: ", generated_num.item())
                
            torch.distributed.barrier()
            dist.broadcast(generated_num, 0)

    if rank == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]

        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        logging.info(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logging.info("sampling complete")
    print('label_idx: ', label_idx)


if __name__ == "__main__":
    main()