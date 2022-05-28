## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import argparse

import numpy as np
import paddle
from paddle.io import DataLoader

from dataset import Dataset_GaussianDenoising
from models.image_restoration_model import ImageCleanModel
from utils.utils import load_pretrained_model

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument(
    '-opt', type=str, default='configs/GaussianColorDenoising_Restormer.yml', required=True, help='Path to option YAML file.')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str,
                    help='Directory for results')
parser.add_argument('--weights', default='/Users/alex/Desktop/restormer.pdparams', type=str, help='Path to weights')
parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')

args = parser.parse_args()

yaml_file = args.opt
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

sigmas = np.int_(args.sigmas.split(','))

for sigma_test in sigmas:
    print("Compute results for noise level", sigma_test)
    x['is_train'] = False
    model = ImageCleanModel(x)

    load_pretrained_model(model.net_g, args.weights)
    print("===>Testing using weights: ", args.weights)
    print("------------------------------------------------")

    x['datasets']['val']['phase'] = 'val'
    x['datasets']['val']['scale'] = 1
    val_set = Dataset_GaussianDenoising(x['datasets']['val'])
    batch_sampler = paddle.io.DistributedBatchSampler(
        val_set, batch_size=1, shuffle=False, drop_last=False)
    val_loader = DataLoader(dataset=val_set,
                            batch_sampler=batch_sampler,
                            num_workers=0)
    current_metric = model.validation(val_loader, 0,
                                 False,
                                 rgb2bgr=True,
                                 use_image=False)
    print(f"[Eval] PSNR: {current_metric}")
