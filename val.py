## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

from paddle.io import DataLoader
from models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
from utils.utils import load_img, save_img
import paddle
import paddle.nn.functional as F
from utils.utils import load_pretrained_model
from metrics import calculate_psnr
from pdb import set_trace as stx
from dataset import Dataset_GaussianDenoising

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_dir', default='/Users/alex/Downloads/Datasets/test/', type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Gaussian_Color_Denoising/', type=str,
                    help='Directory for results')
parser.add_argument('--weights', default='/Users/alex/Desktop/restormer.pdparams', type=str, help='Path to weights')
parser.add_argument('--model_type', required=True, choices=['non_blind', 'blind'], type=str,
                    help='blind: single model to handle various noise levels. non_blind: separate model for each noise level.')
parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')

args = parser.parse_args()

####### Load yaml #######
if args.model_type == 'blind':
    yaml_file = 'configs/GaussianColorDenoising_Restormer.yml'
else:
    yaml_file = f'configs/GaussianColorDenoising_RestormerSigma{args.sigmas}.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

sigmas = np.int_(args.sigmas.split(','))

factor = 8

datasets = ['CBSD68']

for sigma_test in sigmas:
    print("Compute results for noise level", sigma_test)
    model_restoration = Restormer(**x['network_g'])

    load_pretrained_model(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    print("------------------------------------------------")
    model_restoration.eval()

    x['datasets']['val']['phase'] = 'val'
    x['datasets']['val']['scale'] = 1
    val_set = Dataset_GaussianDenoising(x['datasets']['val'])
    batch_sampler = paddle.io.DistributedBatchSampler(
        val_set, batch_size=1, shuffle=False, drop_last=False)
    val_loader = DataLoader(dataset=val_set,
                            batch_sampler=batch_sampler,
                            num_workers=0)
    model_restoration.validation(val_loader, 0,
                                 False,
                                 rgb2bgr=True,
                                 use_image=False)
