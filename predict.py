## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import numpy as np
import os
import argparse
from tqdm import tqdm

from models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
from utils.utils import load_img, save_img
import paddle
import paddle.nn.functional as F
from utils.utils import load_pretrained_model
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Gaussian Color Denoising using Restormer')

parser.add_argument('--input_images', default='/Users/alex/Downloads/Datasets/test/', type=str,
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

for sigma_test in sigmas:
    print("Compute results for noise level", sigma_test)
    model_restoration = Restormer(**x['network_g'])

    load_pretrained_model(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    print("------------------------------------------------")
    model_restoration.eval()

    if os.path.isdir(args.input_images):
        files = natsorted(glob(os.path.join(args.input_images, '*.png')) + glob(os.path.join(args.input_images, '*.tif')))
    else:
        files = [args.input_images]
    result_dir_tmp = args.result_dir
    os.makedirs(result_dir_tmp, exist_ok=True)

    with paddle.no_grad():
        for file_ in tqdm(files):
            img = np.float32(load_img(file_)) / 255.

            np.random.seed(seed=0)  # for reproducibility
            img += np.random.normal(0, sigma_test / 255., img.shape)

            img = paddle.to_tensor(img)
            img = paddle.transpose(img, [2, 0, 1])
            input_ = img.unsqueeze(0)

            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = paddle.clip(restored, 0, 1).detach()
            restored = paddle.transpose(restored, [0, 2, 3, 1]).squeeze(0).numpy()

            save_file = os.path.join(result_dir_tmp, "denoise_" + os.path.split(file_)[-1])
            save_img(save_file, img_as_ubyte(restored))

            img = paddle.transpose(img, [1, 2, 0]).clip(0, 1).numpy()
            save_file = os.path.join(result_dir_tmp, "noise_" + os.path.split(file_)[-1])
            save_img(save_file, img_as_ubyte(img))
    print(f"The predict image save in {result_dir_tmp} path.")
