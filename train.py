import argparse
import random
import math
import logging
import os.path as osp
import time
import datetime

import numpy as np
import paddle
from paddle.io import DataLoader

from utils.options import dict2str, parse
from utils.logger import get_root_logger, MessageLogger
from utils.misc import get_time_str, make_exp_dirs, mkdir_and_rename
from dataset import Dataset_GaussianDenoising
from models.image_restoration_model import ImageCleanModel

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed settings

    opt['dist'] = False
    print('Disable distributed.', flush=True)

    opt['rank'] = 0
    opt['world_size'] = 1
    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed

    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    return opt

def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)

    logger.info(dict2str(opt))

    return logger

def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = Dataset_GaussianDenoising(dataset_opt)
            train_loader = DataLoader(dataset=train_set,
                                      shuffle=True,
                                      drop_last=True,
                                      batch_size=dataset_opt['batch_size_per_gpu'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info(
                'Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                f'\n\tWorld size (gpu number): {opt["world_size"]}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

        elif phase == 'val':
            val_set = Dataset_GaussianDenoising(dataset_opt)
            val_loader = DataLoader(dataset=val_set,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0)
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, val_loader, total_epochs, total_iters



def main():
    opt = parse_options(is_train=True)

    # mkdir for experiments and logger

    make_exp_dirs(opt)
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
            'name'] and opt['rank'] == 0:
        mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger = init_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loader, total_epochs, total_iters = result

    model = ImageCleanModel(opt)
    start_epoch = 0
    current_iter = 0
    msg_logger = MessageLogger(opt, current_iter)
    # training
    logger.info(
        f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = time.time(), time.time()
    start_time = time.time()
    iters = opt['datasets']['train'].get('iters')
    batch_size = opt['datasets']['train'].get('batch_size_per_gpu')
    mini_batch_sizes = opt['datasets']['train'].get('mini_batch_sizes')
    gt_size = opt['datasets']['train'].get('gt_size')
    mini_gt_sizes = opt['datasets']['train'].get('gt_sizes')

    groups = np.array([sum(iters[0:i + 1]) for i in range(0, len(iters))])

    logger_j = [True] * len(groups)

    scale = opt['scale']

    epoch = start_epoch
    while current_iter <= total_iters:
        for idx, train_data in enumerate(train_loader):
            ### ------Progressive learning ---------------------
            j = ((current_iter > groups) != True).nonzero()[0]
            if len(j) == 0:
                bs_j = len(groups) - 1
            else:
                bs_j = j[0]

            mini_gt_size = mini_gt_sizes[bs_j]
            mini_batch_size = mini_batch_sizes[bs_j]

            if logger_j[bs_j]:
                logger.info('\n Updating Patch_Size to {} and Batch_Size to {} \n'.format(mini_gt_size,
                                                                                          mini_batch_size))
                logger_j[bs_j] = False

            lq = train_data['lq']
            gt = train_data['gt']

            if mini_batch_size < batch_size:
                indices = random.sample(range(0, batch_size), k=mini_batch_size)
                lq = lq[indices]
                gt = gt[indices]

            if mini_gt_size < gt_size:
                x0 = int((gt_size - mini_gt_size) * random.random())
                y0 = int((gt_size - mini_gt_size) * random.random())
                x1 = x0 + mini_gt_size
                y1 = y0 + mini_gt_size
                lq = lq[:, :, x0:x1, y0:y1]
                gt = gt[:, :, x0 * scale:x1 * scale, y0 * scale:y1 * scale]
            ###-------------------------------------------

            model.feed_train_data({'lq': lq, 'gt': gt})
            model.optimize_parameters(current_iter)

            iter_time = time.time() - iter_time
            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # save models and training states
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and (current_iter %
                                               opt['val']['val_freq'] == 0):
                rgb2bgr = opt['val'].get('rgb2bgr', True)
                # wheather use uint8 image to compute metrics
                use_image = opt['val'].get('use_image', True)
                model.validation(val_loader, current_iter,
                                 opt['val']['save_img'], rgb2bgr, use_image)

            data_time = time.time()
            iter_time = time.time()
            # end of iter
        epoch += 1

        # end of epoch

        consumed_time = str(
            datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f'End of training. Time consumed: {consumed_time}')
        logger.info('Save the latest model.')
        model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
        if opt.get('val') is not None:
            model.validation(val_loader, current_iter,
                             opt['val']['save_img'])


if __name__ == '__main__':
    main()
