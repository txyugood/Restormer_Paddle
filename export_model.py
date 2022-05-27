import os
import argparse
import yaml

import paddle

from models.archs.restormer_arch import Restormer
from utils.utils import load_pretrained_model

def parse_args():
    parser = argparse.ArgumentParser(description='Model export.')
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        help='The directory for saving the exported model',
        type=str,
        default='./output')
    parser.add_argument(
        '--model_path',
        dest='model_path',
        help='The path of model for export',
        type=str,
        default=None)

    return parser.parse_args()


def main(args):


    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
    s = x['network_g'].pop('type')
    net = Restormer(**x['network_g'])

    if args.model_path:
        para_state_dict = paddle.load(args.model_path)
        net.set_dict(para_state_dict)
        print('Loaded trained params of model successfully.')


    shape = [-1, 3, 256, 256]

    new_net = net

    new_net.eval()
    new_net = paddle.jit.to_static(
        new_net,
        input_spec=[paddle.static.InputSpec(shape=shape, dtype='float32')])
    save_path = os.path.join(args.save_dir, 'model')
    paddle.jit.save(new_net, save_path)

    # yml_file = os.path.join(args.save_dir, 'deploy.yaml')
    # with open(yml_file, 'w') as file:
    #     transforms = cfg.export_config.get('transforms', [{
    #         'type': 'Normalize'
    #     }])
    #     data = {
    #         'Deploy': {
    #             'transforms': transforms,
    #             'model': 'model.pdmodel',
    #             'params': 'model.pdiparams'
    #         }
    #     }
    #     yaml.dump(data, file)

    print(f'Model is saved in {args.save_dir}.')


if __name__ == '__main__':
    args = parse_args()
    main(args)