# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import zipfile
import glob
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmseg.registry import RUNNERS,MODELS
from mmseg.utils import set_env
from AgriFM.utils import path_utils


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path',
                        default=r'configs/ar_pretraining/shrub_fraction_swin.py')
    parser.add_argument('--work-dir', help='the dir to save logs and models',
                        )
    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=True,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    #     # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def zip_codes(zip_file,zip_dir=('.','mmseg','AgriFM')):
    zip_fp=zipfile.ZipFile(zip_file,'w',compression=zipfile.ZIP_DEFLATED)
    for dir in zip_dir:
        if os.getcwd()==os.path.abspath(dir):
            data_files=glob.glob(os.path.join(dir,'*.*'))
        else:
            data_files = glob.glob(os.path.join(dir, '**'),recursive=True)

        data_files=[x for x in data_files if os.path.isfile(x)]
        for data_file in data_files:
            zip_fp.write(data_file,compress_type=zipfile.ZIP_DEFLATED)
    zip_fp.close()

def main():
    args = parse_args()
    set_env.register_all_modules()
    # load config
    cfg = Config.fromfile(args.config)
    cfg_name=path_utils.get_filename(args.config,is_suffix=False)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume
    file_name=path_utils.get_filename(args.config,is_suffix=False)
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir,exist_ok=True)
    zip_file=os.path.join(cfg.work_dir,'%s.zip'%file_name)
    zip_codes(zip_file,zip_dir=('.','mmseg','AgriFM'))
    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
