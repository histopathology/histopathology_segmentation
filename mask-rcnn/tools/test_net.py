"""Perform inference on one or more datasets."""

import argparse
import cv2
import os
import pprint
import sys
import time

import torch

import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine import run_inference
import utils.logging

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)

    assert (torch.cuda.device_count() == 1) ^ bool(args.multi_gpu_testing)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if args.output_dir is None:
        ckpt_path = args.load_ckpt if args.load_ckpt else args.load_detectron
        args.output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    cfg.VIS = args.vis

    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        merge_cfg_from_list(args.set_cfgs)

    if args.dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    if args.dataset == "coco2014":
        cfg.TEST.DATASETS = ('coco_2014_val',)
        cfg.MODEL.NUM_CLASSES = 81
    elif args.dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    elif args.dataset == "monuseg2018":
        cfg.TEST.DATASETS = ('monuseg_2018_train',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.TEST.SCALE = 1000



    elif args.dataset == "monuseg_baseline":
        cfg.TEST.DATASETS = ('monuseg_baseline_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_default":
        cfg.TEST.DATASETS = ('monuseg_baseline_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
    elif args.dataset == "cpm_all":
        cfg.TEST.DATASETS = ('cpm_all_train',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000


    elif args.dataset == "monuseg_all":
        cfg.TEST.DATASETS = ('monuseg_all_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_all_gan":
        cfg.TEST.DATASETS = ('monuseg_all_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_testset_1":
        cfg.TEST.DATASETS = ('monuseg_testset_1_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_testset_2":
        cfg.TEST.DATASETS = ('monuseg_testset_2_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_testset_3":
        cfg.TEST.DATASETS = ('monuseg_testset_3_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_testset_1_gan":
        cfg.TEST.DATASETS = ('monuseg_testset_1_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_testset_2_gan":
        cfg.TEST.DATASETS = ('monuseg_testset_2_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_testset_3_gan":
        cfg.TEST.DATASETS = ('monuseg_testset_3_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8

    elif args.dataset == "BNS_all":
        cfg.TEST.DATASETS = ('BNS_all_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_all_gan":
        cfg.TEST.DATASETS = ('BNS_all_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_1":
        cfg.TEST.DATASETS = ('monuseg_baseline_1_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_1_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_1_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_2":
        cfg.TEST.DATASETS = ('monuseg_baseline_2_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_2_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_2_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_3":
        cfg.TEST.DATASETS = ('monuseg_baseline_3_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_3_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_3_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_4":
        cfg.TEST.DATASETS = ('monuseg_baseline_4_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_5":
        cfg.TEST.DATASETS = ('monuseg_baseline_5_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_6":
        cfg.TEST.DATASETS = ('monuseg_baseline_6_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_10":
        cfg.TEST.DATASETS = ('monuseg_baseline_10_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_11":
        cfg.TEST.DATASETS = ('monuseg_baseline_11_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_16":
        cfg.TEST.DATASETS = ('monuseg_baseline_16_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_16_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_16_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_17":
        cfg.TEST.DATASETS = ('monuseg_baseline_17_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_17_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_17_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_18":
        cfg.TEST.DATASETS = ('monuseg_baseline_18_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_18_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_18_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_19":
        cfg.TEST.DATASETS = ('monuseg_baseline_19_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_19_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_19_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_20":
        cfg.TEST.DATASETS = ('monuseg_baseline_20_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_20_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_20_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_21":
        cfg.TEST.DATASETS = ('monuseg_baseline_21_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_21_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_21_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_22":
        cfg.TEST.DATASETS = ('monuseg_baseline_22_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_22_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_22_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_23":
        cfg.TEST.DATASETS = ('monuseg_baseline_23_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_23_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_23_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_24":
        cfg.TEST.DATASETS = ('monuseg_baseline_24_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_24_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_24_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_25":
        cfg.TEST.DATASETS = ('monuseg_baseline_25_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_25_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_25_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8

    elif args.dataset == "monuseg_baseline_best":
        cfg.TEST.DATASETS = ('monuseg_baseline_best_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_best_gan":
        cfg.TEST.DATASETS = ('monuseg_baseline_best_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_baseline_best_gan_":
        cfg.TEST.DATASETS = ('monuseg_baseline_best_val_gan_',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_0":
        cfg.TEST.DATASETS = ('monuseg_0_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_0_gan":
        cfg.TEST.DATASETS = ('monuseg_0_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_1":
        cfg.TEST.DATASETS = ('monuseg_1_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_1_gan":
        cfg.TEST.DATASETS = ('monuseg_1_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_2":
        cfg.TEST.DATASETS = ('monuseg_2_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_2_gan":
        cfg.TEST.DATASETS = ('monuseg_2_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_3":
        cfg.TEST.DATASETS = ('monuseg_3_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_3_gan":
        cfg.TEST.DATASETS = ('monuseg_3_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_4":
        cfg.TEST.DATASETS = ('monuseg_4_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_4_gan":
        cfg.TEST.DATASETS = ('monuseg_4_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_5":
        cfg.TEST.DATASETS = ('monuseg_5_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_5_gan":
        cfg.TEST.DATASETS = ('monuseg_5_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_6":
        cfg.TEST.DATASETS = ('monuseg_6_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "monuseg_6_gan":
        cfg.TEST.DATASETS = ('monuseg_6_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8




    elif args.dataset == "monuseg_baseline_val":
        cfg.TEST.DATASETS = ('monuseg_baseline_val_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8


    elif args.dataset == "CPM_1":
        cfg.TEST.DATASETS = ('CPM_1_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "CPM_2":
        cfg.TEST.DATASETS = ('CPM_2_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
        
    elif args.dataset == "BNS":
        cfg.TEST.DATASETS = ('BNS_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "TNBC":
        cfg.TEST.DATASETS = ('TNBC_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8




    elif args.dataset == "BNS_0":
        cfg.TEST.DATASETS = ('BNS_0_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_1":
        cfg.TEST.DATASETS = ('BNS_1_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_2":
        cfg.TEST.DATASETS = ('BNS_2_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_3":
        cfg.TEST.DATASETS = ('BNS_3_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_4":
        cfg.TEST.DATASETS = ('BNS_4_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_5":
        cfg.TEST.DATASETS = ('BNS_5_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_6":
        cfg.TEST.DATASETS = ('BNS_6_val',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_0_gan":
        cfg.TEST.DATASETS = ('BNS_0_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_1_gan":
        cfg.TEST.DATASETS = ('BNS_1_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_2_gan":
        cfg.TEST.DATASETS = ('BNS_2_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_3_gan":
        cfg.TEST.DATASETS = ('BNS_3_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_4_gan":
        cfg.TEST.DATASETS = ('BNS_4_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_5_gan":
        cfg.TEST.DATASETS = ('BNS_5_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8
    elif args.dataset == "BNS_6_gan":
        cfg.TEST.DATASETS = ('BNS_6_val_gan',)
        cfg.MODEL.NUM_CLASSES = 2
        cfg.TEST.DETECTIONS_PER_IM = 1000
        cfg.FPN.RPN_ANCHOR_START_SIZE = 8


    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    args.test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    args.cuda = True

    run_inference(
        args,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True)
