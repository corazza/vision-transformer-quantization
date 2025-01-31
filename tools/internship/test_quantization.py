import sys
sys.path.insert(0, '')

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings
import IPython
from numbers import Number
import copy

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, save_checkpoint)

from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, get_root_logger,
                         setup_multi_processes, wrap_distributed_model,
                         wrap_non_distributed_model)

from torch.quantization import quantize_dynamic
from torch.quantization import quantize_fx
from torch import nn
from mmcls.models.internship.backbones.quantized_swin import SwinTransformerQ
from mmcls.models.internship.backbones.quantized_vit import VisionTransformerQ
from mmcls.models.internship.utils.quantized_ops import *
from mmcls.models.internship.utils.quantized_vit_head import VisionTransformerClsHeadQ

from limited_dataset import LimitedDataset

from vt_test import single_gpu_test


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--model-out', help='quantized model output file')
    out_options = ['class_scores', 'pred_score', 'pred_label', 'pred_class']
    parser.add_argument(
        '--out-items',
        nargs='+',
        default=['all'],
        choices=out_options + ['none', 'all'],
        help='Besides metrics, what items will be included in the output '
        f'result file. You can choose some of ({", ".join(out_options)}), '
        'or use "all" to include all above, or use "none" to disable all of '
        'above. Defaults to output all.',
        metavar='')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--alt-attn',
        action='store_true',
        default=False,
        help='use alternative attention implementation')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
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
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    assert args.metrics or args.out, \
        'Please specify at least one of output path and evaluation metrics.'

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = args.device or auto_select_device()

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    dataset = build_dataset(cfg.data.test, default_args=dict(test_mode=True))
    # dataset = LimitedDataset(dataset, 200)

    # build the dataloader
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=1 if cfg.device == 'ipu' else len(cfg.gpu_ids),
        dist=distributed,
        round_up=True,
    )
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'shuffle': False,  # Not shuffle by default
        'sampler_cfg': None,  # Not use sampler by default
        **cfg.data.get('test_dataloader', {}),
    }
    # the extra round_up data will be removed during gpu/cpu collect
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    # cfg.model.backbone.alt_attn = args.alt_attn
    model = build_classifier(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint) # , map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES

    model = wrap_non_distributed_model(
        model, device=cfg.device, device_ids=cfg.gpu_ids)
    model.CLASSES = CLASSES
    show_kwargs = args.show_options or {}

    old_model = copy.deepcopy(model)
    model = static_quantize(model, data_loader)

    size_before = model_size(old_model)
    size_after = model_size(model)
    print(f'size before quantization: {size_before}MB, after: {size_after}MB')

    fps_old, outputs_old = single_gpu_test(old_model, data_loader, args.show, args.show_dir, **show_kwargs)
    results_old = process_outputs(dataset, outputs_old, fps_old, args, model)
    results_old['size'] = size_before
    fps, outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                **show_kwargs)
    results = process_outputs(dataset, outputs, fps, args, model)
    results['size'] = size_after

    print('results on original model:')
    print(results_old)
    print('results on quantized model:')
    print(results)

    final_results = {
        'quantized': results,
        'nonquantized': results_old,
    }

    mmcv.dump(final_results, args.out)


def process_outputs(dataset, outputs, fps, args, model):
    logger = get_root_logger()
    results = {'fps': fps}
    eval_results = dataset.evaluate(
        results=outputs,
        metric=args.metrics,
        metric_options=args.metric_options,
        logger=logger)
    results.update(eval_results)
    for k, v in eval_results.items():
        if isinstance(v, np.ndarray):
            v = [round(out, 2) for out in v.tolist()]
        elif isinstance(v, Number):
            v = round(v, 2)
        else:
            raise ValueError(f'Unsupport metric type: {type(v)}')

    if 'none' not in args.out_items:
        scores = np.vstack(outputs)
        pred_score = np.max(scores, axis=1)
        pred_label = np.argmax(scores, axis=1)
        pred_class = [model.CLASSES[lb] for lb in pred_label]
        res_items = {
            'class_scores': scores,
            'pred_score': pred_score,
            'pred_label': pred_label,
            'pred_class': pred_class
        }
        if 'all' in args.out_items:
            results.update(res_items)
        else:
            for key in args.out_items:
                results[key] = res_items[key]
    return results


def model_size(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p") / 1e6
    os.remove('temp.p')
    return size


import cProfile
import io
import pstats

def profile(func):
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.TIME  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper


@profile
def profiling_helper(m, data_loader):
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            m(return_loss=False, **data)

def profile_both(old_model, model, data_loader):
    print("regular model:")
    profiling_helper(old_model, data_loader)
    print("quantized model:")
    profiling_helper(model, data_loader)


def static_quantize(m, data_loader):
    backend = 'fbgemm'
    torch.backends.quantized.engine = backend
    m.eval()

    m.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    for module in m.modules():        
        module.qconfig = m.qconfig

    if hasattr(m, 'module'):
        m.module.backbone.insert_observers()
    else:
        m.backbone.insert_observers()

    prepare_custom_config_dict = {
        "float_to_observed_custom_module_class": {
            nn.Softmax: ExtendedQuantizedOpsObserver,
            ExtendedQuantizedOpsStub: ExtendedQuantizedOpsObserver
        }
    }
    torch.quantization.prepare(m, inplace=True, prepare_custom_config_dict=prepare_custom_config_dict)
    # torch.quantization.prepare(m, inplace=True)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= 100:
                break
            m(return_loss=False, **data)

    convert_custom_config_dict = {
        "observed_to_quantized_custom_module_class": {
            ExtendedQuantizedOpsObserver: ExtendedQuantizedOps,
        }
    }

    torch.quantization.convert(m, inplace=True, convert_custom_config_dict=convert_custom_config_dict)

    if hasattr(m, 'module'):
        if hasattr(m.module.backbone, 'quantize_rel_position_bias'):
            m.module.backbone.quantize_rel_position_bias()
    else:
        if hasattr(m.backbone, 'quantize_rel_position_bias'):
            m.backbone.quantize_rel_position_bias()

    return m

if __name__ == '__main__':
    main()
