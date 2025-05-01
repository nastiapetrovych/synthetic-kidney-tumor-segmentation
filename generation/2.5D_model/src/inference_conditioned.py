import json
import math
import os
import warnings
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import core.util as Util
import numpy as np
import torch
from PIL import Image
from core.logger import VisualWriter, InfoLogger
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from torchvision import transforms
from torchvision.utils import make_grid


class NoneDict(dict):
    def __missing__(self, key):
        return None


def mkdirs(paths):
    if isinstance(paths, str):
        os.makedirs(paths, exist_ok=True)
    else:
        for path in paths:
            os.makedirs(path, exist_ok=True)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')


def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        return NoneDict({k: dict_to_nonedict(v) for k, v in opt.items()})
    elif isinstance(opt, list):
        return [dict_to_nonedict(i) for i in opt]
    return opt


def load_config(config_path):
    with open(config_path, 'r') as f:
        json_str = ''.join([line.split('//')[0] + '\n' for line in f])
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['distributed'] = len(opt['gpu_ids']) > 1

    experiments_root = os.path.join(opt['path']['base_dir'], f"{opt['name']}_{get_timestamp()}")
    mkdirs(experiments_root)

    write_json(opt, f'{experiments_root}/config.json')
    opt['path']['experiments_root'] = experiments_root

    for key, path in opt['path'].items():
        if 'resume' not in key and 'base' not in key and 'root' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            mkdirs(opt['path'][key])

    return dict_to_nonedict(opt)


def load_test_files(flist_path):
    with open(flist_path, 'r') as f:
        return f.read().splitlines()


def group_files_by_subject(test_files, data_root):
    subjects = defaultdict(list)
    for file_path in test_files:
        subject_id = os.path.basename(file_path).split('_')[0]
        full_path = os.path.join(data_root, file_path)
        subjects[subject_id].append(full_path)
    return subjects


def load_subject_slices(subject_file_paths, data_root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    slices, masks, gts = [], [], []
    for file_path in sorted(subject_file_paths, key=lambda x: int(x.split('_slice_')[1].split('.')[0]))[:150]:
        filename = os.path.basename(file_path)
        slice_number = filename.split('_slice_')[1]

        input_image = Image.open(os.path.join(data_root, 'input', filename.replace('gt', 'masked'))).convert('RGB')
        gt_image = Image.open(os.path.join(data_root, 'gt', filename)).convert('RGB')
        mask_image = Image.open(os.path.join(data_root, 'mask', filename.replace('gt', 'mask'))).convert('L')

        input_tensor = transform(input_image)
        gt_tensor = transform(gt_image)
        mask_tensor = (input_tensor != gt_tensor).float()

        slices.append(input_tensor)
        gts.append(gt_tensor)
        masks.append(mask_tensor)

    return slices, gts, masks


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    tensor = tensor.clamp_(*min_max)
    if tensor.dim() == 4:
        img_np = make_grid(tensor, nrow=int(math.sqrt(len(tensor)))).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif tensor.dim() == 3:
        img_np = tensor.numpy().transpose(1, 2, 0)
    elif tensor.dim() == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(f"Unsupported tensor dimension: {tensor.dim()}")

    if out_type == np.uint8:
        img_np = ((img_np + 1) * 127.5).round()
    return img_np.astype(out_type).squeeze()


def save_results(results, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for subject, output_slices in results.items():
        subject_output_folder = os.path.join(output_folder, subject)
        os.makedirs(subject_output_folder, exist_ok=True)
        for i, output in enumerate(output_slices):
            output_image = tensor2img(output)
            Image.fromarray(output_image).save(os.path.join(subject_output_folder, f'slice_{i + 3}_inpainted.png'))


def run_inpainting_inference(config_path: str, exclude_subjects: list = None, device_override: str = None):
    opt = load_config(config_path)
    opt['phase'] = 'test'

    torch.backends.cudnn.enabled = True
    warnings.warn('Using cudnn for acceleration.')
    Util.set_seed(opt['seed'])

    logger = InfoLogger(opt)
    writer = VisualWriter(opt, logger)
    logger.info(f'Logs saved in: {opt["path"]["experiments_root"]}')

    phase_loader, val_loader = define_dataloader(logger, opt)
    networks = [define_network(logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
    losses = [define_loss(logger, item_opt) for item_opt in opt['model']['which_losses']]
    metrics = [define_metric(logger, item_opt) for item_opt in opt['model']['which_metrics']]

    test_files = load_test_files(opt['datasets']['test']['which_dataset']['args']['data_flist'])
    data_root = opt['datasets']['test']['which_dataset']['args']['data_root']
    subject_folder = os.path.join(data_root, 'input')
    subject_slices = group_files_by_subject(test_files, subject_folder)

    if exclude_subjects:
        for sid in exclude_subjects:
            subject_slices.pop(sid, None)

    output_folder = os.path.join(opt['path']['results'], 'test_results')
    device = torch.device(device_override if device_override else ('cuda' if torch.cuda.is_available() else 'cpu'))

    model = create_model(
        opt=opt,
        networks=networks,
        phase_loader=phase_loader,
        val_loader=val_loader,
        losses=losses,
        metrics=metrics,
        logger=logger,
        writer=writer
    )

    logger.info(f'Model initialized for phase: {opt["phase"]}')

    results = {
        subj: perform_inference(model, load_subject_slices(files, data_root), device, output_folder, subj)
        for subj, files in subject_slices.items()
    }
    save_results(results, output_folder)


if __name__ == '__main__':
    run_inpainting_inference(
        config_path='config/inpainting_ct_kidney_boxed_3slices.json',
        exclude_subjects=['s0503', 's0663'],
        device_override='cuda'
    )
