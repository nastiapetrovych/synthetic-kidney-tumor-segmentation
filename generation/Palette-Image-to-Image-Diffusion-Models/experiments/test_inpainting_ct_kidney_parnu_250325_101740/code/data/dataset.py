import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np

from tifffile import imread
import os
import pandas as pd
import albumentations

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')
    # return Image.open(path).convert('L')


class InpaintDataset_CT_kidney(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[256, 256]):
        self.data_root = data_root
        self.flist = self._read_flist(data_flist)

        if data_len > 0:
            self.flist = self.flist[:int(data_len)]

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.image_size = image_size

    def _read_flist(self, flist_path):
        with open(flist_path, 'r') as f:
            lines = f.readlines()
            file_paths = [line.strip() for line in lines]
        return file_paths

    def transform_filename(self, full_path):  ### ONLY for kist on kits artificail
        directory, filename = os.path.split(full_path)
        import re
        pattern = r"(case_\d+)_with_(case_\d+)_slice_(\d+\.png)"
        match = re.match(pattern, filename)

        if not match:
            print(f"Filename {filename} does not match the expected pattern.")
            return None

        case_number, _, slice_number = match.groups()
        print("case_number, _, slice_number", case_number, _, slice_number)
        new_filename = f"{case_number}_gt_slice_{slice_number}"
        # /gpfs/space/home/sedykh/NN_courseproject/artificial_inpainting_dataset/gt/case_00042_gt_slice_137.png

        new_full_path = os.path.join(directory, new_filename)

        new_full_path = new_full_path.replace("/input/", "/gt/")

        return new_full_path

    def __getitem__(self, index):
        ret = {}
        gt_path = os.path.join(self.data_root, "gt", self.flist[index])

        input_path = gt_path.replace("/gt/", "/input/").replace("_gt_slice_", "_masked_slice_")
        mask_path = gt_path.replace("/gt/", "/mask/").replace("_gt_slice_", "_mask_slice_")
        # /gpfs/space/home/sedykh/NN_courseproject/artificial_inpainting_dataset/input/case_00042_with_case_00194_slice_137.png

        # gt_path = self.transform_filename(input_path)

        input_img = self.tfs(Image.open(input_path).convert('L'))  # .convert('RGB'))
        gt_img = self.tfs(Image.open(gt_path).convert('L'))  # .convert('RGB'))
        mask = self.tfs(Image.open(mask_path).convert('L'))  # .convert('RGB'))

        mask = (input_img != gt_img).float()

        ret['gt_image'] = gt_img
        ret['cond_image'] = input_img
        ret['mask_image'] = input_img * (1. - mask) + mask
        ret['mask'] = mask
        ret['path'] = self.flist[index]
        return ret

    def __len__(self):
        return len(self.flist)


class InpaintDataset_CT_multislices(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[256, 256]):
        self.data_root = data_root
        self.flist = self._read_flist(data_flist)

        if data_len > 0:
            self.flist = self.flist[:int(data_len)]

        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.image_size = image_size

    def _read_flist(self, flist_path):
        with open(flist_path, 'r') as f:
            lines = f.readlines()
            file_paths = [line.strip() for line in lines]
        return file_paths

    def transform_filename(self, full_path):  ### ONLY for kist on kits artificail
        directory, filename = os.path.split(full_path)
        import re
        pattern = r"(case_\d+)_with_(case_\d+)_slice_(\d+\.png)"
        match = re.match(pattern, filename)

        if not match:
            print(f"Filename {filename} does not match the expected pattern.")
            return None

        case_number, _, slice_number = match.groups()
        print("case_number, _, slice_number", case_number, _, slice_number)
        new_filename = f"{case_number}_gt_slice_{slice_number}"
        # /gpfs/space/home/sedykh/NN_courseproject/artificial_inpainting_dataset/gt/case_00042_gt_slice_137.png

        new_full_path = os.path.join(directory, new_filename)

        new_full_path = new_full_path.replace("/input/", "/gt/")

        return new_full_path

    def __getitem__(self, index):
        ret = {}
        gt_path = os.path.join(self.data_root, "gt", self.flist[index])

        input_path = gt_path.replace("/gt/", "/input/").replace("_gt_slice_", "_masked_slice_")
        # /gpfs/space/home/sedykh/NN_courseproject/artificial_inpainting_dataset/input/case_00042_with_case_00194_slice_137.png

        # gt_path = self.transform_filename(input_path)

        input_img = self.tfs(Image.open(input_path).convert('RGB'))
        gt_img = self.tfs(Image.open(gt_path).convert('RGB'))

        mask = (input_img != gt_img).float()

        ret['gt_image'] = gt_img
        ret['cond_image'] = input_img
        ret['mask_image'] = input_img * (1. - mask) + mask
        ret['mask'] = mask
        ret['path'] = self.flist[index]
        return ret

    def __len__(self):
        return len(self.flist)


class InpaintDataset_CT_multislices_tensors(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[256, 256]):
        self.data_root = data_root
        self.flist = self._read_flist(data_flist)

        if data_len > 0:
            self.flist = self.flist[:int(data_len)]

        self.image_size = image_size

    def _read_flist(self, flist_path):
        with open(flist_path, 'r') as f:
            lines = f.readlines()
            file_paths = [line.strip() for line in lines]
        return file_paths

    def _normalize_tensor(self, tensor):
        # Normalize the tensor to the range [-1, 1]
        return (tensor - 0.5) * 2

    def __getitem__(self, index):
        ret = {}
        gt_path = os.path.join(self.data_root, "gt", self.flist[index])

        input_path = gt_path.replace("/gt/", "/input/").replace("_gt_slice_", "_input_slice_")
        mask_path = gt_path.replace("/gt/", "/mask/").replace("_gt_slice_", "_mask_slice_")

        # Load the tensors
        input_img = torch.load(input_path)
        gt_img = torch.load(gt_path)
        mask = torch.load(mask_path)

        # Normalize the tensors to the range [-1, 1]
        input_img = self._normalize_tensor(input_img)
        gt_img = self._normalize_tensor(gt_img)

        # Calculate the binary mask (assuming mask values are 0 or 1 in mask tensor)
        mask = (mask > 0).float()

        ret['gt_image'] = gt_img
        ret['cond_image'] = input_img
        ret['mask_image'] = input_img * (1. - mask) + mask
        ret['mask'] = mask
        ret['path'] = self.flist[index]
        return ret

    def __len__(self):
        return len(self.flist)
