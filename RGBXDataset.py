import os
import cv2
import torch
import numpy as np
import torch.utils.data as data


class RGBXDataset(data.Dataset):
    def __init__(self, setting, split_name, preprocess=None, file_length=None):
        super(RGBXDataset, self).__init__()
        self._split_name        = split_name
        self._rgb_path          = setting['rgb_root']
        self._rgb_format        = setting['rgb_format']
        self._gt_path           = setting['gt_root']
        self._gt_format         = setting['gt_format']
        self._transform_gt      = setting['transform_gt']
        self._x_path            = setting['x_root']
        self._x_format          = setting['x_format']
        self._x_single_channel  = setting['x_single_channel']
        self._train_source      = setting['train_source']
        self._eval_source       = setting['eval_source']
        self._test_source       = setting.get('test_source', setting['eval_source'])
        self.class_names        = setting['class_names']
        self._file_names        = self._get_file_names(split_name)
        self._file_length       = file_length
        self.preprocess         = preprocess

    def __len__(self):
        if self._file_length is not None:
            return self._file_length
        return len(self._file_names)

    def __getitem__(self, index):
        if self._file_length is not None:
            item_name = self._construct_new_file_names(self._file_length)[index]
        else:
            item_name = self._file_names[index]

        rgb_path = os.path.join(self._rgb_path, item_name + self._rgb_format)
        x_path   = os.path.join(self._x_path,   item_name + self._x_format)
        gt_path  = os.path.join(self._gt_path,   item_name + self._gt_format)

        rgb = self._open_image(rgb_path, cv2.COLOR_BGR2RGB)

        gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=np.uint8)
        if self._transform_gt:
            gt = self._gt_transform(gt)

        if self._x_single_channel:
            x = self._open_image(x_path, cv2.IMREAD_GRAYSCALE)
            x = cv2.merge([x, x, x])
        else:
            x = self._open_image(x_path, cv2.COLOR_BGR2RGB)

        if self.preprocess is not None:
            rgb, gt, x = self.preprocess(rgb, gt, x)

        if self._split_name == 'train':
            rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
            gt  = torch.from_numpy(np.ascontiguousarray(gt)).long()
            x   = torch.from_numpy(np.ascontiguousarray(x)).float()

        output_dict = dict(
            data=rgb, label=gt, modal_x=x,
            fn=str(item_name), n=len(self._file_names)
        )
        return output_dict

    def _get_file_names(self, split_name):
        assert split_name in ['train', 'val', 'test']
        if split_name == 'train':
            source = self._train_source
        elif split_name == 'val':
            source = self._eval_source
        else:
            source = self._test_source

        file_names = []
        with open(source) as f:
            for item in f.readlines():
                file_name = item.strip()
                if file_name:
                    file_names.append(file_name)
        return file_names

    def _construct_new_file_names(self, length):
        assert isinstance(length, int)
        files_len      = len(self._file_names)
        new_file_names = self._file_names * (length // files_len)
        rand_indices   = torch.randperm(files_len).tolist()
        new_indices    = rand_indices[:length % files_len]
        new_file_names += [self._file_names[i] for i in new_indices]
        return new_file_names

    def get_length(self):
        return self.__len__()

    # ── FIX 1: เพิ่ม error handling ป้องกัน None crash ────────────
    @staticmethod
    def _open_image(filepath, mode=cv2.IMREAD_COLOR, dtype=None):
        img = cv2.imread(filepath, mode)
        if img is None:
            raise FileNotFoundError(
                f"[RGBXDataset] Image not found or cannot be read: {filepath}"
            )
        return np.array(img, dtype=dtype)

    # ── FIX 2: ปิด gt - 1 ที่ทำให้ class index เลื่อนทุกตัว ───────
    @staticmethod
    def _gt_transform(gt):
        # ลบ gt - 1 ออก เพราะ labelmap เริ่มที่ 0 = Background อยู่แล้ว
        # gt - 1 ทำให้ Background(0) → 255(ignore), Biotite(1) → 0 ฯลฯ
        # ผลคือ class_names เลื่อนไป 1 ทุกตัว
        return gt  # ← คืนค่าตรงๆ ไม่ต้องแปลง

    # ── FIX 3: ใช้สีจาก labelmap จริง แทน bit-manipulation ─────────
    @classmethod
    def get_class_colors(cls, *args):
        return [
            [  0,   0,   0],  # 0  Background
            [170, 118,  57],  # 1  Biotite
            [115,  51, 128],  # 2  Clinopyroxene
            [ 36, 179,  83],  # 3  Hornblende
            [ 51, 221, 255],  # 4  K-feldspar
            [250, 250,  55],  # 5  Leucite
            [255, 204,  51],  # 6  Muscovite
            [ 61, 245,  61],  # 7  Olivine
            [100, 100, 100],  # 8  Opaque
            [250, 125, 187],  # 9  Orthopyroxene
            [ 52, 209, 183],  # 10 Plagioclase
            [250,  50,  83],  # 11 Quartz
            [255,   0,   0],  # 12 Spinel
            [ 61,  61, 245],  # 13 Topaz
            [246, 132,   6],  # 14 Tourmaline
        ]