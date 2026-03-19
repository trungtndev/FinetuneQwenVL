from functools import partial
from typing import Optional

import albumentations as A
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Subset
import pytorch_lightning as pl
from transformers import AutoProcessor

K_MIN = 0.7
K_MAX = 1.4


def find_image_path(base):
    for ext in [".bmp", ".png", ".jpg", ".jpeg"]:
        path = base + ext
        if os.path.exists(path):
            return path
    raise FileNotFoundError(base)


class AlbScaleAugmentation(A.ImageOnlyTransform):
    def __init__(self, lo: float, hi: float, always_apply=False, p=1.0) -> None:
        super().__init__(p)
        assert lo <= hi
        self.lo = lo
        self.hi = hi

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        k = np.random.uniform(self.lo, self.hi)
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        return img


class ResizeLimit(A.ImageOnlyTransform):
    def __init__(self, height, width, always_apply=True, p=1.0):
        super().__init__(p=p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = min(self.height / h, self.width / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return img


class CROHMEDataset(Dataset):
    def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
        super().__init__()
        self.ds = [
            (fname, find_image_path(p), caption)
            for fname, p, caption in ds
        ]

        trans_list = []
        if is_train and scale_aug:
            trans_list.append(AlbScaleAugmentation(K_MIN, K_MAX))

        trans_list.append(ResizeLimit(256, 768))
        self.transform = A.Compose(trans_list)

    def __getitem__(self, idx):
        fname, p, caption = self.ds[idx]

        img = Image.open(p).convert("RGB")
        img = self.transform(image=np.array(img))["image"]

        img = Image.fromarray(img)

        return {
            "fname": fname,
            "caption": caption,
            "image": img,
        }

    def __len__(self):
        return len(self.ds)

def extract_data(archive, dir_name: str):
    with open(f"{archive}/{dir_name}/caption.txt", "rb") as f:
        captions = f.readlines()
    data = []
    for line in captions:
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = " ".join(tmp[1:])
        data.append(
            (img_name, f"{archive}/{dir_name}/img/{img_name}", formula)
        )

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


def collate_fn(batch):
    # Chỉ gom data thô thành mảng, KHÔNG gọi processor ở đây
    fnames = [item["fname"] for item in batch]
    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]

    return {
        "fnames": fnames,
        "images": images,
        "captions": captions
    }

def build_dataset(archive, folder: str, batch_size: int):
    return extract_data(archive, folder)


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
            test_year: str = "2014",
            train_batch_size: int = 8,
            eval_batch_size: int = 4,
            num_workers: int = 5,
            scale_aug: bool = False,
            val_subset_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug
        self.val_subset_size = val_subset_size
        # self.processor = AutoProcessor.from_pretrained(
        #     "Qwen/Qwen3-VL-2B-Instruct",
        #     trust_remote_code=True
        # )
        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, "train", self.train_batch_size),
                True,
                self.scale_aug,
            )

            self.val_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, self.test_year, self.eval_batch_size),
                False,
                self.scale_aug,
            )
            if self.val_subset_size is not None and self.val_subset_size < len(self.val_dataset):
                seed = 42
                g = torch.Generator()
                g.manual_seed(seed)
                indices = torch.randperm(len(self.val_dataset), generator=g)[:self.val_subset_size]
                self.val_dataset = Subset(self.val_dataset, indices)
                print(f"Subset size: {len(self.val_dataset)} with seed: {seed}")

        if stage == "test" or stage is None:
            self.test_dataset = CROHMEDataset(
                build_dataset(self.zipfile_path, self.test_year, self.eval_batch_size),
                False,
                self.scale_aug,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )