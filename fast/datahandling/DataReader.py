import os
import random
from glob import glob

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class DataReader(Dataset):
    def __init__(
        self,
        files_path: str,
        style_path: str,
        image_size: int | tuple = (64, 64),
        preprocess: T.Compose = None,
    ):
        super().__init__()
        self.files_path = files_path
        self.style_path = style_path

        self.files = [file for file in glob(os.path.join(self.files_path, "*.jpg"))]
        self.style_files = [
            file for file in glob(os.path.join(self.style_path, "*.jpg"))
        ]

        print(f"Number of content images: {len(self.files)}")
        print(f"Number of style images: {len(self.style_files)}")

        # TODO - Remember a normalization method of the input images.
        self.transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ],
        )

    def normalize_tensor(self, data):
        return (data - torch.min(data)) / (torch.max(data) - torch.min(data))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        # Read content image.
        content_img_path = os.path.join(self.files_path, self.files[index])
        content_image = Image.open(content_img_path)
        content_image = content_image.convert("RGB")
        content_image = self.normalize_tensor(self.transform(content_image))

        # Random style image, as there are fewer style images than content images.
        style_idx = random.randint(0, len(self.style_files) - 1)
        # style_idx = 3
        style_img_path = os.path.join(self.style_path, self.style_files[style_idx])
        style_image = Image.open(style_img_path)
        style_image = style_image.convert("RGB")
        style_image = self.normalize_tensor(self.transform(style_image))

        return content_image, style_image
