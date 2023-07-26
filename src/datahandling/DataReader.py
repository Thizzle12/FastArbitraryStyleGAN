import os
import random
import cv2
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms as T


class Datareader(Dataset):
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

        self.files = os.listdir(files_path)
        self.style_files = os.listdir(style_path)

        # [print(file) for file in self.files]
        # [print(file) for file in self.style_files]

        self.transform = T.Compose(
            [
                T.Resize(image_size),
                T.ToTensor(),
            ],
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        content_image = 0
        style_image = 0

        # Read content image.
        content_img_path = os.path.join(self.files_path, self.files[index])
        print(content_img_path)
        content_image = Image.open(content_img_path)
        content_image = content_image.convert("RGB")
        content_image = self.transform(content_image)

        # Random style image, as there are fewer style images than content images.
        stlye_idx = random.randint(0, len(self.style_files) - 1)
        style_img_path = os.path.join(self.style_path, self.style_files[stlye_idx])
        print(style_img_path)
        style_image = Image.open(style_img_path)
        style_image = style_image.convert("RGB")
        style_image = self.transform(style_image)

        return content_image, style_image
