from torch.utils.data import Dataset
from torchvision import transforms as T


class Datareader(Dataset):
    def __init__(
        self,
        files_path: str,
        preprocess: T.Compose = None,
    ):
        super().__init__()

        self.files = []

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        pass
