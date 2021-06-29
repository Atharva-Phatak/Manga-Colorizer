import torch
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torchvision import transforms


class ColorizationDataset(torch.utils.data.Dataset):
    """PyTorch Style dataset for training our GANs."""

    def __init__(self, paths, img_size, split="train") -> None:
        """Constructor for ColorizationDataset.
        Args:
            paths: The paths where images are stored.
            img_size: The size of images.
            split: One of train or valid.
        """
        super(ColorizationDataset, self).__init__()
        self.paths = paths
        if split == "train":
            self.transforms = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        elif split == "valid":
            self.transforms = transforms.Resize((256, 256))

    def __len__(self):

        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...] / 50.0 - 1.0
        ab = img_lab[[1, 2], ...] / 110.0

        return L, ab
