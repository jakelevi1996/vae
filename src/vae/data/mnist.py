import torch
import torch.utils.data
import torchvision

class Mnist:
    def __init__(self):
        self.split_dict = {
            "train": torchvision.datasets.MNIST(
                root="data",
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
            "test": torchvision.datasets.MNIST(
                root="data",
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True,
            ),
        }

    @classmethod
    def get_input_dim(cls) -> int:
        return 28*28

    def get_split(self, split: str) -> torch.utils.data.Dataset:
        return self.split_dict[split]

    def get_data_loader(
        self,
        split:      str,
        batch_size: int,
        shuffle:    bool=True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return torch.utils.data.DataLoader(
            dataset=self.get_split(split),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def format_x(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(-3, -1)
