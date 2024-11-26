import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

class SimpleTorchDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, aug: list = []) -> None:
        self.dataset: list[tuple[str, np.ndarray]] = []
        self.root_dir = root_dir
        
        # Updated for 7 classes (5 original + bed + sofa)
        self.__add_dataset__("chair",    [1, 0, 0, 0, 0, 0, 0])
        self.__add_dataset__("cupboard", [0, 1, 0, 0, 0, 0, 0])
        self.__add_dataset__("fridge",   [0, 0, 1, 0, 0, 0, 0])
        self.__add_dataset__("table",    [0, 0, 0, 1, 0, 0, 0])
        self.__add_dataset__("tv",       [0, 0, 0, 0, 1, 0, 0])
        self.__add_dataset__("bed",      [0, 0, 0, 0, 0, 1, 0])
        self.__add_dataset__("sofa",     [0, 0, 0, 0, 0, 0, 1])

        # Image preprocessing pipeline
        post_processing = [
            transforms.CenterCrop((177, 177)),
            transforms.ToTensor()
        ]

        self.augmentation = transforms.Compose(
            [transforms.Resize((200, 200))] +  
            aug +
            post_processing
        )
    
    def __add_dataset__(self, dir_name: str, class_label: list[int]) -> None:
        full_path = os.path.join(self.root_dir, dir_name)
        label = np.array(class_label)
        for fname in os.listdir(full_path):
            if fname.endswith(('.jpg', '.jpeg')):
                fpath = os.path.join(full_path, fname)
                fpath = os.path.abspath(fpath)
                self.dataset.append((fpath, label))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        fpath, label = self.dataset[index]
        image = Image.open(fpath).convert('RGB')
        image = self.augmentation(image)
        image = (image - image.min()) / (image.max() - image.min())
        label = torch.Tensor(label)
        return image, label