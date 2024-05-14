import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from PIL import Image

from torchvision.datasets import VisionDataset


class DatasetFolder(VisionDataset):
    """A generic data loader.
    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        use_ceph = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.loader = loader

        self.use_ceph = use_ceph
        if self.use_ceph:
            from petrel_client.client import Client
            self.ceph_client = Client(enable_mc=True)
            self.loader = self.ceph_load

        classes, class_to_idx, samples = self.find_classes_and_make_dataset(self.root)

        
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
    
    def find_classes_and_make_dataset(self, directory):
        directory = os.path.expanduser(directory)
        classes = []
        class_to_idx = {}
        samples = []
        train_or_val = directory.split('/')[-1]
        meta_file_name = os.path.join(directory.rstrip(train_or_val), 'meta', train_or_val + '.txt')
        meta_file = os.path.join(meta_file_name)

        if self.use_ceph:
            lines = self.ceph_client.Get(meta_file).decode('utf-8').strip().split('\n')
        else:
            with open(meta_file, 'r') as f:
                lines = f.readlines()

        for line in lines:
            line_list = line.strip().split(' ')
            name = line_list[1]
            idx = int(name)
            classes.append(name)
            class_to_idx[name] = idx
            samples.append((os.path.join(directory, line_list[0]), idx))
        return classes, class_to_idx, samples
    
    def ceph_load(self, path):
        try:
            value = self.ceph_client.Get(path)
            assert value is not None, path
            img = self.bytes_to_img(value)
        except Exception as e:  # noqa
            value = self.ceph_client.Get(path, update_cache=True)
            assert value is not None, path
            img = self.bytes_to_img(value)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        return im_pil

    def bytes_to_img(self, value):
        img_array = np.frombuffer(value, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert img is not None
        return img

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImageFolderSensetime(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        if 's3://' in root:
            use_ceph = True
        else:
            use_ceph = False

        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            use_ceph=use_ceph
        )
        self.imgs = self.samples
