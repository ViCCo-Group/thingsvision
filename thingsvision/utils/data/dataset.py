import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import h5py
import tensorflow as tf
from PIL import Image

from .helpers import make_class_dataset, make_instance_dataset, parse_img_name


@dataclass
class ImageDataset:
    """Generic image dataset class for PyTorch and TensorFlow

    Params
    ----------
    root : str
        Root directory. Directory from where to load image files.
    out_path : str
        Directory where the order of the image features should be stored.
    backend: str
        Backend of a neural network model. Must be PyTorch ('pt') or TensorFlow/Keras ('tf).
    class_names : List[str] (optional)
        Explicit list of class names.
        Used to control the order of the classes (otherwise alphanumerical order is used).
    file_names : List[str] (optional)
        List of file names. Used to control the order in which image features are extracted.
    transforms : Any
        Composition of image transformations. Must be either a PyTorch composition
        or a Tensorflow Sequential model.

    Returns
    -------
    output : Dataset[Any]
        Returns a generic image dataset of image instances
        (i.e., image in matrix format)
    """

    root: str
    out_path: str
    backend: str
    class_names: List[str] = None
    file_names: List[str] = None
    transforms: Any = None

    def __post_init__(self) -> None:
        print("\n...Creating dataset.")
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path, exist_ok=True)
            print("...Output directory does not exist.")
            print("...Creating output directory to save order of file names.\n")

        self._find_classes()
        if self.type == "class_dataset":
            if self.file_names:
                cls_to_files = self._get_classes(self.file_names)
                self.samples = make_class_dataset(
                    in_path=self.root,
                    out_path=self.out_path,
                    cls_to_idx=self.cls_to_idx,
                    class_names=self.class_names,
                    cls_to_files=cls_to_files,
                )
            else:
                self.samples = make_class_dataset(
                    in_path=self.root,
                    out_path=self.out_path,
                    cls_to_idx=self.cls_to_idx,
                    class_names=self.class_names,
                )
        else:
            assert not (
                isinstance(self.class_names, list) and isinstance(self.file_names, list)
            ), "\nFor an instance dataset, only use the <file_names> argument and leave <class_names> argument empty.\n"
            self.samples = make_instance_dataset(
                root=self.root, out_path=self.out_path, image_names=self.file_names
            )

    def _find_classes(self) -> None:
        children = sorted([d.name for d in os.scandir(self.root) if d.is_dir()])
        if children:
            setattr(self, "type", "class_dataset")
            if self.class_names:
                setattr(self, "classes", self.class_names)
            elif self.file_names:
                _ = self._get_classes()
            else:
                setattr(self, "classes", children)
            self.idx_to_cls = dict(enumerate(self.classes))
            self.cls_to_idx = {cls: idx for idx, cls in self.idx_to_cls.items()}
        else:
            setattr(self, "type", "instance_dataset")
            if not self.file_names:
                self.file_names = sorted(
                    [
                        f.name
                        for f in os.scandir(self.root)
                        if f.is_file() and parse_img_name(f.name)
                    ]
                )

    def _get_classes(self) -> Dict[str, list]:
        cls_to_files = defaultdict(list)
        classes = []
        for file in self.file_names:
            if re.search(r"\\", file):
                cls, f_name = file.split("\\")
            elif re.search(r"(/|//)", file):
                cls, f_name = file.split("/")
            else:
                continue
            if cls not in classes:
                classes.append(cls)
            cls_to_files[cls].append(f_name)
        setattr(self, "classes", classes)
        return cls_to_files

    def __getitem__(self, idx: int) -> Any:
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self._transform_image(img)
        return img

    def _transform_image(self, img: Any) -> Any:
        if self.backend == "pt":
            img = self.transforms(img)
        elif self.backend == "tf":
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = self.transforms(img)
        else:
            raise ValueError(
                "\nImage transformations only implemented for PyTorch or TensorFlow/Keras models.\n"
            )
        return img

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def images(self):
        return self.samples


@dataclass
class HDF5Dataset:
    """Generic HDF5 dataset class for PyTorch and TensorFlow

    Params
    ----------
    hdf5_fp : str
        Path to the HDF5 file. Must contain images in the <img_ds_key> dataset.
    img_ds_key : str
        Key to the HDF5 dataset containing the images.
    backend: str
        Backend of a neural network model. Must be PyTorch ('pt') or TensorFlow/Keras ('tf).
    transforms : Any
        Composition of image transformations. Must be either a PyTorch composition
        or a Tensorflow Sequential model.
    img_indices : List[int] (optional)
        Only extract features for images with the given indices from the HDF5 file.

    Returns
    -------
    output : Dataset[Any]
        Returns a generic image dataset of image instances
        (i.e., image in matrix format)

    Example for NSD dataset
    -------
    >>> from thingsvision.utils.data import HDF5Dataset
    >>>
    >>> # get indices of all 10000 images shown to first subject
    >>> img_indices = np.unique(
    >>>     experiment['subjectim'][:, experiment['masterordering'][0] - 1][0]
    >>> )
    >>>
    >>> dataset = HDF5Dataset(
    >>>     hdf5_fp="<path_to_nsd>/nsddata_stimuli/stimuli/nsd_stimuli.hdf5",
    >>>     img_ds_key="imgBrick",
    >>>     transforms=extractor.get_transformations(),
    >>>     backend=extractor.backend,
    >>>     img_indices=img_indices
    >>> )
    """

    hdf5_fp: str
    img_ds_key: str
    backend: str
    transforms: Any = None
    img_indices: List[int] = None

    def __post_init__(self) -> None:
        self._load_hdf5()

    def _load_hdf5(self) -> None:
        self.samples = h5py.File(self.hdf5_fp, "r")
        self.n_samples = (
            self.samples[self.img_ds_key].shape[0]
            if self.img_indices is None
            else len(self.img_indices)
        )

    def __getitem__(self, idx: int) -> Any:
        if self.img_indices is not None:
            idx = self.img_indices[idx]
        img_np = self.samples[self.img_ds_key][idx]
        img = Image.fromarray(img_np).convert("RGB")
        img = self._transform_image(img)
        return img

    def _transform_image(self, img: Any) -> Any:
        if self.backend == "pt":
            img = self.transforms(img)
        elif self.backend == "tf":
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = self.transforms(img)
        else:
            raise ValueError(
                "\nImage transformations only implemented for PyTorch or TensorFlow/Keras models.\n"
            )
        return img

    def __len__(self) -> int:
        return self.n_samples

    @property
    def images(self):
        return self.samples
