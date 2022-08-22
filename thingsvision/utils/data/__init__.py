import os
from typing import Iterator, List

from .dataloader import DataLoader
from .dataset import ImageDataset


def load_dl(
    root: str,
    out_path: str,
    backend: str,
    batch_size: int,
    class_names: List[str] = None,
    file_names: List[str] = None,
    transforms=None,
) -> Iterator:
    """Create a data loader for custom image dataset

    Parameters
    ----------
    root : str
        Root directory. Directory where images are stored.
    out_path : str
        PATH where order of images features should be stored.
    batch_size : int (optional)
        Number of samples (i.e., images) per mini-batch.
    class_names : List[str] (optional)
        Explicit list of class names.
        Used to control the order of the classes (otherwise alphanumerical order is used).
    file_names : List[str] (optional)
        List of file names. A list of file names that determines
        the order in which image features are extracted can optionally
        be passed.
    transforms : Any
        Composition of image transformations. Must be either a PyTorch composition
        or a Tensorflow Sequential model.

    Returns
    -------
    output : Iterator
        Returns an iterator of image mini-batches.
        Each mini-batch consists of <batch_size> samples.
    """
    print("\n...Creating dataset.")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("...Creating output directory.")
    dataset = ImageDataset(
        root=root,
        out_path=out_path,
        backend=backend,
        class_names=class_names,
        file_names=file_names,
        transforms=transforms,
    )
    print(f"...Transforming dataset into {backend} DataLoader.\n")
    batches = DataLoader(dataset=dataset, batch_size=batch_size, backend=backend)
    return batches
