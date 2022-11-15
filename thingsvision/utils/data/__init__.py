from typing import Iterator, List

from .data_loader import DataLoader
from .dataset import HDF5Dataset, ImageDataset


def load_batches(
    root: str,
    out_path: str,
    backend: str,
    batch_size: int,
    class_names: List[str] = None,
    file_names: List[str] = None,
    transforms=None,
) -> Iterator:
    """Create a data loader that yields mini-batches of size <batch_size>

    Parameters
    ----------
    root : str
        Root directory. Directory from where to load the image files.
    out_path : str
        Directory where the order of the image features should be stored.
    backend: str
        Backend of a neural network model. Must be PyTorch ('pt') or TensorFlow/Keras ('tf').
    batch_size : int
        Number of samples (i.e., images) per mini-batch.
    class_names : List[str] (optional)
        Explicit list of class names.
        Used to control the order of the classes (otherwise alphanumerical order is used).
    file_names : List[str] (optional)
        List of file names. A list of file names that determines
        the order in which image features are extracted can optionally
        be passed.
    transforms : Any
        Composition of image transformations. Transformations are determined by a model and its backend.
        Must be either a PyTorch composition or a Tensorflow Sequential model.

    Returns
    -------
    output : Iterator
        Returns an iterator of mini-batches.
        Each mini-batch consists of <batch_size> samples.
        The order is determined by <file_names>, <class_names> or is alphanumeric.
    """
    dataset = ImageDataset(
        root=root,
        out_path=out_path,
        backend=backend,
        class_names=class_names,
        file_names=file_names,
        transforms=transforms,
    )
    print(
        f"...Transforming dataset into a {backend} DataLoader for batch-wise feature extraction.\n"
    )
    batches = DataLoader(dataset=dataset, batch_size=batch_size, backend=backend)
    return batches
