import os
import re
import warnings
from typing import Dict, List

EXTENSIONS = r"(.eps|.jpg|.JPG|.jpeg|.JPEG|.png|.PNG|.tif|.tiff)$"


def parse_img_name(img_name: str) -> bool:
    """Check whether image file has allowed extension."""
    return re.search(EXTENSIONS, img_name)


def rm_suffix(img_name: str) -> str:
    """Remove suffix from image file."""
    return re.sub(EXTENSIONS, "", img_name)


def make_instance_dataset(
    root: str,
    out_path: str,
    image_names: List[str],
) -> List[str]:
    """Creates a custom <instance> image dataset of images and writes its order to file."""
    instances = []
    with open(os.path.join(out_path, "file_names.txt"), "w") as f:
        for image_name in image_names:
            f.write(f"{image_name}\n")
            instances.append(os.path.join(root, image_name))
    return instances


def make_class_dataset(
    in_path: str,
    out_path: str,
    cls_to_idx: Dict[str, int],
    class_names: List[str] = None,
    cls_to_files: Dict[str, List[str]] = None,
) -> List[str]:
    """Creates a custom <class> image dataset of image and class label pairs.

    Parameters
    ----------
    in_path : str
        Root directory. Directory from where to load the image files.
    out_path : str
        path/to/filenames.
    cls_to_idx : Dict[str, int]
        Dictionary of class to numeric label mapping.
    class_names: List[str]
        List of class names according to which samples are sorted.
    cls_to_files : Dict[str, List[str]] (optional)
        Dictionary that maps each class to a list of file names.
        For each class, the list of file names determines
        the order in which image features are extracted.

    Returns
    -------
    output : List[str]
        Returns a list of image file names.
    """
    samples = []
    with open(os.path.join(out_path, "file_names.txt"), "w") as f:
        classes = cls_to_idx.keys() if class_names else sorted(cls_to_idx.keys())
        for target_cls in classes:
            target_dir = os.path.join(in_path, target_cls)
            if os.path.isdir(target_dir):
                if cls_to_files is None:
                    for root, _, files in sorted(os.walk(target_dir, followlinks=True)):
                        for file in sorted(files):
                            path = os.path.join(root, file)
                            if os.path.isfile(path) and parse_img_name(file):
                                samples.append(path)
                                f.write(f"{path}\n")
                else:
                    for f_name in cls_to_files[target_cls]:
                        path = os.path.join(target_dir, f_name)
                        if os.path.isfile(path) and parse_img_name(f_name):
                            samples.append(path)
                            f.write(f"{path}\n")
            else:
                warnings.warn(f"\nDirectory for class <{target_cls}> does not exist.\n")
    return samples
