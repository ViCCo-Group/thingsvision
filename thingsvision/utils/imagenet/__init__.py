import json
import os
import re
from typing import Dict, List

import numpy as np

Array = np.ndarray


def parse_imagenet_synsets(root: str) -> List[str]:
    """Convert wordnet synsets into classes."""

    def parse_str(str):
        return re.sub(r"[^a-zA-Z]", "", str).rstrip("n").lower()

    imagenet_synsets = []
    with open(root, "r") as file:
        for line in file:
            line = line.split("_")
            cls = "_".join(list(map(parse_str, line)))
            imagenet_synsets.append(cls)
    return imagenet_synsets


def parse_imagenet_classes(root: str) -> List[str]:
    """Disambiguate ImageNet classes."""
    imagenet_classes = []
    with open(root, "r") as file:
        for line in file:
            line = line.strip().split()
            cls = "_".join(line[1:]).rstrip(",").strip("'").lower()
            cls = cls.split(",")
            cls = cls[0]
            imagenet_classes.append(cls)
    return imagenet_classes


def get_cls_mapping_imagenet(root: str, save_as_json: bool = False) -> dict:
    """store imagenet classes in an *index_to_class* dictionary, and subsequently save as .json file."""
    if re.search(r"synset", root.split("/")[-1]):
        imagenet_classes = parse_imagenet_synsets(root)
    else:
        imagenet_classes = parse_imagenet_classes(root)
    idx2cls = dict(enumerate(imagenet_classes))
    if save_as_json:
        filename = "imagenet_idx2class.json"
        root = "/".join(root.split("/")[:-1])
        with open(os.path.join(root, filename), "w") as f:
            json.dump(idx2cls, f)
    return idx2cls


def get_class_probabilities(
    probas: Array,
    out_path: str,
    cls_file: str,
    top_k: int,
    save_as_json: bool,
) -> Dict[str, Dict[str, float]]:
    """Compute probabilities per ImageNet class."""
    file_names = open(os.path.join(out_path, "file_names.txt"), "r").read().splitlines()
    idx2cls = get_cls_mapping_imagenet(cls_file)
    class_probas = {}
    for i, (file, p_i) in enumerate(zip(file_names, probas)):
        sorted_predictions = np.argsort(-p_i)[:top_k]
        class_probas[file] = {
            idx2cls[pred]: float(p_i[pred]) for pred in sorted_predictions
        }
    if save_as_json:
        with open(os.path.join(out_path, "class_probabilities.json"), "w") as f:
            json.dump(class_probas, f)
    return class_probas
