__all__ = ["Transform"]

import os
from typing import Any

import numpy as np
import requests

Array = np.ndarray

gLocal_URL = "https://raw.githubusercontent.com/LukasMut/gLocal/main/transforms/"


class Transform:
    def __init__(
        self, model_name: str, module_name: str, alignment_type: str = "gLocal"
    ) -> None:
        self.model_name = model_name
        self.module_name = module_name
        if alignment_type == "gLocal":
            self.url = os.path.join(
                gLocal_URL, self.model_name, self.module_name, "transform.npz"
            )
            self.transform = self._load_transform()
        else:
            raise NotImplementedError(
                f"\nRepresentational alignment of type: {alignment_type} is not yet implemented.\nChange to gLocal!\n"
            )

    def _load_transform(self) -> Any:
        # Download the transform
        response = requests.get(self.url)
        # Check for successful download of transform
        if response.status_code == requests.codes.ok:
            # Write the content of transform to a temporary file
            with open("temp.npz", "wb") as f:
                f.write(response.content)
            transform = np.load("temp.npz")
            os.remove("temp.npz")
        else:
            raise FileNotFoundError(
                f"\nError downloading transform: {response.status_code}\nModel: {self.model_name}, Module: {self.module_name}\n"
            )
        return transform

    def apply_transform(self, features: Array) -> Array:
        features = (features - self.transform["mean"]) / self.transform["std"]
        features = features @ self.transform["weights"]
        if "bias" in self.transform:
            features += self.transform["bias"]
        return features
