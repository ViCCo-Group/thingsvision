__all__ = ["gLocal"]

import abc
import os
from typing import Any, Union

import numpy as np
import requests
import torch

Array = np.ndarray
Tensor = torch.Tensor

gLocal_URL = "https://raw.githubusercontent.com/LukasMut/gLocal/main/transforms/"


class Transform(metaclass=abc.ABCMeta):
    def __init__(
        self, model_name: str, module_name: str, alignment_type: str = "gLocal"
    ) -> None:
        self.model_name = model_name
        self.module_name = module_name
        if alignment_type != "gLocal":
            raise NotImplementedError(
                f"\nRepresentational alignment of type: {alignment_type} is not yet implemented.\nChange type to gLocal!\n"
            )

    @abc.abstractmethod
    def load_transform_from_remote(self) -> Any:
        """Load transformation from remote."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_transform(self, features: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Apply (affine) transformation to a model's representation space."""
        raise NotImplementedError


class gLocal(Transform):
    def __init__(self, model_name: str, module_name: str) -> None:
        super().__init__(
            model_name=model_name, module_name=module_name, alignment_type="gLocal"
        )
        self.url = os.path.join(
            gLocal_URL, self.model_name, self.module_name, "transform.npz"
        )
        self.transform = self.load_transform_from_remote()

    def load_transform_from_remote(self) -> Any:
        """Load gLocal (affine) transform from official gLocal GitHub repo."""
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

    def apply_transform(self, features: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Apply the gLocal transform to a model's representation space."""
        features = (features - self.transform["mean"]) / self.transform["std"]
        features = features @ self.transform["weights"]
        if "bias" in self.transform:
            features += self.transform["bias"]
        return features
