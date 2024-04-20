import abc
from typing import Optional, Union

import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor


class CKABase(metaclass=abc.ABCMeta):
    def __init__(
        self, m: int, kernel: str, unbiased: bool = False, sigma: Optional[float] = None
    ) -> None:
        self.m = m  # number of examples
        self.kernel = kernel  # linear or rbf kernel
        self.unbiased = unbiased  # whether to use the unbiased version of CKA
        self.sigma = sigma
        self.H = self.centering_matrix(m)

    @abc.abstractmethod
    def centering_matrix(m: int) -> Union[Array, Tensor]:
        """Compute the centering matrix H."""
        raise NotImplementedError

    @abc.abstractmethod
    def centering(self, K: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Centering of the gram matrix K."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_kernel(self, X: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Compute the gram matrix K."""
        raise NotImplementedError

    @abc.abstractmethod
    def linear_kernel(self, X: Union[Array, Tensor]) -> Union[Array, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def rbf_kernel(
        self, X: Union[Array, Tensor], sigma: float = None
    ) -> Union[Array, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def _hsic(
        self, X: Union[Array, Tensor], Y: Union[Array, Tensor]
    ) -> Union[Array, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def compare(
        self, X: Union[Array, Tensor], Y: Union[Array, Tensor]
    ) -> Union[Array, Tensor]:
        """Compare the two representation spaces X and Y."""
        raise NotImplementedError
