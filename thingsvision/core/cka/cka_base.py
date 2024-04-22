import abc
from typing import Optional, Union

import numpy as np
import torch

Array = np.ndarray
Tensor = torch.Tensor


class CKABase(metaclass=abc.ABCMeta):
    def __init__(
        self, m: int, kernel: str, unbiased: bool = False, sigma: Optional[float] = 1.0
    ) -> None:
        """
        CKA abstract base class from which other CKA classes inherit.
        Args:
            m (int) - number of images / examples in a mini-batch or the full dataset;
            kernel (str) - 'linear' or 'rbf' kernel for computing the gram matrix;
            unbiased (bool) - whether to compute an unbiased version of CKA;
            sigma (float) - for 'rbf' kernel sigma defines the width of the Gaussian;
        """
        self.m = m  # number of examples
        self.kernel = kernel  # linear or rbf kernel
        self.unbiased = unbiased  # whether to use the unbiased version of CKA
        self.sigma = sigma  # width of the Gaussian in an rbf kernel

    @abc.abstractmethod
    def centering(self, K: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Centering of the (square) gram matrix K."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply_kernel(self, X: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Compute the (square) gram matrix K."""
        raise NotImplementedError

    @abc.abstractmethod
    def linear_kernel(self, X: Union[Array, Tensor]) -> Union[Array, Tensor]:
        """Use a linear kernel for computing the gram matrix."""
        raise NotImplementedError

    @abc.abstractmethod
    def rbf_kernel(
        self, X: Union[Array, Tensor], sigma: float = None
    ) -> Union[Array, Tensor]:
        """Use an rbf kernel for computing the gram matrix. Sigma defines the width."""
        raise NotImplementedError

    @abc.abstractmethod
    def _hsic(
        self, X: Union[Array, Tensor], Y: Union[Array, Tensor]
    ) -> Union[Array, Tensor]:
        """Compute the Hilbert-Schmidt independence criterion."""
        raise NotImplementedError

    @abc.abstractmethod
    def compare(
        self, X: Union[Array, Tensor], Y: Union[Array, Tensor]
    ) -> Union[Array, Tensor]:
        """Compare two representation spaces X and Y using CKA."""
        raise NotImplementedError
