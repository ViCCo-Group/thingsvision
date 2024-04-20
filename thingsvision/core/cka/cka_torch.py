import re
import warnings
from typing import Optional, Union

import torch
from torchtyping import TensorType

from .base import CKABase


class CKATorch(CKABase):
    def __init__(
        self, m: int, kernel: str, device: str = "cpu", sigma: Optional[float] = None
    ) -> None:
        super().__init__(m=m, kernel=kernel, sigma=sigma)
        self.device = device
        self._check_device()
        if self.device == "cpu":
            self.hsic = self._hsic
        else:
            self.hsic = torch.compile(self._hsic)
        self.device = torch.device(self.device)
        self.H = self.H.to(self.device)

    def _check_device(self) -> None:
        if self.device.startswith("cuda"):
            gpu_index = re.search(r"cuda:(\d+)", self.device)

            if not torch.cuda.is_available():
                warnings.warn(
                    "\nCUDA is not available on your system. Switching to device='cpu'.\n",
                    category=UserWarning,
                )
                self.device = "cpu"
            elif gpu_index and int(gpu_index.group(1)) >= torch.cuda.device_count():
                warnings.warn(
                    f"\nGPU index {gpu_index.group(1)} is out of range. "
                    f"Available GPUs: {torch.cuda.device_count()}. "
                    f"Switching to device='cuda:0'.\n",
                    category=UserWarning,
                )
                self.device = "cuda:0"

        print("Using device: ", self.device)

    @staticmethod
    def centering_matrix(m: int) -> TensorType["m", "m"]:
        """Compute the centering matrix H."""
        H = torch.eye(m) - torch.ones((m, m)) / m
        return H

    def centering(self, K: TensorType["m", "m"]) -> TensorType["m", "m"]:
        """Centering of the gram matrix K."""
        if not torch.allclose(K, K.T, rtol=1e-03, atol=1e-04):
            raise ValueError("\nInput array must be a symmetric matrix.\n")
        K_c = self.H @ K @ self.H
        return K_c

    def apply_kernel(
        self, X: Union[TensorType["m", "d"], TensorType["m", "p"]]
    ) -> TensorType["m", "m"]:
        """Compute the gram matrix K."""
        try:
            K = getattr(self, f"{self.kernel}_kernel")(X)
        except AttributeError:
            raise NotImplementedError
        return K

    def linear_kernel(
        self, X: Union[TensorType["m", "d"], TensorType["m", "p"]]
    ) -> TensorType["m", "m"]:
        return X @ X.T

    def rbf_kernel(
        self, X: TensorType["m", "d"], sigma: float = None
    ) -> TensorType["m", "m"]:
        GX = X @ X.T
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = torch.sqrt(mdist)
        KX *= -0.5 / sigma**2
        KX = KX.exp()
        return KX

    def _hsic(
        self, X: TensorType["m", "d"], Y: TensorType["m", "p"]
    ) -> TensorType["1"]:
        K = self.apply_kernel(X)
        L = self.apply_kernel(Y)
        K_c = self.centering(K)
        L_c = self.centering(L)
        # np.sum(K_c * L_c) is equivalent to K_c.flatten() @ L_c.flatten() or in math
        # sum_{i=0}^{m} sum_{j=0}^{m} K^{\prime}_{ij} * L^{\prime}_{ij} = vec(K_c)^{T}vec(L_c)
        return torch.sum(K_c * L_c)

    @torch.inference_mode()
    def compare(
        self, X: TensorType["m", "d"], Y: TensorType["m", "p"]
    ) -> TensorType["1"]:
        X = X.to(self.device)
        Y = Y.to(self.device)
        hsic_xy = self.hsic(X, Y)
        hsic_xx = self.hsic(X, X)
        hsic_yy = self.hsic(Y, Y)
        rho = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
        if rho.is_cuda:
            rho = rho.cpu()
        return rho
