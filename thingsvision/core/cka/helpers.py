from typing import Optional, Union

from .cka_numpy import CKANumPy
from .cka_torch import CKATorch

BACKENDS = ["numpy", "torch"]


def get_cka(
    backend: str,
    m: int,
    kernel: str = "linear",
    unbiased: bool = False,
    sigma: Optional[float] = 1.0,
    device: Optional[str] = None,
    verbose: Optional[bool] = False,
) -> Union[CKANumPy, CKATorch]:
    """Return a NumPy or PyTorch implementation of CKA."""
    assert backend in BACKENDS, f"\nSupported backends are: {BACKENDS}\n"
    if backend == "numpy":
        cka = CKANumPy(m=m, kernel=kernel, unbiased=unbiased, sigma=sigma)
    else:
        assert isinstance(
            device, str
        ), "\nDevice must be set for using PyTorch backend.\n"
        cka = CKATorch(
            m=m,
            kernel=kernel,
            unbiased=unbiased,
            device=device,
            sigma=sigma,
            verbose=verbose,
        )
    return cka
