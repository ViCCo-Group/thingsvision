from typing import Optional, Union

from .cka_numpy import CKANumPy
from .cka_torch import CKATorch

BACKENDS = ["numpy", "torch"]


def get_cka(
    backend: str,
    m: int,
    kernel: str,
    unbiased: bool = False,
    sigma: Optional[float] = None,
    device: Optional[str] = None,
) -> Union[CKANumPy, CKATorch]:
    assert backend in BACKENDS, f"\nSupported backends are: {BACKENDS}\n"
    if backend == "numpy":
        cka = CKANumPy(m=m, kernel=kernel, unbiased=unbiased, sigma=sigma)
    else:
        assert isinstance(
            device, str
        ), "\nDevice must be set for using the PyTorch backend.\n"
        cka = CKATorch(
            m=m, kernel=kernel, unbiased=unbiased, device=device, sigma=sigma
        )
    return cka
