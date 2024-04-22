---
title: Comparing representations
nav_order: 8
---

# RSA tools

We provide some basic functions to conduct a similarity analysis on the extracted features. Note that these provide only a basic functionality and are not optimized for speed. For more advanced and optimized analysis tools, we recommend to use the [rsatoolbox](https://rsatoolbox.readthedocs.io/en/latest/) library.

## Representational Similarity Analysis (RSA) 

Compare representational (dis-)similarity matrices (RDMs) corresponding to model features and human representations (e.g., fMRI recordings).

```python
from thingsvision.core.rsa import compute_rdm, correlate_rdms

rdm_dnn = compute_rdm(features, method='correlation')
corr_coeff = correlate_rdms(rdm_dnn, rdm_human, correlation='pearson')
```

## Centered Kernel Alignment (CKA)

Perform CKA to compare image features of two different model architectures for the same layer, or two different layers of the same architecture.

```python
from thingsvision.core.cka import get_cka

backend = "torch" # can be set to either 'torch' or 'numpy'
m = # number of images (e.g., features_i.shape[0])
kernel = 'linear' # linear or rbf kernel (for rbf kernel define sigma, i.e., the width of the Gaussian)
unbiased = True # whether to compute an unbiased version of CKA
device = "cuda" # only necessary to be defined for the 'torch' backend (NumPy runs on CPU only)
sigma = None # needs to be defined for 'rbf' kernel

cka = get_cka(backend=backend, m=m, kernel=kernel, unbiased=unbiased, device=device, sigma=sigma)
rho = cka.compare(X=features_i, Y=features_j)
```
