---
title: Tools
nav_order: 4
---



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
from thingsvision.core.cka import CKA

m = # number of images (e.g., features_i.shape[0])
kernel = 'linear'
cka = CKA(m=m, kernel=kernel)
rho = cka.compare(X=features_i, Y=features_j)
```
