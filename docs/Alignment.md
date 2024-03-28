---
layout: default
title: Human alignment
nav_order: 7
---

# Aligning neural network representations with human similarity judgments

Recent research in the space of representation learning has demonstrated the usefulness of aligning neural network representations with human similarity judgments for both machine learning dowsntream tasks and the Cognitive Sciences (see [here]((https://proceedings.neurips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)) and [here](https://arxiv.org/pdf/2310.13018.pdf)).

## [gLocal](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)

If you want to align the extracted representations with human object similarity according to the approach introduced in *[Improving neural network representations using human similiarty judgments](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9febda1c8344cc5f2d51713964864e93-Abstract-Conference.html)* you can optionally `align` the extracted features using the following method:

```python
aligned_features = extractor.align(
    features=features
    module_name=module_name,
    alignment_type="gLocal",
)
```

For now, representational alignment is only implemented for `gLocal` and the following list of models: `clip_RN50`, `clip_ViT-L/14`, `OpenCLIP_ViT-L-14_laion400m_e32`, `OpenCLIP_ViT-L-14_laion2b_s32b_b82k` `dinov2-vit-base-p14`, `dinov2-vit-large-p14`, `dino-vit-base-p16`, `dino-vit-base-p8`. However, we plan to extend both the type of representation alignment and the range of models in future versions of `thingsvision`.
