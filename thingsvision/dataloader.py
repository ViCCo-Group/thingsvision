#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from typing import Iterator, Tuple


class DataLoader(object):

    def __init__(
        self,
        dataset: Tuple,
        batch_size: int,
        backend: str,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.backend = backend
        self.n_batches = len(self.dataset) // self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator:
        return self.get_batches(self.dataset)

    def get_batches(self, dataset: Tuple) -> Iterator:
        for k in range(self.n_batches):
            X, y = zip(
                *[dataset[i] for i in range(k * self.batch_size, (k + 1) * self.batch_size)])
            if self.backend == 'pt':
                X = torch.stack(X, dim=0)
                y = torch.stack(y, dim=0)
            else:
                # TODO: implement the same for TensorFlow
                raise NotImplementedError
            yield (X, y)
