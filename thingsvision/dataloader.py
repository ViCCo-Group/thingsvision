#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import torch
import math

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
        self.n_batches = int(math.ceil(len(self.dataset) / self.batch_size))
        self.remainder = len(self.dataset) % self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator:
        return self.get_batches(self.dataset)

    def stack_samples(self, X, y) -> Tuple:
        if self.backend == 'pt':
            X = torch.stack(X, dim=0)
            y = torch.stack(y, dim=0)
        else:
            X = tf.stack(X, axis=0)
            y = tf.stack(y, axis=0)
        return (X, y)

    def get_batches(self, dataset: Tuple) -> Iterator:
        for k in range(self.n_batches):
            if (self.remainder != 0 and k == int(self.n_batches - 1)):
                subset = range(k * self.batch_size, k *
                               self.batch_size + self.remainder)
            else:
                subset = range(k * self.batch_size, (k + 1) * self.batch_size)
            X, y = zip(
                *[dataset[i] for i in subset])
            batch = self.stack_samples(X, y)
            yield batch
