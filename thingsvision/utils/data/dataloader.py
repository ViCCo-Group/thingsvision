#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
from typing import Any, Iterator, Tuple

import tensorflow as tf
import torch


@dataclass
class DataLoader:
    dataset: Tuple
    batch_size: int
    backend: str

    def __post_init__(self) -> None:
        self.n_batches = int(math.ceil(len(self.dataset) / self.batch_size))
        self.remainder = len(self.dataset) % self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator:
        return self.get_batches(self.dataset)

    def stack_samples(self, images: Any) -> Tuple:
        batch = (
            torch.stack(images, dim=0)
            if self.backend == "pt"
            else tf.stack(images, axis=0)
        )
        return batch

    def get_batches(self, dataset: Tuple) -> Iterator:
        for k in range(self.n_batches):
            if self.remainder != 0 and k == int(self.n_batches - 1):
                subset = range(
                    k * self.batch_size, k * self.batch_size + self.remainder
                )
            else:
                subset = range(k * self.batch_size, (k + 1) * self.batch_size)
            X = [dataset[i] for i in subset]
            batch = self.stack_samples(X)
            yield batch
