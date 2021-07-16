#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Iterator, Tuple


class DataLoader(object):

    def __init__(
        self,
        dataset: Tuple,
        batch_size: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = len(self.dataset) // self.batch_size

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator:
        return self.get_batches(self.dataset)

    def get_batches(self, dataset: Tuple) -> Iterator:
        for i in range(self.n_batches):
            batch = dataset[i * self.batch_size: (i + 1) * self.batch_size]
            yield batch
