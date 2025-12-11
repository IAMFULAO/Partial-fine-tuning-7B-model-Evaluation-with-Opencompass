import os
from os import environ
from typing import List
import datasets
from ..base import BaseDataset
from opencompass.registry import LOAD_DATASET

@LOAD_DATASET.register_module()
class C4Dataset(BaseDataset):
    @staticmethod
    def load(path: str, num: int = None, **kwargs) -> datasets.Dataset:
        dataset = datasets.load_from_disk(path)
        if num is not None:
            dataset = dataset.select(range(min(num, len(dataset))))
        return dataset