from __future__ import annotations

from .ms1mv3 import MS1MV3Dataset


class WebFace4MDataset(MS1MV3Dataset):
    dataset_name = "webface4m"


class WebFace12MDataset(MS1MV3Dataset):
    dataset_name = "webface12m"
