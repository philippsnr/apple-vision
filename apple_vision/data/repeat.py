from __future__ import annotations

from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    """Oversample a dataset by an integer factor ("augmentation factor").

    The length becomes ``len(base) * factor``. Each of the ``factor`` accesses
    of a given base image is drawn independently through the base dataset, so
    with *online* augmentation this yields ``factor`` differently-augmented
    views of every image per epoch (e.g. factor=4 -> 4 augmented copies per
    image each epoch).

    Note: with augmentation disabled the copies are identical, so a factor > 1
    only makes sense together with training augmentation.
    """

    def __init__(self, base: Dataset, factor: int) -> None:
        if factor < 1:
            raise ValueError(f"factor must be >= 1, got {factor}")
        self.base = base
        self.factor = int(factor)

    def __len__(self) -> int:
        return len(self.base) * self.factor

    def __getitem__(self, idx: int):
        return self.base[idx % len(self.base)]
