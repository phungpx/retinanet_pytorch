import torch
import random
from typing import List
from torch.utils.data.sampler import Sampler


__all__ = ['AspectRatioBasedSampler', 'Collater']


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle, drop_last):
        super(AspectRatioBasedSampler, self).__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batch_groups = self.group_batch_indices

        if shuffle:
            random.shuffle(self.batch_groups)

    def __iter__(self):
        for batch_indices in self.batch_groups:
            yield batch_indices

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    @property
    def group_batch_indices(self) -> List[List[int]]:
        # determine the order of indices of all images
        indices = list(range(len(self.data_source)))
        indices = sorted(indices, key=lambda idx: self.data_source.image_aspect_ratio(idx))

        # divide image index into groups of batch, one group = one batch
        batch_indices_groupds = [
            [indices[x % len(indices)] for x in range(i, i + self.batch_size)]
            for i in range(0, len(indices), self.batch_size)
        ]

        return batch_indices_groupds


class Collater:
    def __call__(self, batch_data):
        batch_size = len(batch_data)
        samples = [data[0] for data in batch_data]  # List[Tensor[3 x H x W]]
        targets = tuple([data[1] for data in batch_data])  # List[Dict[str, torch.Tensor]]
        image_infos = tuple([data[2] for data in batch_data])  # List[Tuple[str, Tuple[int, int]]]

        max_H = max([int(sample.shape[1]) for sample in samples])
        max_W = max([int(image.shape[2]) for image in samples])

        padded_samples = torch.zeros(size=(batch_size, 3, max_H, max_W))
        for i in range(batch_size):
            sample = samples[i]
            padded_samples[i, :, :int(sample.shape[1]), :int(sample.shape[2])] = sample

        return padded_samples, targets, image_infos
