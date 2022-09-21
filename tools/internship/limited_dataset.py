import numpy as np

class LimitedDataset:
    def __init__(self,
                 dataset,
                 num=100,
                 seed=12345):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.num = num

        self.indices = list(range(len(self.dataset)))
        rng = np.random.default_rng(seed)
        rng.shuffle(self.indices)
        self.indices = self.indices[:self.num]

    def get_cat_ids(self, idx):
        return self.dataset.get_cat_ids(self.indices[idx])

    def get_gt_labels(self):
        dataset_gt_labels = self.dataset.get_gt_labels()
        gt_labels = np.array([dataset_gt_labels[idx] for idx in self.indices])
        return gt_labels

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def evaluate(self, *args, **kwargs):
        kwargs['indices'] = self.indices
        return self.dataset.evaluate(*args, **kwargs)
