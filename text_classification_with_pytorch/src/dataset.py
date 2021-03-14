import torch

class IMDBDataset:
    def __init__(self, reviews, targets):
        """
        : param reviews: this is an numpy array
        : param targets: a vector, numpy array

        """

        self.reviews = reviews
        self.targets = targets

    def __len__(self):
        # returns the len of the dataset
        return len(self.reviews)

    def __getitem__(self, idx):
        # any given idx, returns reviews & targets
        review = self.reviews[idx, :]
        target = self.targets[idx]

        return {
            
            "review": torch.tensor(review, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.float)
        }
