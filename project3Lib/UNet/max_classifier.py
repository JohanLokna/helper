import torch
from torch import nn

class MaxClassifier(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.threshold = None

    def fit(self, train_dataset):

        sample = torch.concat([x for x, _, _ in train_dataset])
        label = torch.BoolTensor([label for x, _, label in train_dataset]).to(device=sample.device)

        thresholds = torch.arange(0.5, 1, 0.025).to(device=sample.device)
        scores = torch.empty_like(thresholds)

        for i in range(torch.numel(thresholds)):
            self.threshold = thresholds[i]
            scores[i] = torch.count_nonzero(self(sample) == label)

        self.threshold = torch.mean(thresholds[scores == scores.max()])
    
    def __call__(self, x):
        pred = self.model(x)
        mask = (pred > self.threshold).view((pred.shape[0], -1))
        return torch.any(mask, dim=1)
