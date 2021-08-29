import torch.nn as nn


class MLPBikePrediction(nn.Module):
    """
    2 layer Multilayer perceptron with ReLU activations which predicts the casual and registered users
    """
    def __init__(self):
        super(MLPBikePrediction, self).__init__()

        embedding_size_1 = 64
        embedding_size_2 = 16
        self.mlp = nn.Sequential(
            nn.Linear(60, embedding_size_1),
            nn.ReLU(),
            nn.Linear(embedding_size_1, embedding_size_2),
            nn.ReLU(),
            nn.Linear(embedding_size_2, 2),
        )

    def forward(self, h):
        return self.mlp(h)
