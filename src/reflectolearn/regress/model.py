import torch.nn as nn


class ThicknessRegressor(nn.Module):
    """
    Simple fully connected regressor for thickness prediction.
    Input: reflectivity curve (q_points)
    Output: thickness values (n_layer)
    """

    def __init__(self, input_dim: int, n_layer: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_layer)  # regression output
        )

    def forward(self, x):
        return self.net(x)
