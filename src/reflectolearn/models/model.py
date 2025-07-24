import torch
import torch.nn as nn
import torch.nn.functional as F

from reflectolearn.types import ModelType


class XRRRegressor(nn.Module):
    def __init__(self, output_length=3):
        super(XRRRegressor, self).__init__()

        # 1D CNN feature extractor
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.bn1 = nn.BatchNorm1d(16)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)

        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling

        # Fully connected regression head
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, output_length)  # Output: sld, thickness, roughness...

    def forward(self, x):
        # x: (batch_size, input_length) → reshape to (batch_size, 1, input_length)
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # shape: (batch_size, 32)
        x = F.relu(self.fc1(x))
        output = self.fc2(x)  # shape: (batch_size, 3)
        return output


class XRRHybridRegressor(nn.Module):
    def __init__(self, input_length, output_length=3):
        super(XRRHybridRegressor, self).__init__()

        # Multi-kernel 1D CNN for multi-scale fringe feature capture
        self.conv_small = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(1, 16, kernel_size=9, padding=4)
        self.conv_large = nn.Conv1d(1, 16, kernel_size=21, padding=10)

        self.bn = nn.BatchNorm1d(48)

        # Global average pooling to summarize fringe structure
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Feedforward MLP for global Rs(q) trend
        self.fc_mlp = nn.Sequential(
            nn.Linear(input_length, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
        )

        # Final combined regression head
        self.final_fc = nn.Sequential(
            nn.Linear(48 + 128, 64),
            nn.ReLU(),
            nn.Linear(64, output_length),  # sld, thickness, roughness
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # shape: (B, 1, L)

        # Multi-scale fringe feature
        x1 = F.relu(self.conv_small(x))
        x2 = F.relu(self.conv_mid(x))
        x3 = F.relu(self.conv_large(x))
        x_cat = torch.cat([x1, x2, x3], dim=1)  # shape: (B, 48, L)
        x_cat = self.bn(x_cat)
        x_feat = self.pool(x_cat).squeeze(-1)  # shape: (B, 48)

        # Global MLP features from raw input
        x_mlp = x.squeeze(1)  # shape: (B, L)
        x_mlp = self.fc_mlp(x_mlp)  # shape: (B, 128)

        # Final prediction
        x_final = torch.cat([x_feat, x_mlp], dim=1)  # shape: (B, 176)
        out = self.final_fc(x_final)
        return out


def get_model(
    model_type: ModelType, input_length: int, output_length: int
) -> nn.Module:
    if not isinstance(model_type, ModelType):
        raise TypeError(
            f"model_typle should be an instance of {ModelType.__name__} not {type(model_type)}"
        )
    match model_type:
        case ModelType.HYBRID:
            if input_length is None:
                raise ValueError("input_length is required for hybrid model")
            return XRRHybridRegressor(input_length, output_length)
        case ModelType.REGRESSOR:
            return XRRRegressor(output_length)
        case _:
            raise ValueError(f"Unknown model type: {model_type}")
