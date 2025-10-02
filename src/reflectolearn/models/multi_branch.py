# models/multi_branch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RawCurveBranch(nn.Module):
    """Branch for raw reflectivity curve input (R vs q)"""
    def __init__(self, input_length, hidden_dim=128):
        super().__init__()
        # Multi-scale CNN
        self.conv_small = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv_mid = nn.Conv1d(1, 16, kernel_size=9, padding=4)
        self.conv_large = nn.Conv1d(1, 16, kernel_size=21, padding=10)

        self.bn = nn.BatchNorm1d(48)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Global MLP for residual learning
        self.fc_mlp = nn.Sequential(
            nn.Linear(input_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(1)  # (B, 1, L)
        # CNN multi-scale feature
        x1 = F.relu(self.conv_small(x))
        x2 = F.relu(self.conv_mid(x))
        x3 = F.relu(self.conv_large(x))
        x_cat = torch.cat([x1, x2, x3], dim=1)  # (B, 48, L)
        x_cat = self.bn(x_cat)
        x_feat = self.pool(x_cat).squeeze(-1)  # (B, 48)

        # Global MLP
        x_mlp = x.squeeze(1)  # (B, L)
        x_mlp = self.fc_mlp(x_mlp)  # (B, hidden_dim//2)

        # Combine raw branch features
        return torch.cat([x_feat, x_mlp], dim=1)  # (B, 48 + hidden_dim//2)


class PhysicsFeatureBranch(nn.Module):
    """Branch for physics-informed features (FFT peak, q_c, etc.)"""
    def __init__(self, num_features, hidden_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x_feat):
        # x_feat: (B, num_features)
        return self.fc(x_feat)  # (B, hidden_dim)


class XRRMultiBranchRegressor(nn.Module):
    """Final multi-branch regressor combining raw curve + physics features"""
    def __init__(self, input_length, num_phys_features, output_length=3):
        super().__init__()
        self.raw_branch = RawCurveBranch(input_length)
        self.phys_branch = PhysicsFeatureBranch(num_phys_features)

        # Fusion + regression head
        fusion_dim = (48 + 64) + 32  # (CNN+MLP from raw) + physics branch
        self.final_fc = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_length),
        )

    def forward(self, x_curve, x_phys):
        """
        x_curve: (B, L) raw reflectivity curve
        x_phys: (B, num_phys_features) physics features
        """
        raw_feat = self.raw_branch(x_curve)
        phys_feat = self.phys_branch(x_phys)
        fused = torch.cat([raw_feat, phys_feat], dim=1)
        return self.final_fc(fused)
