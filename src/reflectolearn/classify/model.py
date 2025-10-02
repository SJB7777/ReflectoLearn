import torch.nn as nn
import torch.nn.functional as F


class NlayerClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_feat = nn.Linear(32, 128)
        self.classifier = nn.Linear(128, num_classes)
        # Regressor 헤드는 완전히 제거

    def forward(self, R):
        x = R.unsqueeze(1)
        feat = self.encoder(x).squeeze(-1)
        feat = F.relu(self.fc_feat(feat))
        n_layer_logits = self.classifier(feat)
        return n_layer_logits
