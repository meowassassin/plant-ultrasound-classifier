# src/models/baseline_cnn.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KhaitCNN1D(nn.Module):
    """
    1D CNN ported from CNN_sounds_classifier.py in Cell paper appendix to PyTorch.

    Keras architecture:

        batch_size = 64
        num_epochs = 50
        kernel_size = 9
        pool_size = 4
        conv_depth_1 = 32
        conv_depth_2 = 64
        conv_depth_3 = 128
        hidden_size = 128
        drop_prob_1 = 0.5
        drop_prob_2 = 0.5

        # Conv blocks (1D)
        [Conv1D(32, 9, padding='same', relu) x2] + MaxPool1D(4) + Dropout(0.5)
        [Conv1D(64, 9, padding='same', relu) x2] + MaxPool1D(4) + Dropout(0.5)
        [Conv1D(128, 9, padding='same', relu) x2] + MaxPool1D(4) + Dropout(0.5)

        Flatten -> Dense(128, relu) -> Dropout(0.5) -> Dense(1, sigmoid)

    Here we use Dense(1) + BCEWithLogitsLoss
    instead of Dense(1, sigmoid).

    Input:
        x: [B, 1, L]  (L ≈ 1000 ~ 1001)
    Output:
        logits: [B]  (logits before sigmoid)
    """

    def __init__(self, input_length: int = 1000):
        super().__init__()

        kernel_size = 9
        padding = kernel_size // 2        # Corresponds to 'same' (odd kernel)
        pool_size = 4
        drop_prob = 0.5
        hidden_size = 128

        # --- Conv block 1 ---
        self.conv1_1 = nn.Conv1d(1, 32, kernel_size, padding=padding)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size, padding=padding)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.drop1 = nn.Dropout(drop_prob)

        # --- Conv block 2 ---
        self.conv2_1 = nn.Conv1d(32, 64, kernel_size, padding=padding)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size, padding=padding)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.drop2 = nn.Dropout(drop_prob)

        # --- Conv block 3 ---
        self.conv3_1 = nn.Conv1d(64, 128, kernel_size, padding=padding)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size, padding=padding)
        self.pool3 = nn.MaxPool1d(pool_size)
        self.drop3 = nn.Dropout(drop_prob)

        # --- Flatten + Dense(128) + Dropout + Dense(1) ---
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            feat = self._forward_conv(dummy)
            flat_dim = feat.shape[1] * feat.shape[2]

        self.fc = nn.Linear(flat_dim, hidden_size)
        self.drop_fc = nn.Dropout(drop_prob)
        self.fc_out = nn.Linear(hidden_size, 1)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = self.drop3(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, L]
        return: logits [B]  (for BCEWithLogitsLoss)
        """
        x = self._forward_conv(x)
        x = x.flatten(1)              # [B, C, T] -> [B, C*T]
        x = F.relu(self.fc(x))
        x = self.drop_fc(x)
        logits = self.fc_out(x).squeeze(-1)  # [B, 1] -> [B]
        return logits


# For compatibility with old name
BaselineCNN = KhaitCNN1D


def create_baseline_cnn_with_optim(
    input_length: int,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
):
    """
    Helper function to port Keras configuration to PyTorch:
      - Model: KhaitCNN1D
      - Loss: BCEWithLogitsLoss(pos_weight=pos_weight)
      - Optimizer: Adam(lr=1e-3)
    """
    model = KhaitCNN1D(input_length=input_length).to(device)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, criterion, optimizer
