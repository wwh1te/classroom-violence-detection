
"""MLP and LSTM classifiers for classroom activity recognition."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class MLPClassifier(nn.Module):
    """Classify a video using aggregated temporal features."""

    def __init__(self, input_size=492, num_classes=2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class LSTMClassifier(nn.Module):
    """Classify a video while preserving its temporal sequence."""

    def __init__(
        self,
        input_size=246,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed_input)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        final_hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        return self.classifier(final_hidden)
