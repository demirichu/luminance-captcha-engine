import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import NUM_CLASSES, CAPTCHA_LENGTH, IMG_CHANNELS


def build_positional_encoding(length, dim):
    """Build sinusoidal positional encoding table."""
    pe = torch.zeros(length, dim, dtype=torch.float32)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * (-np.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class CaptchaAttentionCRNN(nn.Module):
    """Teacher Architecture: High-capacity Convolutional Recurrent Neural Network

    Combines 4-stage feature extraction, 2-directional GRU recurrent modeling,
    and a multi-head character attention mechanism for robust text alignment.
    Pulls default dimensions directly from config.py.
    """
    def __init__(self, num_classes=NUM_CLASSES, num_characters=CAPTCHA_LENGTH,
                 in_channels=IMG_CHANNELS, pos_encoding_max_length=100):
        super(CaptchaAttentionCRNN, self).__init__()
        self.num_classes = num_classes
        self.num_characters = num_characters
        self.in_channels = in_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1))
        )

        self.register_buffer('pos_embedding', build_positional_encoding(pos_encoding_max_length, 256))

        self.norm_cnn = nn.LayerNorm(256)
        self.rnn = nn.GRU(256, 128, bidirectional=True, batch_first=True)
        self.norm_rnn = nn.LayerNorm(256)

        self.dropout = nn.Dropout(0.3)
        self.attention = nn.Linear(256, num_characters)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, temperature=1.0, return_attention=False):
        x = self.cnn(x)
        # Collapse remaining height dimension to 1 (handles inputs with height >= 16px)
        if x.size(2) > 1:
            x = x.mean(dim=2, keepdim=True)
        x = x.squeeze(2).permute(0, 2, 1)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.norm_cnn(x)
        x, _ = self.rnn(x)
        x = self.norm_rnn(x)
        x = self.dropout(x)

        attn_scores = self.attention(x)
        attn_weights = torch.softmax(attn_scores / temperature, dim=1)
        char_feats = torch.bmm(attn_weights.transpose(1, 2), x)
        out = self.fc(char_feats)

        if return_attention:
            return out, attn_weights
        return out, attn_weights


class CaptchaStudentCNN(nn.Module):
    """Student Architecture: Ultra-lightweight Pure-CNN

    Optimized for single-sample microsecond CPU/INT8 deployment.
    Eliminates recurrent GRU loops in favor of 1D sequence mixing and
    learnable spatial alignment projection.
    Pulls default dimensions directly from config.py.
    """
    def __init__(self, num_classes=NUM_CLASSES, num_characters=CAPTCHA_LENGTH,
                 in_channels=IMG_CHANNELS, pool_output_width=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_characters = num_characters
        self.in_channels = in_channels

        # Dynamically determine pooling width to maintain adequate spatial resolution
        if pool_output_width is None:
            pool_output_width = max(8, num_characters * 2)
        self.pool_output_width = pool_output_width

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, pool_output_width))
        )

        self.sequence_mixer = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.alignment = nn.Linear(pool_output_width, num_characters)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(2)
        x = F.relu(self.sequence_mixer(x))
        x = self.alignment(x)
        x = x.permute(0, 2, 1)
        out = self.classifier(x)
        return out
