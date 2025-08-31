
import math
import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    def __init__(self, in_feats: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_feats, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fc = nn.Linear(64, n_classes)
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, F, T)
        x = self.net(x).squeeze(-1)
        return self.fc(x)

class BiLSTM(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_feats, hidden_size=hidden,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden, n_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class CNN_LSTM(nn.Module):
    def __init__(self, in_feats: int, hidden: int = 64, n_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_feats, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*hidden, n_classes)
    def forward(self, x):
        x = x.transpose(1,2)
        x = self.conv(x)
        x = x.transpose(1,2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, in_feats: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, n_classes: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_feats, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model, n_classes)
    def forward(self, x):
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.cls(x)

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, modality_dims, d_model: int = 64, n_classes: int = 2):
        super().__init__()
        self.mods = list(modality_dims.keys())
        self.encoders = nn.ModuleDict({ m: nn.Sequential(nn.Linear(modality_dims[m], d_model), nn.ReLU()) for m in self.mods })
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        self.cls = nn.Linear(d_model, n_classes)
    def forward(self, x_dict):
        tokens = []
        for m in self.mods:
            h = self.encoders[m](x_dict[m])  # (B, T, d)
            h = h.mean(dim=1, keepdim=True)  # (B, 1, d)
            tokens.append(h)
        X = torch.cat(tokens, dim=1)  # (B, M, d)
        H, _ = self.attn(X, X, X)
        H = H.mean(dim=1)            # (B, d)
        return self.cls(H)
