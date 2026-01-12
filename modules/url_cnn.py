# modules/url_cnn.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# Character vocabulary for URLs
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
CHAR2IDX = {c: i+1 for i, c in enumerate(CHARS)}  # 0 = padding

MAX_LEN = 200  # max URL length

def encode_url(url):
    url = url.lower()[:MAX_LEN]
    x = [CHAR2IDX.get(c, 0) for c in url]
    x += [0] * (MAX_LEN - len(x))  # padding
    return x

class URLDataset(Dataset):
    def __init__(self, urls, labels):
        self.X = torch.tensor([encode_url(u) for u in urls], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class URLCNN(nn.Module):
    def __init__(self, vocab_size=len(CHAR2IDX)+1, embed_dim=32, num_filters=64, kernel_size=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(num_filters * ((MAX_LEN - kernel_size + 1)//2), 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # [batch, embed, seq_len]
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x
