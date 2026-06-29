import torch
import torch.nn as nn


class RNNSentiment(nn.Module):
    """LSTM or GRU, unidirectional or bidirectional, for sentiment classification.

    Architecture:
        Embedding (300-dim fastText)
          → Dropout
          → 2-layer RNN (dropout between layers)
          → Dropout
          → Linear(rnn_out → hidden) + LayerNorm + ReLU + Dropout   [hidden]
          → Linear(hidden → num_classes)                             [output]
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout, rnn_type='lstm', bidirectional=True,
                 num_classes=3, embedding_matrix=None):
        super().__init__()
        assert rnn_type in ('lstm', 'gru'), "rnn_type must be 'lstm' or 'gru'"
        self.rnn_type      = rnn_type
        self.bidirectional = bidirectional

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=False,
                padding_idx=0,
            )
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.1)
            with torch.no_grad():
                self.embedding.weight[0].zero_()   # PAD stays zero

        rnn_cls  = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # inter-layer dropout
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.dropout  = nn.Dropout(dropout)
        directions    = 2 if bidirectional else 1
        rnn_out_dim   = hidden_dim * directions

        # 1 hidden layer + 1 output layer, dropout at every transition
        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        emb    = self.dropout(self.embedding(x))   # (B, L, E)
        out, h = self.rnn(emb)

        if self.rnn_type == 'lstm':
            h = h[0]                               # (h_n, c_n) → keep h_n

        # h: (num_layers * directions, B, H)
        if self.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=1)   # last layer fwd+bwd → (B, H*2)
        else:
            h = h[-1]                              # last layer → (B, H)

        return self.head(self.dropout(h))          # (B, num_classes)
