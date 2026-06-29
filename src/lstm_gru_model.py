import torch
import torch.nn as nn


class RNNSentiment(nn.Module):
    """
    Architecture:
        Embedding (300-dim fastText)
          → BatchNorm1d(embed_dim) + Dropout(p)         [input regularisation]
          → 1-layer RNN cell with variational recurrent dropout(rec_p)
          → Dropout(p)                                   [output regularisation]
          → Linear(rnn_out → hidden) + LayerNorm + ReLU + Dropout(p)  [hidden]
          → Linear(hidden → num_classes)                               [output]

    Variational recurrent dropout: one binary mask sampled per forward pass,
    applied identically at every time step — same mask, whole sequence.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 dropout, rec_dropout=0.25,
                 rnn_type='lstm', bidirectional=True,
                 num_classes=3, embedding_matrix=None):
        super().__init__()
        assert rnn_type in ('lstm', 'gru'), "rnn_type must be 'lstm' or 'gru'"
        self.rnn_type      = rnn_type
        self.bidirectional = bidirectional
        self.hidden_dim    = hidden_dim
        self.rec_dropout_p = rec_dropout

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
                self.embedding.weight[0].zero_()

        # BatchNorm on input: normalises each embedding dim across batch×time
        self.input_bn = nn.BatchNorm1d(embed_dim)

        # Single-layer cells (LSTMCell/GRUCell) give step-level dropout control
        cell_cls      = nn.LSTMCell if rnn_type == 'lstm' else nn.GRUCell
        self.cell_fwd = cell_cls(embed_dim, hidden_dim)
        if bidirectional:
            self.cell_bwd = cell_cls(embed_dim, hidden_dim)

        self.embed_drop = nn.Dropout(dropout)
        self.out_drop   = nn.Dropout(dropout)

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(rnn_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _run_cell(self, cell, emb):
        """Run one RNN cell over a sequence with variational recurrent dropout."""
        B, L, _ = emb.shape
        h = emb.new_zeros(B, self.hidden_dim)
        c = emb.new_zeros(B, self.hidden_dim) if self.rnn_type == 'lstm' else None

        # Sample mask once — same mask applied at every time step (variational)
        if self.training and self.rec_dropout_p > 0:
            scale = 1.0 / (1.0 - self.rec_dropout_p)
            mask  = torch.empty_like(h).bernoulli_(1.0 - self.rec_dropout_p).mul_(scale)
        else:
            mask = None

        for t in range(L):
            if self.rnn_type == 'lstm':
                h, c = cell(emb[:, t], (h, c))
            else:
                h = cell(emb[:, t], h)
            if mask is not None:
                h = h * mask

        return h

    def forward(self, x):
        emb = self.embedding(x)           # (B, L, E)
        B, L, E = emb.shape

        # BatchNorm1d expects (N, C) — flatten time into batch, normalise, restore
        emb = self.input_bn(emb.reshape(B * L, E)).reshape(B, L, E)
        emb = self.embed_drop(emb)

        h_fwd = self._run_cell(self.cell_fwd, emb)

        if self.bidirectional:
            h_bwd = self._run_cell(self.cell_bwd, emb.flip(1))
            h = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            h = h_fwd

        return self.head(self.out_drop(h))
