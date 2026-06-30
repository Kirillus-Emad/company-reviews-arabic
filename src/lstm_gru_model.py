import torch
import torch.nn as nn


class RNNSentiment(nn.Module):
    """
    Architecture:
        Embedding (300-dim fastText) — raw, no dropout before RNN
          → N-layer stacked RNN cells with variational recurrent dropout per layer
          → Dropout → Linear(hidden → num_classes)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout, rec_dropout=0.3,
                 rnn_type='lstm', bidirectional=False,
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

        # Stacked cells: first layer input=embed_dim, subsequent layers input=hidden_dim
        cell_cls = nn.LSTMCell if rnn_type == 'lstm' else nn.GRUCell
        self.cells_fwd = nn.ModuleList([
            cell_cls(embed_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        if bidirectional:
            self.cells_bwd = nn.ModuleList([
                cell_cls(embed_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)
            ])

        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim, num_classes),
        )

    def _run_cells(self, cells, emb):
        """Stacked RNN cells with variational recurrent dropout per layer."""
        B, L, _ = emb.shape

        h_list = [emb.new_zeros(B, self.hidden_dim) for _ in cells]
        c_list = [emb.new_zeros(B, self.hidden_dim) for _ in cells] \
                 if self.rnn_type == 'lstm' else None

        # One dropout mask per layer, same mask applied at every time step
        if self.training and self.rec_dropout_p > 0:
            scale = 1.0 / (1.0 - self.rec_dropout_p)
            masks = [torch.empty_like(h).bernoulli_(1.0 - self.rec_dropout_p).mul_(scale)
                     for h in h_list]
        else:
            masks = [None] * len(cells)

        for t in range(L):
            inp = emb[:, t]
            for i, cell in enumerate(cells):
                if self.rnn_type == 'lstm':
                    h_list[i], c_list[i] = cell(inp, (h_list[i], c_list[i]))
                else:
                    h_list[i] = cell(inp, h_list[i])
                if masks[i] is not None:
                    h_list[i] = h_list[i] * masks[i]
                inp = h_list[i]   # output of layer i feeds into layer i+1

        return h_list[-1]         # final layer hidden state

    def forward(self, x):
        emb   = self.embedding(x)                    # (B, L, E) — raw, no dropout
        h_fwd = self._run_cells(self.cells_fwd, emb)

        if self.bidirectional:
            h_bwd = self._run_cells(self.cells_bwd, emb.flip(1))
            h = torch.cat([h_fwd, h_bwd], dim=1)
        else:
            h = h_fwd

        return self.head(h)
