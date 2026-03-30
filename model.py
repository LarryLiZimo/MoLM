import torch
import torch.nn as nn

from config import ModelConfig


class LLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.seq_len = config.seq_len

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)  # (vocab_size, d_model)
        self.pos_emb   = nn.Embedding(config.seq_len,    config.d_model)  # (seq_len,    d_model)
        self.emb_drop  = nn.Dropout(config.dropout)

        assert config.d_model % config.nhead == 0
        if config.d_model // config.nhead != 64:
            print(f"WARNING: config.d_model // config.nhead != 64")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,   # (B, T, d_model) throughout
            norm_first=True,    # pre-norm: more stable for deep models
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,  # required when norm_first=True
        )

        self.norm    = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

        # precomputed; moved to device automatically with .to(device)
        self.register_buffer("pos",         torch.arange(config.seq_len))                              # (seq_len,)
        self.register_buffer("causal_mask", nn.Transformer.generate_square_subsequent_mask(config.seq_len))  # (seq_len, seq_len)

    def forward(self, x: torch.Tensor, targets: torch.Tensor | None = None) -> torch.Tensor:
        # x:       (B, T)
        # targets: (B, T) — when given, returns scalar loss; else returns logits (B, T, vocab_size)
        B, T = x.shape
        emb    = self.emb_drop(self.token_emb(x) + self.pos_emb(self.pos[:T]))            # (B, T, d_model)
        out    = self.transformer(emb, mask=self.causal_mask[:T, :T], is_causal=True)     # (B, T, d_model)
        logits = self.lm_head(self.norm(out))                                              # (B, T, vocab_size)

        if targets is None:
            return logits

        B, T, V = logits.shape
        return nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))    # scalar
