from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotonicAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()

        q = (
            self.q_linear(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        k = (
            self.k_linear(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )
        v = (
            self.v_linear(x)
            .view(batch_size, seq_len, self.n_heads, self.d_k)
            .transpose(1, 2)
        )

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)
        r = torch.exp(scores)
        c = torch.cumsum(r, dim=-1)
        a = r / (c[..., -1:] + 1e-9)

        out = torch.matmul(a, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)


class MonotonicTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attention = MonotonicAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ffn(x))
        return x


class FinalTransformerModel(nn.Module):
    def __init__(
        self,
        n_features: int,
        variant: str = "A",
        d_model: int = 32,
        n_heads: int = 2,
        num_layers: int = 2,
        T: int = 20,
        K: int = 2,
    ) -> None:
        super().__init__()
        self.T = T
        self.K = K
        self.variant = variant

        self.feature_proj = nn.Linear(n_features, d_model)
        self.time_embeddings = nn.Parameter(torch.randn(T, d_model))

        self.encoder: nn.Module

        if variant == "A":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif variant == "B":
            layers = [
                MonotonicTransformerLayer(d_model, n_heads) for _ in range(num_layers)
            ]
            self.encoder = nn.Sequential(*layers)

        self.output_proj = nn.Linear(d_model, K + 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        e_i = self.feature_proj(x)
        seq_input = e_i.unsqueeze(1) + self.time_embeddings.unsqueeze(0)

        encoded_seq = self.encoder(seq_input)
        raw_logits = self.output_proj(encoded_seq).permute(0, 2, 1)

        hazards = F.softmax(raw_logits, dim=1)
        hazards_survival = hazards[:, 0, :]
        hazards_causes = hazards[:, 1:, :]

        S_t = torch.cumprod(hazards_survival, dim=1)
        S_t_minus_1 = torch.cat(
            [torch.ones(batch_size, 1, device=x.device), S_t[:, :-1]], dim=1
        )

        cif_increments = hazards_causes * S_t_minus_1.unsqueeze(1)
        cif = torch.cumsum(cif_increments, dim=2)

        return hazards, S_t, cif
