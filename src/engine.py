import copy
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models import FinalTransformerModel


class CompetingRisksLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(
        self,
        hazards: torch.Tensor,
        S_t: torch.Tensor,
        events: torch.Tensor,
        time_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hazards.size(0)
        loss: torch.Tensor = torch.tensor(0.0, device=hazards.device)
        S_t_minus_1 = torch.cat(
            [torch.ones(batch_size, 1, device=hazards.device), S_t[:, :-1]], dim=1
        )

        for i in range(batch_size):
            t_i = time_indices[i]
            k_i = events[i]

            if k_i > 0:
                h_k = hazards[i, k_i, t_i]
                s_prev = S_t_minus_1[i, t_i]
                # ЗМІНА: torch.clamp для жорсткого обмеження знизу
                loss -= torch.log(torch.clamp(h_k, min=self.eps)) + torch.log(
                    torch.clamp(
                        s_prev, min=self.eps
                    )  # гарант, що логарифм ніколи не отримає 0 або від'ємне число.
                )
            else:
                s_curr = S_t[i, t_i]
                loss -= torch.log(torch.clamp(s_curr, min=self.eps))  # Аналогічно

        return loss / batch_size


def train_model_with_history(
    variant_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_features: int,
    num_bins: int,
    max_epochs: int = 200,
    patience: int = 15,
) -> Tuple[nn.Module, List[float], List[float]]:
    print(f"\n[Trainer] Запуск навчання для Variant {variant_name}...")
    model = FinalTransformerModel(
        n_features=n_features,
        variant=variant_name,
        d_model=32,
        n_heads=2,
        num_layers=2,
        T=num_bins,
        K=2,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = CompetingRisksLoss()

    best_val_loss = float("inf")
    best_weights: Optional[Dict[str, Any]] = None
    epochs_no_improve = 0

    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(max_epochs):
        model.train()
        train_loss: float = 0.0
        for batch_X, batch_events, batch_times in train_loader:
            optimizer.zero_grad()
            hazards, S_t, _ = model(batch_X)
            loss = criterion(hazards, S_t, batch_events, batch_times)
            # ЗМІНА: Запобіжник від NaN колапсу
            if torch.isnan(loss):
                print(
                    f"\n[!] Увага: Loss став NaN на епосі {epoch+1}. Переривання батчу."
                )
                break
            loss.backward()

            # ЗМІНА: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_history.append(train_loss)

        model.eval()
        val_loss: float = 0.0
        valid_batches: int = 0  # лічильник успішних батчів
        with torch.no_grad():
            for batch_X, batch_events, batch_times in val_loader:
                hazards, S_t, _ = model(batch_X)
                loss = criterion(hazards, S_t, batch_events, batch_times)

                if not torch.isnan(loss):
                    val_loss += loss.item()
                    valid_batches += 1
                else:
                    print(f"\n[!] Увага: Val Loss став NaN. Батч пропущено.")

                # Уникаємо ділення на нуль, якщо раптом всі батчі були NaN
        if valid_batches > 0:
            val_loss /= valid_batches
        else:
            val_loss = float("inf")

        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Епоха [{epoch+1:3d}/{max_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

        if epochs_no_improve >= patience:
            print(
                f"  [-] Рання зупинка на епосі {epoch+1}. Ваги найкращої епохи відновлено."
            )
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model, train_history, val_history
