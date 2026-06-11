import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.util import Surv
from typing import List, Tuple, Union


def load_and_prepare_data(
    url="https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/mgus2.csv",
) -> pd.DataFrame:
    df = pd.read_csv(url)

    # Кодування подій та визначення часу
    conditions = [
        (df["pstat"] == 1),  # Подія 1: Прогресія (PCM)
        (df["death"] == 1) & (df["pstat"] == 0),  # Подія 2: Смерть без прогресії
        (df["pstat"] == 0) & (df["death"] == 0),  # Подія 0: Цензурування
    ]
    choices_event = [1, 2, 0]
    choices_time = [df["ptime"], df["futime"], df["futime"]]

    df["event_type"] = np.select(conditions, choices_event, default=np.nan)
    df["time"] = np.select(conditions, choices_time, default=np.nan)

    return df.dropna(subset=["event_type", "time"])


def split_and_scale_data(
    df: pd.DataFrame, features: List[str]
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    # Тестова вибірка (20%)
    df_train_val, df_test = train_test_split(
        df, test_size=0.20, stratify=df["event_type"], random_state=42
    )
    # Валідаційна вибірка (10% від загального об'єму)
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=0.125,
        stratify=df_train_val["event_type"],
        random_state=42,
    )

    # Нормалізація ознак
    for data_frame in [df_train, df_val, df_test]:
        if data_frame["sex"].dtype == "O":
            data_frame["sex"] = data_frame["sex"].map({"F": 0, "M": 1})
        data_frame.dropna(subset=features, inplace=True)

    X_train = df_train[features]
    X_val = df_val[features]
    X_test = df_test[features]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return df_train, df_val, df_test, X_train_scaled, X_val_scaled, X_test_scaled


def create_time_bins(
    time_array: Union[np.ndarray, pd.Series], num_bins: int = 20
) -> np.ndarray:
    times = time_array[time_array > 0]
    bins = np.quantile(times, np.linspace(0, 1, num_bins + 1))
    bins[0] = 0.0
    bins[-1] = np.inf
    return bins


def digitize_time(
    time_array: Union[np.ndarray, pd.Series], bins: np.ndarray
) -> np.ndarray:
    bin_indices = np.digitize(time_array, bins) - 1
    return np.clip(bin_indices, 0, len(bins) - 2)


def prepare_tensor_loaders(
    X_train_s: pd.DataFrame,
    df_train: pd.DataFrame,
    X_val_s: pd.DataFrame,
    df_val: pd.DataFrame,
    X_test_s: pd.DataFrame,
    df_test: pd.DataFrame,
    bins: np.ndarray,
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
    def to_tensors(
        X_df: pd.DataFrame, df: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_tensor = torch.tensor(X_df.values, dtype=torch.float32)
        events_tensor = torch.tensor(df["event_type"].values, dtype=torch.long)
        time_idx = digitize_time(df["time"].values, bins)
        times_tensor = torch.tensor(time_idx, dtype=torch.long)
        return X_tensor, events_tensor, times_tensor

    X_tr, y_tr_e, y_tr_t = to_tensors(X_train_s, df_train)
    X_va, y_va_e, y_va_t = to_tensors(X_val_s, df_val)
    X_te, y_te_e, y_te_t = to_tensors(X_test_s, df_test)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr_e, y_tr_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_va, y_va_e, y_va_t), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, X_tr, X_va, X_te


def prepare_survival_data(df: pd.DataFrame, target_event: int) -> np.ndarray:
    event_happened = (df["event_type"] == target_event).astype(bool)
    return Surv.from_arrays(event=event_happened, time=df["time"])
