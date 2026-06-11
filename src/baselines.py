import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from comprisk import FineGrayRegression
import pandas as pd
from src.data_prep import prepare_survival_data
from typing import List, Dict, Any

def train_cox_models(
    df_train: pd.DataFrame, 
    X_train_scaled: pd.DataFrame, 
    causes: List[int] = [1, 2]
) -> Dict[int, Any]:
    print("\n[Baselines] Навчання Cause-specific Cox...")
    models_cox = {}
    for cause in causes:
        y_train_cause = prepare_survival_data(df_train, cause)
        cox_model = CoxPHSurvivalAnalysis()
        cox_model.fit(X_train_scaled, y_train_cause)
        models_cox[cause] = cox_model
        print(f" Cox для Події {cause} навчено.")
    return models_cox


def train_fine_gray_models(
    df_train: pd.DataFrame, 
    X_train_scaled: pd.DataFrame, 
    causes: List[int] = [1, 2]
) -> Dict[int, Any]:
    print("\n[Baselines] Навчання Fine-Gray (comprisk)...")
    models_fg = {}

    time_train = df_train["time"].values
    event_train = df_train["event_type"].values
    X_train_vals = X_train_scaled.values

    for cause in causes:
        try:
            fg_model = FineGrayRegression(cause=cause)
            fg_model.fit(X_train_vals, time=time_train, event=event_train)
            models_fg[cause] = fg_model
            print(f"Fine-Gray для Події {cause} навчено.")
        except Exception as e:
            print(f" Помилка Fine-Gray для Події {cause}: {e}")

    return models_fg