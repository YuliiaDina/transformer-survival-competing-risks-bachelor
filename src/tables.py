import pandas as pd
import numpy as np
import torch
import inspect
import torch.optim as optim
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score, cumulative_dynamic_auc

# Імпорти з твоїх власних модулів
from src.data_prep import prepare_survival_data, digitize_time
from src.metrics import compute_cox_cif
from src.models import FinalTransformerModel

def print_table_1_demographics(df_original):
    """
    Таблиця 1: Описова статистика вибірки (Demographics).
    Виводить відформатований текст у консоль.
    """
    print("\033[4m Таблиця описової статистики (Table 1)\033[0m")

    print("\n[Характеристики пацієнтів (Mean ± SD)]")
    for col in ['age', 'dxyr', 'hgb', 'creat', 'mspike']:
        mean_val = df_original[col].mean()
        sd_val = df_original[col].std()
        print(f"- {col.upper()}: {mean_val:.2f} ± {sd_val:.2f}")

    sex_counts = df_original['sex'].value_counts()
    sex_total = len(df_original)
    print("\n[Стать (Кількість / Відсоток)]")
    for sex_val, count in sex_counts.items():
        print(f"- Стать '{sex_val}': {count} ({count/sex_total*100:.1f}%)")

    print("\n[Розподіл подій]")
    event_counts = df_original['event_type'].value_counts().sort_index()
    event_names = {0: "0 (Цензуровано)", 1: "1 (Прогресія)", 2: "2 (Смерть)"}
    for ev_code, count in event_counts.items():
        print(f"- {event_names[ev_code]}: {count} ({count/sex_total*100:.1f}%)")

    median_time = df_original['time'].median()
    q1 = df_original['time'].quantile(0.25)
    q3 = df_original['time'].quantile(0.75)
    print(f"\n[Час спостереження (Місяці)]")
    print(f"- Медіана (IQR): {median_time:.1f} ({q1:.1f} - {q3:.1f})")


def build_table_2_metrics(models_cox, models_fg_python, model_A_hist, model_B_hist, df_train, df_test, X_test_scaled, X_test_tensor, bins, brier_times, times_to_evaluate=[24, 60, 120]):
    """
    Таблиця 2: Зведена таблиця метрик (C-index, IBS, AUC).
    Повертає відформатований Pandas DataFrame.
    """
    unified_results = []

    model_A_hist.eval()
    model_B_hist.eval()

    with torch.no_grad():
        _, _, cif_A_all = model_A_hist(X_test_tensor)
        _, _, cif_B_all = model_B_hist(X_test_tensor)

    # Загальне виживання для Cox
    surv_fns_1 = models_cox[1].predict_survival_function(X_test_scaled)
    surv_fns_2 = models_cox[2].predict_survival_function(X_test_scaled)
    surv_fns_overall = [(lambda t, i=i: surv_fns_1[i](t) * surv_fns_2[i](t)) for i in range(len(surv_fns_1))]

    for cause in [1, 2]:
        y_train_cause = prepare_survival_data(df_train, cause)
        y_test_cause = prepare_survival_data(df_test, cause)

        # 1. Cox
        model_cox = models_cox[cause]
        cox_risk_static = model_cox.predict(X_test_scaled)
        try: c_index_cox = concordance_index_ipcw(y_train_cause, y_test_cause, cox_risk_static)[0]
        except: c_index_cox = np.nan
        
        surv_fns_k = model_cox.predict_survival_function(X_test_scaled)
        cif_at_horizons = compute_cox_cif(surv_fns_k, surv_fns_overall, times_to_evaluate)
        
        try: auc_cox, _ = cumulative_dynamic_auc(y_train_cause, y_test_cause, cif_at_horizons, times_to_evaluate)
        except: auc_cox = [np.nan, np.nan, np.nan]
            
        cif_matrix_brier = compute_cox_cif(surv_fns_k, surv_fns_overall, brier_times)
        try: ibs_cox = integrated_brier_score(y_train_cause, y_test_cause, 1.0 - cif_matrix_brier, brier_times)
        except: ibs_cox = np.nan
            
        unified_results.append({
            "Model": "Cox", "Event": cause, "C-index": c_index_cox, "IBS": ibs_cox,
            "AUC 2yr": auc_cox[0], "AUC 5yr": auc_cox[1], "AUC 10yr": auc_cox[2]
        })

        # 2. Fine-Gray
        if cause in models_fg_python:
            fg_model = models_fg_python[cause]
            try:
                cif_final = fg_model.predict_cumulative_incidence(X_test_scaled.values, times=np.array([df_train['time'].max()]))[:, -1]
                c_index_fg = concordance_index_ipcw(y_train_cause, y_test_cause, cif_final)[0]
            except: c_index_fg = np.nan
                
            auc_fg = []
            for t in times_to_evaluate:
                try:
                    risk_at_t_fg = fg_model.predict_cumulative_incidence(X_test_scaled.values, times=np.array([t]))[:, 0]
                    auc_t_fg, _ = cumulative_dynamic_auc(y_train_cause, y_test_cause, risk_at_t_fg, np.array([t]))
                    auc_fg.append(auc_t_fg[0])
                except: auc_fg.append(np.nan)
                    
            try:
                cif_brier = fg_model.predict_cumulative_incidence(X_test_scaled.values, times=brier_times)
                ibs_fg = integrated_brier_score(y_train_cause, y_test_cause, 1.0 - cif_brier, brier_times)
            except: ibs_fg = np.nan
                
            unified_results.append({
                "Model": "Fine-Gray", "Event": cause, "C-index": c_index_fg, "IBS": ibs_fg,
                "AUC 2yr": auc_fg[0], "AUC 5yr": auc_fg[1], "AUC 10yr": auc_fg[2]
            })

        # 3. Трансформери
        auc_time_indices = digitize_time(times_to_evaluate, bins)
        brier_time_indices = digitize_time(brier_times, bins)
        
        for model_name, cif_pred in [("Transformer A", cif_A_all.numpy()), ("Transformer B", cif_B_all.numpy())]:
            cif_cause = cif_pred[:, cause - 1, :]
            try: c_index_tr = concordance_index_ipcw(y_train_cause, y_test_cause, cif_cause[:, -1])[0]
            except: c_index_tr = np.nan
                
            try: ibs_tr = integrated_brier_score(y_train_cause, y_test_cause, 1.0 - cif_cause[:, brier_time_indices], brier_times)
            except: ibs_tr = np.nan
                
            auc_tr = []
            for i, t_idx in enumerate(auc_time_indices):
                try:
                    auc_t, _ = cumulative_dynamic_auc(y_train_cause, y_test_cause, cif_cause[:, t_idx], np.array([times_to_evaluate[i]]))
                    auc_tr.append(auc_t[0])
                except: auc_tr.append(np.nan)
                    
            unified_results.append({
                "Model": model_name, "Event": cause, "C-index": c_index_tr, "IBS": ibs_tr,
                "AUC 2yr": auc_tr[0], "AUC 5yr": auc_tr[1], "AUC 10yr": auc_tr[2]
            })

    df_results = pd.DataFrame(unified_results)
    numeric_cols = ["C-index", "IBS", "AUC 2yr", "AUC 5yr", "AUC 10yr"]
    df_results[numeric_cols] = df_results[numeric_cols].round(4)
    df_results = df_results.sort_values(by=["Event", "Model"]).reset_index(drop=True)
    return df_results


def build_table_3_hyperparams(model_instance, batch_size=64, lr=0.001, num_bins=20, patience=15):
    """
    Таблиця 3: Гіперпараметри та налаштування експерименту.
    Повертає Pandas DataFrame, витягуючи значення через інтроспекцію.
    """
    try: current_d_model = model_instance.feature_proj.out_features
    except AttributeError: current_d_model = "N/A"

    model_sig = inspect.signature(FinalTransformerModel.__init__)
    current_n_heads = model_sig.parameters['n_heads'].default if 'n_heads' in model_sig.parameters else "N/A"
    current_num_layers = model_sig.parameters['num_layers'].default if 'num_layers' in model_sig.parameters else "N/A"

    hyperparams_data = {
        "Гіперпараметр": [
            "d_model", "n_heads", "num_layers", "T (Time Bins)",
            "Batch Size", "Learning Rate", "Early Stopping Patience", "Optimizer"
        ],
        "Значення": [
            current_d_model, current_n_heads, current_num_layers, num_bins,
            batch_size, lr, patience, optim.Adam.__name__
        ],
        "Опис (Призначення)": [
            "Розмірність векторних ембедингів (feature projection)",
            "Кількість голів у механізмі уваги (Multi-head Attention)",
            "Кількість шарів енкодера Трансформера",
            "Кількість дискретних часових кошиків (за квантилями)",
            "Розмір пакету для навчання моделі",
            "Швидкість навчання оптимізатора",
            "Епох без покращення валідаційної помилки до зупинки",
            "Алгоритм оптимізації ваг нейромережі"
        ]
    }
    return pd.DataFrame(hyperparams_data)