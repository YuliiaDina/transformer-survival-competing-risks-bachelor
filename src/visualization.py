import matplotlib.pyplot as plt
import numpy as np
import torch
from sksurv.nonparametric import kaplan_meier_estimator
from src.data_prep import digitize_time

def plot_loss_curves(train_hist_A, val_hist_A, train_hist_B, val_hist_B):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(train_hist_A, label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(val_hist_A, label='Validation Loss', color='orange', linewidth=2)
    axes[0].set_title('Variant A (Стандартна увага)')
    axes[0].set_xlabel('Епоха')
    axes[0].set_ylabel('Loss (NLL)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_hist_B, label='Train Loss', color='blue', linewidth=2)
    axes[1].plot(val_hist_B, label='Validation Loss', color='orange', linewidth=2)
    axes[1].set_title('Variant B (Монотонна увага)')
    axes[1].set_xlabel('Епоха')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_cif_comparison(models_cox, model_A, model_B, X_test_scaled, X_test_tensor, brier_times, bins):
    patients_to_plot = X_test_scaled.iloc[0:2]
    patients_tensor = X_test_tensor[0:2]

    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        _, _, cif_A = model_A(patients_tensor)
        _, _, cif_B = model_B(patients_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {'Cox': 'green', 'VarA': 'blue', 'VarB': 'red'}

    for i in range(2): 
        ax = axes[i]
        for cause in [1, 2]:
            linestyle = '--' if cause == 1 else '-'
            
            surv_funcs = models_cox[cause].predict_survival_function(patients_to_plot)
            cif_cox = 1 - surv_funcs[i](brier_times)
            ax.step(brier_times, cif_cox, where="post", color=colors['Cox'], linestyle=linestyle, alpha=0.7, linewidth=2, label=f'Cox (Подія {cause})' if i==0 else "")

            brier_time_indices = digitize_time(brier_times, bins)
            cif_A_plot = cif_A[i, cause-1, brier_time_indices].numpy()
            ax.step(brier_times, cif_A_plot, where="post", color=colors['VarA'], linestyle=linestyle, alpha=0.7, linewidth=2, label=f'Variant A (Подія {cause})' if i==0 else "")

            cif_B_plot = cif_B[i, cause-1, brier_time_indices].numpy()
            ax.step(brier_times, cif_B_plot, where="post", color=colors['VarB'], linestyle=linestyle, alpha=0.7, linewidth=2, label=f'Variant B (Подія {cause})' if i==0 else "")

        ax.set_title(f'Пацієнт {i+1}: Порівняння моделей')
        ax.set_xlabel('Час (місяці)')
        ax.set_ylabel('Ймовірність події (CIF)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=6)
    plt.tight_layout()
    plt.show()