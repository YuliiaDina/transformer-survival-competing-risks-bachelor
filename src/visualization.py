import matplotlib.pyplot as plt
import numpy as np
import torch
from sksurv.nonparametric import kaplan_meier_estimator
from src.data_prep import digitize_time, prepare_survival_data


def plot_loss_curves(train_hist_A, val_hist_A, train_hist_B, val_hist_B):

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_hist_A, label="Train Loss", color="blue", linewidth=2)
    axes[0].plot(val_hist_A, label="Validation Loss", color="orange", linewidth=2)
    axes[0].set_title("Variant A (Стандартна увага)")
    axes[0].set_xlabel("Епоха")
    axes[0].set_ylabel("Loss (NLL)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(train_hist_B, label="Train Loss", color="blue", linewidth=2)
    axes[1].plot(val_hist_B, label="Validation Loss", color="orange", linewidth=2)
    axes[1].set_title("Variant B (Монотонна увага)")
    axes[1].set_xlabel("Епоха")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_cif_comparison(
    models_cox, model_A, model_B, X_test_scaled, X_test_tensor, brier_times, bins
):

    patients_to_plot = X_test_scaled.iloc[0:2]
    patients_tensor = X_test_tensor[0:2]

    model_A.eval()
    model_B.eval()
    with torch.no_grad():
        _, _, cif_A = model_A(patients_tensor)
        _, _, cif_B = model_B(patients_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"Cox": "green", "VarA": "blue", "VarB": "red"}

    for i in range(2):
        ax = axes[i]
        for cause in [1, 2]:
            linestyle = "--" if cause == 1 else "-"

            surv_funcs = models_cox[cause].predict_survival_function(patients_to_plot)
            cif_cox = 1 - surv_funcs[i](brier_times)
            ax.step(
                brier_times,
                cif_cox,
                where="post",
                color=colors["Cox"],
                linestyle=linestyle,
                alpha=0.7,
                linewidth=2,
                label=f"Cox (Подія {cause})" if i == 0 else "",
            )

            brier_time_indices = np.digitize(brier_times, bins) - 1

            brier_time_indices = np.clip(brier_time_indices, 0, cif_A.shape[2] - 1)

            cif_A_plot = cif_A[i, cause - 1, brier_time_indices].numpy()
            ax.step(
                brier_times,
                cif_A_plot,
                where="post",
                color=colors["VarA"],
                linestyle=linestyle,
                alpha=0.7,
                linewidth=2,
                label=f"Variant A (Подія {cause})" if i == 0 else "",
            )

            cif_B_plot = cif_B[i, cause - 1, brier_time_indices].numpy()
            ax.step(
                brier_times,
                cif_B_plot,
                where="post",
                color=colors["VarB"],
                linestyle=linestyle,
                alpha=0.7,
                linewidth=2,
                label=f"Variant B (Подія {cause})" if i == 0 else "",
            )

        ax.set_title(f"Пацієнт {i+1}: Порівняння моделей")
        ax.set_xlabel("Час (місяці)")
        ax.set_ylabel("Ймовірність події (CIF)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=6)
    plt.tight_layout()
    plt.show()


def get_observed_incidence(y_true, t_eval):

    times, survival_probs = kaplan_meier_estimator(y_true["event"], y_true["time"])
    valid_idx = np.where(times <= t_eval)[0]
    if len(valid_idx) == 0:
        return 0.0
    return 1.0 - survival_probs[valid_idx[-1]]


def plot_calibration_curves_survival(
    models_cox,
    model_A_hist,
    model_B_hist,
    X_test_scaled,
    X_test_tensor,
    df_test,
    bins,
    target_time=60,
):

    target_bin_idx = digitize_time(np.array([target_time]), bins)[0]

    model_A_hist.eval()
    model_B_hist.eval()
    with torch.no_grad():
        _, _, cif_A_all = model_A_hist(X_test_tensor)
        _, _, cif_B_all = model_B_hist(X_test_tensor)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for cause in [1, 2]:
        ax = axes[cause - 1]
        y_test_cause = prepare_survival_data(df_test, cause)

        preds = {}

        # Cox
        surv_funcs = models_cox[cause].predict_survival_function(X_test_scaled)
        preds["Cox"] = np.array([1 - fn(target_time) for fn in surv_funcs])

        # Variant A
        preds["Variant A (Стандартна)"] = cif_A_all[
            :, cause - 1, target_bin_idx
        ].numpy()

        # Variant B
        preds["Variant B (Монотонна)"] = cif_B_all[:, cause - 1, target_bin_idx].numpy()

        colors = {
            "Cox": "green",
            "Variant A (Стандартна)": "blue",
            "Variant B (Монотонна)": "red",
        }

        ax.plot([0, 1], [0, 1], "k--", label="Ідеальне калібрування")

        for name, risk_probs in preds.items():

            quantiles = np.quantile(risk_probs, np.linspace(0, 1, 6))
            quantiles[0] -= 1e-6
            quantiles[-1] += 1e-6

            bin_indices = np.digitize(risk_probs, quantiles) - 1

            mean_pred = []
            obs_incidence = []

            for b in range(5):
                mask = bin_indices == b
                if mask.sum() > 0:
                    mean_pred.append(risk_probs[mask].mean())
                    obs_incidence.append(
                        get_observed_incidence(y_test_cause[mask], target_time)
                    )

            ax.plot(
                mean_pred,
                obs_incidence,
                marker="o",
                color=colors[name],
                linewidth=2,
                markersize=8,
                label=name,
            )

        ax.set_title(f"Калібрування: Подія {cause} (на {target_time} місяців)")
        ax.set_xlabel("Прогнозована ймовірність (Predicted CIF)")
        ax.set_ylabel("Спостережувана ймовірність (Observed)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
