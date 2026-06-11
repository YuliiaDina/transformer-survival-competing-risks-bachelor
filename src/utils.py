import torch

def print_table_1(df_original):
    print("\n[Table 1] Характеристики пацієнтів (Mean ± SD)")
    for col in ['age', 'dxyr', 'hgb', 'creat', 'mspike']:
        print(f"- {col.upper()}: {df_original[col].mean():.2f} ± {df_original[col].std():.2f}")

    print("\n[Стать (Кількість / Відсоток)]")
    sex_total = len(df_original)
    for sex_val, count in df_original['sex'].value_counts().items():
        print(f"- {sex_val}: {count} ({count/sex_total*100:.1f}%)")

    print("\n[Розподіл подій]")
    event_names = {0: "0 (Цензуровано)", 1: "1 (Прогресія)", 2: "2 (Смерть)"}
    for ev_code, count in df_original['event_type'].value_counts().sort_index().items():
        print(f"- {event_names[ev_code]}: {count} ({count/sex_total*100:.1f}%)")

def run_cif_checks(model, model_name, X_tensor):
    print(f"\nМатематична перевірка для Variant {model_name}:")
    model.eval()
    with torch.no_grad():
        _, _, cif_pred = model(X_tensor)

    assert (cif_pred >= 0).all(), "Помилка 1: Є від'ємні значення CIF"
    print("  [✓] Невід'ємність: всі значення CIF >= 0")

    diffs = cif_pred[:, :, 1:] - cif_pred[:, :, :-1]
    assert (diffs >= -1e-6).all(), "Помилка 2: Порушено монотонність"
    print("  [✓] Монотонність: CIF не спадає з часом")

    cif_sum = cif_pred.sum(dim=1)
    assert (cif_sum <= 1.0 + 1e-6).all(), "Помилка 3: Сума CIF перевищує 1.0"
    print("  [✓] Обмеження суми: сума за причинами <= 1")
    print(f"  [i] Середня ймовірність вижити без подій до кінця: {(1 - cif_sum[:, -1].mean().item())*100:.1f}%")