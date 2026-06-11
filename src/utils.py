import torch

def run_cif_checks(model, model_name, X_tensor):

    print(f"\nПеревірка для Variant {model_name}:")
    model.eval()
    with torch.no_grad():
        _, _, cif_pred = model(X_tensor)

    # 1. Невід'ємність (>= 0)
    assert (cif_pred >= 0).all(), "Помилка 1: Є від'ємні значення CIF"
    print(" 1. Невід'ємність: всі значення CIF >= 0")

    # 2. Монотонність (не зменшується з часом)
    diffs = cif_pred[:, :, 1:] - cif_pred[:, :, :-1]
    assert (diffs >= -1e-6).all(), "Помилка 2: Порушено монотонність"
    print(" 2. Монотонність: CIF не спадає з часом")

    # 3. Обмеження суми (<= 1)
    cif_sum = cif_pred.sum(dim=1) # [batch, T]
    assert (cif_sum <= 1.0 + 1e-6).all(), "Помилка 3: Сума CIF перевищує 1.0"
    print(" 3. Обмеження суми: сума за причинами <= 1")

    # 4. Звіт по останньому інтервалу
    mean_sum = cif_sum[:, -1].mean().item()
    min_sum = cif_sum[:, -1].min().item()
    max_sum = cif_sum[:, -1].max().item()
    print(f" [i] Сума CIF на останньому інтервалі: середнє={mean_sum:.4f}, min={min_sum:.4f}, max={max_sum:.4f}")

    # Ймовірність вижити S(t)
    print(f"  -> Це означає, що середня ймовірність пацієнта вижити без жодної події S(t) до кінця дослідження складає {(1-mean_sum)*100:.1f}%")
