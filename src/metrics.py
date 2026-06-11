from typing import Callable, List, Union,Sequence

import numpy as np


def compute_cox_cif(
    surv_fns_k: Sequence[Callable[[float], float]],
    surv_fns_all: Union[np.ndarray, List[Callable[[float], float]]],
    times: Union[List[float], np.ndarray],
) -> np.ndarray:
    n = len(surv_fns_k)
    cif = np.zeros((n, len(times)))

    for i in range(n):
        S_k = np.array([surv_fns_k[i](t) for t in times])
        S_all = np.array([surv_fns_all[i](t) for t in times])

        S_k_prev = np.concatenate([[1.0], S_k[:-1]])
        S_all_prev = np.concatenate([[1.0], S_all[:-1]])

        h_k_dt = np.where(S_k_prev > 0, (S_k_prev - S_k) / S_k_prev, 0.0)
        cif[i] = np.cumsum(h_k_dt * S_all_prev)

    return cif
