"""
spectrolib.physics
==================
Физические константы, перевод единиц, закон Бугера–Ламберта–Бера.

Константы берутся из scipy.constants (CODATA), чтобы обеспечить
точное совпадение с табличными HITRAN-данными в валидационных тестах.
"""

import numpy as np
from scipy import constants as _const


# Физические константы из scipy (CODATA)
K_B    = _const.Boltzmann            # Дж/К
H      = _const.Planck               # Дж·с
C_CGS  = _const.c * 100              # см/с
N_A    = _const.Avogadro             # 1/моль
ATM_PA = _const.atm                  # Па/атм


# ---------------------------------------------------------------------------
# Перевод единиц
# ---------------------------------------------------------------------------

def nm_to_wavenumber(wavelength_nm):
    """Нанометры → волновые числа (см⁻¹). Порядок переворачивается."""
    return 1e7 / np.asarray(wavelength_nm, dtype=float)


def wavenumber_to_nm(wavenumber_cm):
    """Волновые числа (см⁻¹) → нанометры."""
    return 1e7 / np.asarray(wavenumber_cm, dtype=float)


def ppm_to_fraction(c_ppm):
    """ppm → мольная доля. 1 ppm = 1e-6."""
    return c_ppm * 1e-6


def fraction_to_ppm(x):
    """Мольная доля → ppm."""
    return x * 1e6


# ---------------------------------------------------------------------------
# Термодинамика
# ---------------------------------------------------------------------------

def number_density(T_K, p_atm):
    """
    Числовая плотность идеального газа (молекул/см³).

    Parameters
    ----------
    T_K : float
        Температура, K.
    p_atm : float
        Давление, атм.

    Returns
    -------
    float
        Молекул/см³. При нормальных условиях (273.15 K, 1 атм) ≈ 2.687e19
        (число Лошмидта).
    """
    return p_atm * ATM_PA / (K_B * T_K) * 1e-6


# ---------------------------------------------------------------------------
# Закон Бугера–Ламберта–Бера
# ---------------------------------------------------------------------------

def beer_lambert(absorption_coeff, c_ppm, L_cm, T_K=296, p_atm=1.0):
    """
    Оптическая плотность по закону БЛБ для одной молекулы.

    OD = σ · N · L,  где
        σ — сечение поглощения [см²/молекула],
        N — числовая плотность молекулы данного сорта [молекул/см³],
        L — длина пути [см].

    T = exp(−OD).

    Parameters
    ----------
    absorption_coeff : np.ndarray
        Сечение поглощения σ, см²/молекула (выдаёт hapi).
    c_ppm : float
        Концентрация молекулы в ppm (parts per million).
    L_cm : float
        Длина оптического пути, см.
    T_K, p_atm : float
        Температура и давление для расчёта числовой плотности.

    Returns
    -------
    OD : np.ndarray
        Оптическая плотность (безразмерная).
    """
    N_total    = number_density(T_K, p_atm)
    N_molecule = ppm_to_fraction(c_ppm) * N_total
    return absorption_coeff * N_molecule * L_cm
