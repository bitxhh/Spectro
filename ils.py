"""
spectrolib.ils
==============
Аппаратная функция прибора (instrument line shape, ILS).

Все ILS реализуют единый интерфейс: метод `kernel(grid_step, n_points=None)`
возвращает дискретное нормированное ядро на сетке с шагом `grid_step`.

Ширина ядра выбирается автоматически по характерной ширине ILS
(или задаётся через `width_factor`).

Поддерживаются:
    GaussILS   — гауссова форма (типичная для монохроматоров с щелью)
    LorentzILS — лоренцева форма
    VoigtILS   — свёртка Гаусс ⊗ Лоренц
    FromFileILS — произвольная экспериментальная форма из файла
                  (например, измеренный спектр излучения QD)

Свёртка делается через scipy.signal.fftconvolve, режим 'same',
с краевой защитой через расширение spectrum-данных по краю.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import wofz


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def fwhm_to_sigma(fwhm):
    """FWHM → σ для гауссианы. FWHM = 2·sqrt(2·ln2)·σ ≈ 2.355·σ"""
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def sigma_to_fwhm(sigma):
    """σ → FWHM для гауссианы."""
    return sigma * 2.0 * np.sqrt(2.0 * np.log(2.0))


def _check_uniform(grid, rtol=1e-6):
    """Проверяет равномерность сетки и возвращает шаг."""
    diffs = np.diff(grid)
    d_mean = np.mean(diffs)
    if np.any(np.abs(diffs - d_mean) > rtol * abs(d_mean)):
        raise ValueError(
            "Сетка должна быть равномерной для свёртки с ILS. "
            f"Максимальное отклонение шага: {np.max(np.abs(diffs - d_mean)):.3e}, "
            f"средний шаг: {d_mean:.3e}"
        )
    return float(abs(d_mean))


# ---------------------------------------------------------------------------
# Базовый класс ILS
# ---------------------------------------------------------------------------

class ILS:
    """
    Базовый интерфейс аппаратной функции.

    Подклассы реализуют:
        _profile(x) — нормированная (по площади) форма на сетке `x`,
                      где x — отклонение от центра в тех же единицах,
                      что и `grid_step` в .kernel().

    Поле `fwhm` хранит характерную ширину; по нему рассчитывается
    необходимая полуширина окна ядра (по умолчанию 5·FWHM).
    """

    fwhm: float
    width_factor: float = 5.0   # окно = ±width_factor · fwhm

    def _profile(self, x):
        raise NotImplementedError

    def kernel(self, grid_step, width_factor=None):
        """
        Дискретное нормированное ядро на сетке с шагом grid_step.

        Parameters
        ----------
        grid_step : float
            Шаг сетки спектра (в тех же единицах, что и fwhm: обычно нм).
        width_factor : float, optional
            Полуширина окна ядра в единицах FWHM. Дефолт — self.width_factor.
            Делать слишком маленьким нельзя (обрежется хвост), слишком большим
            — лишние вычисления.

        Returns
        -------
        np.ndarray
            Ядро длины 2k+1, нормированное так, что ∑kernel · grid_step ≈ 1.
        """
        wf = self.width_factor if width_factor is None else width_factor
        half_window = wf * self.fwhm
        k = max(1, int(np.ceil(half_window / grid_step)))
        x = np.arange(-k, k + 1) * grid_step
        ker = self._profile(x)
        # Нормировка по сумме · шаг (дискретный аналог ∫ = 1)
        ker = ker / (ker.sum() * grid_step)
        # Возвращаем уже отнормированное под convolve: ker * grid_step,
        # тогда ∑(ker_disc) = 1 и `convolve(values, ker_disc)` сохраняет уровень
        return ker * grid_step

    def convolve(self, values, grid):
        """
        Свернуть массив `values` на сетке `grid` с этой ILS.

        Краевая защита — отражение (как у np.pad с mode='reflect'):
        даёт меньше артефактов на краях, чем 'nearest', особенно для
        широкой ILS на узком окне.
        """
        step = _check_uniform(grid)
        ker = self.kernel(step)
        # Отступ для краевой защиты
        pad = (len(ker) - 1) // 2
        padded = np.pad(values, pad, mode='reflect')
        result = fftconvolve(padded, ker, mode='same')
        return result[pad:-pad] if pad > 0 else result


# ---------------------------------------------------------------------------
# Конкретные ILS
# ---------------------------------------------------------------------------

class GaussILS(ILS):
    """Гауссова ILS. Параметр — FWHM в тех же единицах, что сетка спектра."""

    def __init__(self, fwhm):
        self.fwhm = float(fwhm)
        self._sigma = fwhm_to_sigma(self.fwhm)

    def _profile(self, x):
        return np.exp(-0.5 * (x / self._sigma) ** 2) / (
            self._sigma * np.sqrt(2.0 * np.pi)
        )

    def __repr__(self):
        return f"GaussILS(fwhm={self.fwhm})"


class LorentzILS(ILS):
    """Лоренцева ILS. Параметр — FWHM."""

    def __init__(self, fwhm):
        self.fwhm = float(fwhm)
        self._gamma = self.fwhm / 2.0   # HWHM

    def _profile(self, x):
        return (self._gamma / np.pi) / (x ** 2 + self._gamma ** 2)

    def __repr__(self):
        return f"LorentzILS(fwhm={self.fwhm})"


class VoigtILS(ILS):
    """
    Voigt ILS = свёртка Гаусс ⊗ Лоренц.

    Параметры:
        fwhm_g — FWHM гауссовой компоненты
        fwhm_l — FWHM лоренцевой компоненты

    Эффективная FWHM voigt-профиля для выбора окна оценивается
    приближённой формулой Olivero & Longbothum:
        FWHM_v ≈ 0.5346·FWHM_l + sqrt(0.2166·FWHM_l² + FWHM_g²)
    """

    def __init__(self, fwhm_g, fwhm_l):
        self.fwhm_g = float(fwhm_g)
        self.fwhm_l = float(fwhm_l)
        self._sigma = fwhm_to_sigma(self.fwhm_g)
        self._gamma = self.fwhm_l / 2.0
        # Эффективная FWHM (для размера окна)
        self.fwhm = (
            0.5346 * self.fwhm_l
            + np.sqrt(0.2166 * self.fwhm_l ** 2 + self.fwhm_g ** 2)
        )

    def _profile(self, x):
        z = (x + 1j * self._gamma) / (self._sigma * np.sqrt(2.0))
        return np.real(wofz(z)) / (self._sigma * np.sqrt(2.0 * np.pi))

    def __repr__(self):
        return f"VoigtILS(fwhm_g={self.fwhm_g}, fwhm_l={self.fwhm_l})"


class FromFileILS(ILS):
    """
    Произвольная ILS, заданная массивами (offset, intensity).

    Использование:
    - `offset` — отклонение от центра (нм или см⁻¹, **те же единицы**, что
      и сетка спектра, на котором будет применяться ILS).
    - `intensity` — амплитуды формы. Знак не важен (внутренне нормируется).

    При вызове `.kernel(grid_step)` форма интерполируется на равномерную
    сетку с шагом `grid_step` и нормируется по площади.

    Это основной канал интеграции реальных измеренных спектров излучения
    QD/OLED-источников: измерил → передал в FromFileILS → ILS прибора готова.

    Опционально:
    - `auto_center=True` (дефолт) — сместит профиль так, чтобы максимум
      оказался в нуле. Полезно когда экспериментальный файл начинается
      не от центра.
    - `clip_negative=True` (дефолт) — обрежет отрицательные значения
      (артефакты вычитания фона).
    """

    def __init__(self, offset, intensity, auto_center=True, clip_negative=True):
        offset = np.asarray(offset, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        if offset.shape != intensity.shape:
            raise ValueError("offset и intensity должны быть одной длины")
        if len(offset) < 3:
            raise ValueError("Нужно минимум 3 точки в профиле ILS")

        # Сортировка по offset
        order = np.argsort(offset)
        offset = offset[order]
        intensity = intensity[order]

        if clip_negative:
            intensity = np.clip(intensity, 0.0, None)

        if auto_center:
            i_max = int(np.argmax(intensity))
            offset = offset - offset[i_max]

        self._offset = offset
        self._intensity = intensity

        # Оценка FWHM по дискретным данным — для подбора размера окна
        self.fwhm = self._estimate_fwhm(offset, intensity)

    @staticmethod
    def _estimate_fwhm(offset, intensity):
        """Оценка FWHM по точкам пересечения с уровнем 0.5·max."""
        peak = intensity.max()
        if peak <= 0:
            raise ValueError("Профиль ILS пуст (max <= 0)")
        half = 0.5 * peak
        above = intensity >= half
        if not np.any(above):
            return abs(offset[-1] - offset[0])
        idx = np.where(above)[0]
        # Берём ровно дискретные точки пересечения уровня 0.5·max,
        # без линейной интерполяции между бинами. Точность — порядка
        # шага сетки offset; для оценки размера окна ядра этого достаточно.
        left = offset[idx[0]]
        right = offset[idx[-1]]
        fwhm = right - left
        if fwhm <= 0:
            # Один пик в одной точке — вернём шаг сетки как минимальную ширину
            d = np.median(np.diff(offset))
            fwhm = max(d, 1e-12)
        return float(fwhm)

    def _profile(self, x):
        # Линейная интерполяция, за пределами — ноль
        return np.interp(x, self._offset, self._intensity, left=0.0, right=0.0)

    def __repr__(self):
        return (f"FromFileILS(n_points={len(self._offset)}, "
                f"fwhm≈{self.fwhm:.4g})")


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def gauss_convolve(values, grid, fwhm):
    """
    Свёртка с гауссовой ILS — обёртка для обратной совместимости.

    Эквивалент: GaussILS(fwhm).convolve(values, grid).
    """
    return GaussILS(fwhm).convolve(values, grid)
