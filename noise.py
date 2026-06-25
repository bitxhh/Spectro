"""
spectrolib.noise
================
Модели шумов для синтетических спектров.

Поддерживается шесть физически осмысленных видов шума:

1. Тепловой (Johnson) — гауссов аддитивный, не зависит от сигнала.
   Параметр: sigma (СКО в единицах transmittance).

2. Дробовой (shot) — пуассонов, СКО ∝ √I.
   Параметр: n_photons_max — число фотонов на детекторе при T=1
   (на максимуме сигнала). Чем больше — тем меньше относительный шум.

3. Цветной AR(1) — гауссов с ненулевой автокорреляцией.
   Параметр: sigma, ar_coefficient (-1 < ρ < 1).
   ρ > 0 — низкочастотный шум, ρ < 0 — высокочастотный.

4. Низкочастотный дрейф (1/f-подобный) — медленный полиномиальный
   или синусоидальный сдвиг базовой линии.
   Параметры: amplitude, n_terms (порядок Фурье-разложения).

5. Периодические наводки — синусоидальные пики на фиксированных
   частотах (50 Гц от сети, мерцание источника и т.п.).
   Параметры: список (частота_в_единицах_сетки, амплитуда, фаза).

6. Одиночные выбросы (spikes) — космические частицы, дефекты пикселей.
   Параметры: rate (вероятность на пиксель), amplitude_range.

Все шумы добавляются в пространстве **transmittance** (а не OD),
потому что:
- именно там детектор регистрирует сигнал;
- дробовой шум ∝ √I невозможен в OD-пространстве;
- тепловой шум на ADC по физике аддитивен в счётах детектора, не в OD.

Дрейф базовой линии — единственное исключение, обычно его удобнее
задавать в OD (медленные изменения коэффициента поглощения среды
или загрязнения окон). Поэтому для него — отдельный механизм.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Отдельные источники шума
# ---------------------------------------------------------------------------

def thermal_noise(transmittance, sigma, rng):
    """
    Тепловой (Johnson) шум: гауссов аддитивный в T.

    Возвращает шумовую добавку (не зашумлённый сигнал).
    """
    return rng.normal(0.0, sigma, size=transmittance.shape)


def shot_noise(transmittance, n_photons_max, rng):
    """
    Дробовой шум: пуассонов в счётах фотонов.

    Модель: число фотонов на пиксель ~ Poisson(I·n_max),
    где I = transmittance ∈ [0, 1]. Возвращаем относительную ошибку:
    (counts/n_max) − I.

    При больших n_max приближается к Гауссу с σ = √(I/n_max).
    """
    expected = np.clip(transmittance, 0.0, None) * n_photons_max
    counts = rng.poisson(expected)
    return counts / n_photons_max - transmittance


def colored_ar1_noise(transmittance, sigma, ar_coefficient, rng):
    """
    Цветной шум: процесс AR(1) x[n] = ρ·x[n-1] + ε[n], ε ~ N(0, σ_ε).

    Параметр sigma задаёт **установившееся** СКО процесса (не σ_ε):
        σ_ε = σ · √(1 − ρ²)
    Это удобнее для пользователя: задаёшь интенсивность шума, а не
    параметр генератора.

    ρ > 0 — низкочастотный (соседние отсчёты коррелированы).
    ρ < 0 — высокочастотный (знакочередующийся).
    """
    rho = float(ar_coefficient)
    if not -1.0 < rho < 1.0:
        raise ValueError(f"ar_coefficient должен быть в (−1, 1), получено {rho}")
    n = len(transmittance)
    sigma_eps = sigma * np.sqrt(1.0 - rho ** 2)
    eps = rng.normal(0.0, sigma_eps, size=n)
    x = np.empty(n)
    # Старт из стационарного распределения, чтобы не было разогрева
    x[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + eps[i]
    return x


def baseline_drift(grid, amplitude, n_terms, rng):
    """
    Низкочастотный дрейф как сумма синусов со случайными фазами и
    убывающими амплитудами (имитация 1/f-спектра).

    drift(λ) = A · Σₖ (1/k) · sin(π·k·(λ−λ₀)/L + φₖ),  k = 1..n_terms

    Возвращает добавку **в OD** (медленные изменения уровня поглощения
    из-за дрейфа источника, нагрева, загрязнения окон).
    """
    n = len(grid)
    if n_terms < 1:
        return np.zeros(n)
    x = (grid - grid[0]) / (grid[-1] - grid[0])  # ∈ [0, 1]
    drift = np.zeros(n)
    for k in range(1, n_terms + 1):
        phi = rng.uniform(0, 2 * np.pi)
        drift += (1.0 / k) * np.sin(np.pi * k * x + phi)
    # Нормировка под амплитуду размах ≈ 2A
    drift = drift / np.max(np.abs(drift)) * amplitude
    return drift


def periodic_interference(grid, components, rng):
    """
    Сумма синусоидальных наводок.

    Parameters
    ----------
    components : list of (period, amplitude, phase_or_None)
        period — пространственный период в единицах сетки (нм);
        amplitude — амплитуда (в T);
        phase — если None, выбирается случайно.

    Возвращает добавку в T.
    """
    n = len(grid)
    out = np.zeros(n)
    for period, amp, phase in components:
        if phase is None:
            phase = rng.uniform(0, 2 * np.pi)
        out += amp * np.sin(2 * np.pi * grid / period + phase)
    return out


def spike_noise(transmittance, rate, amplitude_range, rng):
    """
    Одиночные выбросы (космические частицы, hot pixels).

    Parameters
    ----------
    rate : float
        Вероятность спайка на пиксель ∈ [0, 1].
    amplitude_range : (lo, hi)
        Диапазон амплитуд (предполагается lo, hi ≥ 0).
        Знак выбирается случайно (в обе стороны от текущего уровня T).

    Notes
    -----
    Параметр `transmittance` принимается ради единообразия сигнатуры
    с другими функциями шума, но сам массив значений не используется —
    позиции и амплитуды спайков от текущего T не зависят.

    Добавка идёт в T-пространство как есть, без последующего
    клиппинга к [0, 1]: после большого положительного спайка
    наблюдаемое T в этой точке может оказаться > 1, а соответствующая
    OD — отрицательной. Это намеренное поведение (имитация выброса
    АЦП); если в downstream-обработке нужны физически валидные
    значения, обрабатывай выбросы отдельным шагом.
    """
    n = len(transmittance)
    mask = rng.random(n) < rate
    n_spikes = int(mask.sum())
    if n_spikes == 0:
        return np.zeros(n)
    lo, hi = amplitude_range
    amps = rng.uniform(lo, hi, size=n_spikes)
    signs = rng.choice([-1, 1], size=n_spikes)
    out = np.zeros(n)
    out[mask] = amps * signs
    return out


# ---------------------------------------------------------------------------
# NoiseModel — контейнер для всех видов шума
# ---------------------------------------------------------------------------

@dataclass
class NoiseModel:
    """
    Контейнер параметров для всех видов шума.

    Все поля опциональны — если параметр не задан, соответствующий
    шум не добавляется.

    Применяется к спектру через NoiseModel.apply(spectrum), который
    разделяет шум по пространствам:
      - thermal, shot, colored_ar1, periodic, spikes  →  в T
      - baseline_drift                                 →  в OD

    Returns: тройка (delta_T, delta_OD, contributions),
        где contributions — dict с раздельными вкладами каждого
        источника для диагностики.

    Параметры по умолчанию подобраны так, чтобы соответствующий шум
    был выключен.
    """
    # 1. Тепловой
    thermal_sigma: Optional[float] = None

    # 2. Дробовой (число фотонов при T=1)
    shot_n_photons_max: Optional[float] = None

    # 3. Цветной AR(1)
    colored_sigma: Optional[float] = None
    colored_ar: float = 0.3

    # 4. Дрейф базовой линии (в OD)
    drift_amplitude: Optional[float] = None
    drift_n_terms: int = 3

    # 5. Периодические наводки: list of (period_nm, amplitude, phase)
    periodic: List[Tuple[float, float, Optional[float]]] = field(
        default_factory=list
    )

    # 6. Одиночные выбросы
    spike_rate: Optional[float] = None
    spike_amplitude_range: Tuple[float, float] = (0.05, 0.2)

    def apply(self, transmittance, grid, rng):
        """
        Сгенерировать суммарный шум.

        Возвращает тройку (delta_T, delta_OD, contributions) — все три
        значения, а не два (как могло читаться раньше). Распаковывай
        соответственно: `dT, dOD, contrib = model.apply(...)`.

        Returns
        -------
        delta_T : np.ndarray
            Аддитивная добавка к transmittance.
        delta_OD : np.ndarray
            Аддитивная добавка к OD (только дрейф базовой линии).
        contributions : dict
            Раздельные вклады каждого источника (для диагностики).
        """
        n = len(transmittance)
        delta_T = np.zeros(n)
        delta_OD = np.zeros(n)
        contributions = {}

        if self.thermal_sigma is not None:
            c = thermal_noise(transmittance, self.thermal_sigma, rng)
            delta_T += c
            contributions['thermal'] = c

        if self.shot_n_photons_max is not None:
            c = shot_noise(transmittance, self.shot_n_photons_max, rng)
            delta_T += c
            contributions['shot'] = c

        if self.colored_sigma is not None:
            c = colored_ar1_noise(transmittance, self.colored_sigma,
                                   self.colored_ar, rng)
            delta_T += c
            contributions['colored'] = c

        if self.periodic:
            c = periodic_interference(grid, self.periodic, rng)
            delta_T += c
            contributions['periodic'] = c

        if self.spike_rate is not None:
            c = spike_noise(transmittance, self.spike_rate,
                            self.spike_amplitude_range, rng)
            delta_T += c
            contributions['spikes'] = c

        if self.drift_amplitude is not None:
            c = baseline_drift(grid, self.drift_amplitude,
                               self.drift_n_terms, rng)
            delta_OD += c
            contributions['drift'] = c

        return delta_T, delta_OD, contributions

    def to_dict(self):
        """Сериализация для метаданных спектра."""
        return {
            'thermal_sigma': self.thermal_sigma,
            'shot_n_photons_max': self.shot_n_photons_max,
            'colored_sigma': self.colored_sigma,
            'colored_ar': self.colored_ar,
            'drift_amplitude': self.drift_amplitude,
            'drift_n_terms': self.drift_n_terms,
            'periodic': self.periodic,
            'spike_rate': self.spike_rate,
            'spike_amplitude_range': self.spike_amplitude_range,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'NoiseModel':
        """
        Создать NoiseModel из словаря (YAML/JSON-конфиг).

        Поддерживаются все поля to_dict; дополнительно допускаются
        мета-поля 'name', 'reference', 'notes' — они игнорируются здесь
        (но могут читаться panel-like загрузчиком).
        """
        fields = {
            'thermal_sigma', 'shot_n_photons_max',
            'colored_sigma', 'colored_ar',
            'drift_amplitude', 'drift_n_terms',
            'periodic', 'spike_rate', 'spike_amplitude_range',
        }
        # Скалярные float-поля: страхуемся от YAML 1.1, который
        # распознаёт "1.0e6" как строку (без знака в экспоненте).
        float_fields = {
            'thermal_sigma', 'shot_n_photons_max',
            'colored_sigma', 'colored_ar',
            'drift_amplitude', 'spike_rate',
        }
        kwargs = {k: v for k, v in d.items() if k in fields and v is not None}
        for k in list(kwargs):
            if k in float_fields and isinstance(kwargs[k], str):
                kwargs[k] = float(kwargs[k])
        # periodic в YAML — список списков (period, amp, phase); приводим к tuple.
        if 'periodic' in kwargs:
            kwargs['periodic'] = [tuple(p) for p in kwargs['periodic']]
        if 'spike_amplitude_range' in kwargs:
            kwargs['spike_amplitude_range'] = tuple(kwargs['spike_amplitude_range'])
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path) -> 'NoiseModel':
        """
        Загрузить NoiseModel из YAML/JSON.

        Поддерживаются те же расширения, что и MixturePanel: .yaml/.yml/.json.
        Поля верхнего уровня — параметры NoiseModel; меta-поля name/reference/notes
        игнорируются (хранятся отдельно через meta-словарь, если нужно).
        """
        from pathlib import Path as _Path
        import json as _json

        p = _Path(path)
        if not p.exists():
            raise FileNotFoundError(f"NoiseModel: файл не найден: {p}")
        text = p.read_text(encoding='utf-8')
        ext = p.suffix.lower()
        if ext in ('.yaml', '.yml'):
            try:
                import yaml as _yaml
            except ImportError as e:
                raise ImportError(
                    "Для чтения YAML установи pyyaml: pip install pyyaml."
                ) from e
            data = _yaml.safe_load(text)
        elif ext == '.json':
            data = _json.loads(text)
        else:
            raise ValueError(
                f"Поддерживаются .yaml/.yml/.json, получено {ext!r}"
            )
        return cls.from_dict(data or {})


def load_noise_model(path) -> 'NoiseModel':
    """Шорткат: load_noise_model('foo.yaml') == NoiseModel.from_file(...)."""
    return NoiseModel.from_file(path)
