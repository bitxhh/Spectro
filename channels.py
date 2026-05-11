"""
spectrolib.channels
===================
Многоканальная регистрация: имитация прибора с дискретными
спектральными каналами (OLED-пиксели, фильтры на квантовых точках).

Концепция:

Реальный спектрометр высокого разрешения регистрирует тысячи точек на
тонкой сетке. Прибор на OLED/QD-пикселях — N интегральных значений по
широким полосам каналов. Каждый канал — это полоса пропускания, и
зарегистрированный сигнал в канале это:

    I_k = ∫ T(λ) · φ_k(λ) dλ  /  ∫ φ_k(λ) dλ

где φ_k — нормированная функция чувствительности k-го канала
(гауссова, лоренцева, или измеренная экспериментально через FromFileILS).

Этот модуль предоставляет:

- **Channel** — один канал с центром, шириной и формой полосы
- **ChannelSet** — набор каналов (15-30 штук для типичного прибора)
- **ChannelizedSpectrum** — результат регистрации спектра прибором,
  массив длины N плюс метаданные

Конфиги наборов каналов — в YAML, по аналогии с биомаркерными панелями.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence, Union
from pathlib import Path
import json
import numpy as np

from .ils import GaussILS, LorentzILS, VoigtILS, FromFileILS, ILS


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

@dataclass
class Channel:
    """
    Один спектральный канал прибора.

    center_nm : float
        Центральная длина волны полосы, нм.
    fwhm_nm : float
        Полная ширина на полувысоте полосы пропускания, нм.

        Внимание: используется для аналитических форм ('gauss',
        'lorentz', 'voigt') и для подбора ширины бара в .plot().
        Если в `shape` передан ILS-объект (например, FromFileILS) —
        реальная ширина определяется самим объектом, а `fwhm_nm`
        для построения профиля не учитывается (см. ниже).
    shape : str | ILS
        Форма полосы. Строки: 'gauss' (типично для QD), 'lorentz', 'voigt'.
        Альтернативно — готовый ILS-объект (например, FromFileILS из
        измеренного спектра эмиссии конкретной квантовой точки).

        В случае ILS-объекта профиль канала берётся целиком из
        `shape._profile(x)`, а параметр `fwhm_nm` используется только
        в репрезентации и в визуализации. Если нужно, чтобы заявленная
        и реальная ширина совпадали — следи за этим на уровне самого
        ILS-объекта.
    name : str, optional
        Метка канала (например, "QD-755" или "ch_0").
    notes : str, optional
        Произвольный комментарий (партия QD, дата измерения...)
    """
    center_nm: float
    fwhm_nm: float
    shape: Union[str, ILS] = 'gauss'
    name: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if self.fwhm_nm <= 0:
            raise ValueError(f"fwhm_nm должна быть > 0, получено {self.fwhm_nm}")
        if isinstance(self.shape, str):
            self.shape = self.shape.lower()
            if self.shape not in ('gauss', 'lorentz', 'voigt'):
                raise ValueError(
                    f"shape должна быть 'gauss', 'lorentz', 'voigt' "
                    f"или объектом ILS, получено {self.shape!r}"
                )

    def response(self, wavelength_nm):
        """
        Функция чувствительности канала на сетке wavelength_nm.

        Нормирована так, что ∫ response · dλ = 1, то есть это функция
        веса, с которой канал интегрирует входной спектр.

        Parameters
        ----------
        wavelength_nm : array-like
            Сетка длин волн (нм), на которой нужно вычислить отклик.

        Returns
        -------
        np.ndarray

        Notes
        -----
        Финальная нормировка выполняется численно через
        `area = profile.sum() * d_wl`, где `d_wl = mean(diff(wl))`.
        Это требует, чтобы переданная сетка была равномерной;
        для неравномерных сеток нормировка вернёт некорректный масштаб.
        Если центр канала лежит вне переданного диапазона и в окне
        нет заметной части профиля, `area` может оказаться около нуля
        и нормировка пропустится (вернётся профиль как есть, фактически
        нули). В обычном пайплайне такие случаи отсекает `channelize`
        выше по стеку и возвращает T = 1.0 для канала.
        """
        wl = np.asarray(wavelength_nm, dtype=float)
        x = wl - self.center_nm

        if isinstance(self.shape, ILS):
            # Если форма — объект ILS, используем её _profile
            # (FromFileILS, например). Forma уже нормирована по площади.
            profile = self.shape._profile(x)
        elif self.shape == 'gauss':
            sigma = self.fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
            profile = np.exp(-0.5 * (x / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi)
            )
        elif self.shape == 'lorentz':
            gamma = self.fwhm_nm / 2
            profile = (gamma / np.pi) / (x ** 2 + gamma ** 2)
        elif self.shape == 'voigt':
            # Симметричный voigt: σ_g = σ_l (грубое приближение,
            # для канала обычно достаточно)
            from scipy.special import wofz
            sigma = self.fwhm_nm / (2 * np.sqrt(2 * np.log(2))) / np.sqrt(2)
            gamma = self.fwhm_nm / 4
            z = (x + 1j * gamma) / (sigma * np.sqrt(2.0))
            profile = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
        else:
            raise ValueError(f"Неизвестная форма: {self.shape}")

        # Дополнительная нормировка по площади (на случай дискретизации)
        d_wl = np.abs(np.mean(np.diff(wl)))
        area = profile.sum() * d_wl
        if area > 0:
            profile = profile / area
        return profile

    def __repr__(self):
        shape_str = (self.shape if isinstance(self.shape, str)
                     else type(self.shape).__name__)
        nm = f' "{self.name}"' if self.name else ''
        return (f"Channel({self.center_nm:.1f} нм ± {self.fwhm_nm/2:.2f}, "
                f"{shape_str}{nm})")

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация для YAML/JSON. ILS-объекты не сохраняются."""
        d = {
            'center_nm': self.center_nm,
            'fwhm_nm': self.fwhm_nm,
        }
        if isinstance(self.shape, str):
            d['shape'] = self.shape
        else:
            d['shape'] = type(self.shape).__name__
            d['shape_note'] = (
                "ILS-объект не сериализуется в YAML; "
                "загрузить отдельно через FromFileILS."
            )
        if self.name:
            d['name'] = self.name
        if self.notes:
            d['notes'] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Channel':
        return cls(
            center_nm=float(d['center_nm']),
            fwhm_nm=float(d['fwhm_nm']),
            shape=d.get('shape', 'gauss'),
            name=d.get('name'),
            notes=d.get('notes'),
        )


# ---------------------------------------------------------------------------
# ChannelSet
# ---------------------------------------------------------------------------

@dataclass
class ChannelSet:
    """
    Набор каналов прибора — конфигурация многоканального регистратора.

    Хранит каналы плюс метаданные о наборе (имя, источник, заметки).
    Поддерживает загрузку и сохранение в YAML/JSON.

    Главное использование:

        channels = ChannelSet.from_file('qd_set_v1.yaml')
        ch_spec = spec.to_channels(channels)

    Создание программно:

        channels = ChannelSet(
            name="Линейный набор 750-790 нм, 15 каналов",
            channels=[
                Channel(center_nm=750 + i*2.86, fwhm_nm=25, shape='gauss')
                for i in range(15)
            ],
        )

    Удобные конструкторы:

        ChannelSet.uniform(start=750, stop=790, n=15, fwhm=25)
        ChannelSet.from_centers([750, 760, 765, 770], fwhm=20)
    """
    name: str
    channels: List[Channel]
    reference: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.channels:
            raise ValueError("ChannelSet не может быть пустым")

    # --- доступ ---

    def __len__(self):
        return len(self.channels)

    def __iter__(self):
        return iter(self.channels)

    def __getitem__(self, idx):
        return self.channels[idx]

    @property
    def centers(self) -> np.ndarray:
        return np.array([c.center_nm for c in self.channels])

    @property
    def fwhms(self) -> np.ndarray:
        return np.array([c.fwhm_nm for c in self.channels])

    @property
    def wavelength_range(self):
        """Грубый диапазон, охваченный набором каналов (для setup прибора)."""
        c = self.centers
        f = self.fwhms
        return float(c.min() - f.max()), float(c.max() + f.max())

    # --- удобные конструкторы ---

    @classmethod
    def uniform(cls, start_nm: float, stop_nm: float, n: int,
                 fwhm_nm: float, shape: str = 'gauss',
                 name: Optional[str] = None) -> 'ChannelSet':
        """
        Линейно равномерное распределение центров на [start, stop].

        Удобно для базовой конфигурации прибора и для baseline-эксперимента
        перед оптимизацией.
        """
        if n < 1:
            raise ValueError(f"n должен быть ≥ 1, получено {n}")
        centers = np.linspace(start_nm, stop_nm, n)
        channels = [
            Channel(center_nm=float(c), fwhm_nm=fwhm_nm, shape=shape,
                     name=f'ch_{i:02d}')
            for i, c in enumerate(centers)
        ]
        return cls(
            name=name or f"Uniform {n} channels {start_nm:.0f}-{stop_nm:.0f}нм",
            channels=channels,
        )

    @classmethod
    def from_centers(cls, centers_nm: Sequence[float],
                      fwhm_nm: Union[float, Sequence[float]],
                      shape: str = 'gauss',
                      name: Optional[str] = None) -> 'ChannelSet':
        """
        Произвольный набор центров. fwhm может быть скаляром (одна ширина
        на все каналы) или массивом той же длины, что centers_nm.
        """
        centers = list(centers_nm)
        if np.isscalar(fwhm_nm):
            fwhms = [float(fwhm_nm)] * len(centers)
        else:
            fwhms = list(fwhm_nm)
            if len(fwhms) != len(centers):
                raise ValueError(
                    f"Длина fwhm_nm ({len(fwhms)}) не совпадает "
                    f"с длиной centers_nm ({len(centers)})"
                )
        channels = [
            Channel(center_nm=float(c), fwhm_nm=float(f), shape=shape,
                     name=f'ch_{i:02d}')
            for i, (c, f) in enumerate(zip(centers, fwhms))
        ]
        return cls(
            name=name or f"Custom {len(channels)} channels",
            channels=channels,
        )

    # --- I/O ---

    @classmethod
    def from_file(cls, path) -> 'ChannelSet':
        """Загрузить набор каналов из YAML или JSON файла."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        text = path.read_text(encoding='utf-8')
        ext = path.suffix.lower()
        if ext in ('.yaml', '.yml'):
            from .panels import _load_yaml
            data = _load_yaml(text)
        elif ext == '.json':
            data = json.loads(text)
        else:
            raise ValueError(f"Поддерживаются .yaml/.yml/.json, получено {ext!r}")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChannelSet':
        if 'channels' not in data or not data['channels']:
            raise ValueError("В наборе должны быть каналы (поле 'channels')")
        return cls(
            name=data.get('name', 'Unnamed channel set'),
            channels=[Channel.from_dict(c) for c in data['channels']],
            reference=data.get('reference'),
            notes=data.get('notes'),
        )

    def save(self, path) -> None:
        """Сохранить набор каналов в YAML или JSON."""
        path = Path(path)
        data = self.to_dict()
        ext = path.suffix.lower()
        if ext in ('.yaml', '.yml'):
            from .panels import _dump_yaml
            text = _dump_yaml(data)
        elif ext == '.json':
            text = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Поддерживаются .yaml/.yml/.json, получено {ext!r}")
        path.write_text(text, encoding='utf-8')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'reference': self.reference,
            'notes': self.notes,
            'channels': [c.to_dict() for c in self.channels],
        }

    def __repr__(self):
        return f"ChannelSet({self.name!r}, {len(self.channels)} channels)"


# ---------------------------------------------------------------------------
# ChannelizedSpectrum — результат регистрации прибором
# ---------------------------------------------------------------------------

@dataclass
class ChannelizedSpectrum:
    """
    Спектр после многоканальной регистрации: массив длины N каналов.

    Поля:
        channels : ChannelSet
        values_T : np.ndarray, shape (N,)
            Значение в transmittance-пространстве в каждом канале:
                I_k = ∫ T(λ) · φ_k(λ) dλ
        values_T_true : np.ndarray, shape (N,) — то же без шума (истина)
        source_meta : dict — метаданные исходного Spectrum
    """
    channels: ChannelSet
    values_T: np.ndarray
    values_T_true: np.ndarray
    source_meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def centers(self) -> np.ndarray:
        return self.channels.centers

    @property
    def fwhms(self) -> np.ndarray:
        return self.channels.fwhms

    @property
    def values(self) -> np.ndarray:
        """Алиас для values_T — самое частое использование."""
        return self.values_T

    @property
    def absorbance(self) -> np.ndarray:
        """A = -log10(T)."""
        return -np.log10(np.clip(self.values_T, 1e-12, None))

    @property
    def true_absorbance(self) -> np.ndarray:
        return -np.log10(np.clip(self.values_T_true, 1e-12, None))

    @property
    def optical_depth(self) -> np.ndarray:
        """OD = -ln(T)."""
        return -np.log(np.clip(self.values_T, 1e-12, None))

    @property
    def true_optical_depth(self) -> np.ndarray:
        return -np.log(np.clip(self.values_T_true, 1e-12, None))

    def __len__(self):
        return len(self.values_T)

    def __repr__(self):
        n = len(self)
        rng = (self.centers.min(), self.centers.max())
        return (f"ChannelizedSpectrum({n} каналов, "
                f"{rng[0]:.1f}-{rng[1]:.1f} нм)")

    def plot(self, kind: str = 'transmittance', which: str = 'auto',
              ax=None, figsize=None, title=None, **kwargs):
        """
        Столбчатая диаграмма каналов.

        kind: 'transmittance' | 'absorbance' | 'optical_depth'
        which: 'auto' | 'observed' | 'true' | 'compare'
            'auto' — compare если values_T != values_T_true, иначе true
        """
        from .plotting import _import_mpl
        plt = _import_mpl()

        has_noise = not np.allclose(self.values_T, self.values_T_true)
        if which == 'auto':
            which = 'compare' if has_noise else 'true'

        # Выбор величины
        def _vals_for(kind, which):
            if kind == 'transmittance':
                return self.values_T if which == 'observed' else self.values_T_true
            elif kind == 'absorbance':
                return self.absorbance if which == 'observed' else self.true_absorbance
            elif kind == 'optical_depth':
                return self.optical_depth if which == 'observed' else self.true_optical_depth
            raise ValueError(f"Unknown kind {kind!r}")

        ylabel = {
            'transmittance': 'Transmittance T',
            'absorbance':    r'Absorbance $A = -\log_{10}(T)$',
            'optical_depth': r'Optical depth $\tau = -\ln(T)$',
        }[kind]

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize or (9, 4.5))
        else:
            fig = ax.figure

        x = self.centers
        # Ширина столбца — половина FWHM, чтобы было видно структуру
        width = self.fwhms * 0.5

        if which == 'compare':
            true_v = _vals_for(kind, 'true')
            obs_v = _vals_for(kind, 'observed')
            ax.bar(x, obs_v, width=width, color='C0', alpha=0.6,
                   label='наблюдаемое', edgecolor='C0')
            ax.plot(x, true_v, 'o-', color='C3', lw=1.0, markersize=5,
                    label='истина')
        elif which == 'true':
            ax.bar(x, _vals_for(kind, 'true'), width=width, color='C0',
                   alpha=0.7, edgecolor='C0', label='истина')
        elif which == 'observed':
            ax.bar(x, _vals_for(kind, 'observed'), width=width, color='C0',
                   alpha=0.7, edgecolor='C0', label='наблюдаемое')

        ax.set_xlabel('Центр канала, нм')
        ax.set_ylabel(ylabel)
        ax.set_title(title or f"Channelized: {self.channels.name}")
        ax.grid(True, alpha=0.3)
        if which == 'compare':
            ax.legend()
        fig.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# Главная функция: интегрирование Spectrum по каналам
# ---------------------------------------------------------------------------

def channelize(spec, channels: ChannelSet) -> ChannelizedSpectrum:
    """
    Преобразовать Spectrum (тонкая сетка) в ChannelizedSpectrum
    (массив значений по каналам).

    В каждом канале значение — это интеграл от transmittance
    с весом, равным нормированной форме канала:

        I_k = ∫ T(λ) · φ_k(λ) dλ,   ∫φ_k dλ = 1

    Считается отдельно для истины (true_transmittance) и наблюдаемого
    (transmittance), чтобы можно было оценивать качество регистрации.

    Каналы вне диапазона спектра дают значение 1.0 (полное пропускание),
    предполагая, что вне диапазона нет поглощения. Это приближение —
    в боевом коде нужно следить, чтобы каналы попадали в диапазон.

    Parameters
    ----------
    spec : Spectrum
    channels : ChannelSet

    Returns
    -------
    ChannelizedSpectrum
    """
    wl = spec.wavelength_nm
    T_obs = spec.transmittance
    T_true = spec.true_transmittance

    n_ch = len(channels)
    values_T = np.zeros(n_ch)
    values_T_true = np.zeros(n_ch)

    out_of_range_warned = False

    for k, ch in enumerate(channels):
        # Центр канала вне диапазона спектра?
        if ch.center_nm < wl[0] - 3 * ch.fwhm_nm or \
           ch.center_nm > wl[-1] + 3 * ch.fwhm_nm:
            if not out_of_range_warned:
                import warnings
                warnings.warn(
                    f"Канал {k} ({ch.name or ch.center_nm} нм) находится "
                    f"вне диапазона спектра ({wl[0]:.1f}-{wl[-1]:.1f} нм). "
                    f"Возвращаю T=1.0 для этого канала."
                )
                out_of_range_warned = True
            values_T[k] = 1.0
            values_T_true[k] = 1.0
            continue

        weight = ch.response(wl)
        # ∫ T·φ dλ — численное интегрирование по существующей сетке
        values_T[k] = float(np.trapezoid(T_obs * weight, wl))
        values_T_true[k] = float(np.trapezoid(T_true * weight, wl))

    # Метаданные источника
    source_meta = {}
    if hasattr(spec, 'metadata'):
        try:
            source_meta = dict(spec.metadata)
        except Exception:
            pass

    return ChannelizedSpectrum(
        channels=channels,
        values_T=values_T,
        values_T_true=values_T_true,
        source_meta=source_meta,
    )


# ---------------------------------------------------------------------------
# Утилитарные функции
# ---------------------------------------------------------------------------

def load_channel_set(path) -> ChannelSet:
    """Шорткат: load_channel_set('foo.yaml') == ChannelSet.from_file(...)."""
    return ChannelSet.from_file(path)
