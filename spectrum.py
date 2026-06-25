"""
spectrolib.spectrum
===================
Класс Spectrum с fluent-интерфейсом (метод-чейнинг).

Ключевые инварианты:

1. Внутренние величины — на сетке длин волн (нм). OD аддитивна
   по молекулам, поэтому она и хранится как «рабочая» величина.

2. Чистая истина и шум разделены:
       _clean_optical_depth — всё, что добавлено до шума (молекулы +
                              ILS + дрейф базовой линии в OD-простр.)
       _noise_T            — шумовая добавка в transmittance
       _noise_OD            — шумовая добавка в OD (дрейф)
   Свойство `optical_depth` собирает всё вместе.
   Свойство `true_optical_depth` возвращает только идеал (без шума и дрейфа).
   Это критично для эксперимента 1 диплома (RMSE препроцессинга против истины).

3. Метаданные обновляются на каждом шаге (molecules, history, noise, ils).
   Этого достаточно для воспроизведения спектра по seed.

Пример:

    spec = (Spectrum.from_range(750, 770, step_nm=0.001)
            .add_molecule('O2', c_ppm=210000, L_cm=10, T_K=296, p_atm=1)
            .add_molecule('H2O', c_ppm=10000, L_cm=10)
            .convolve_ils(GaussILS(fwhm=1.0))
            .add_noise_model(NoiseModel(thermal_sigma=0.005,
                                        shot_n_photons_max=1e4),
                             seed=42))

    plt.plot(spec.wavelength_nm, spec.transmittance, label='наблюдаемое')
    plt.plot(spec.wavelength_nm, spec.true_transmittance, label='истина')
"""

from __future__ import annotations

import numpy as np
import hapi

from .physics import nm_to_wavenumber, wavenumber_to_nm, beer_lambert
from .hitran import fetch_molecule, init_db
from .ils import ILS, GaussILS, gauss_convolve  # noqa: F401  (gauss_convolve для обратной совместимости)
from .noise import NoiseModel
from .databases import (
    resolve_source, DB_HITRAN, DB_PNNL, DB_MPI, DB_HITRAN_XSC,
)
from . import pnnl as _pnnl
from . import mpi as _mpi
from . import hitran_xsc as _hxsc


class Spectrum:
    """
    Спектр на равномерной сетке длин волн.

    Внутренние массивы:
        wavelength_nm        — сетка (нм), монотонно возрастает
        wavenumber_cm        — та же сетка в см⁻¹ (монотонно убывает)
        _clean_optical_depth — OD без шума
        _noise_T             — аддитивная добавка к transmittance
        _noise_OD            — аддитивная добавка к OD (дрейф)

    Метаданные:
        molecules : list[dict]
        ils       : dict | None
        noise     : dict | None
        history   : list[str]
        meta      : dict (произвольное)

    Семантика fluent: методы модифицируют объект и возвращают self.
    Для ветвлений — .copy().
    """

    # -----------------------------------------------------------------
    # Создание
    # -----------------------------------------------------------------

    def __init__(self, wavelength_nm, optical_depth=None):
        wl = np.asarray(wavelength_nm, dtype=float)
        if not np.all(np.diff(wl) > 0):
            raise ValueError("wavelength_nm должна монотонно возрастать")
        self.wavelength_nm = wl
        self.wavenumber_cm = nm_to_wavenumber(wl)

        if optical_depth is None:
            self._clean_optical_depth = np.zeros_like(wl)
        else:
            self._clean_optical_depth = np.asarray(optical_depth, dtype=float)

        self._noise_T = np.zeros_like(wl)
        self._noise_OD = np.zeros_like(wl)

        self.molecules = []
        self.ils = None
        self.noise = None
        self.history = []
        self.meta = {}

    @classmethod
    def from_range(cls, wl_min_nm, wl_max_nm, step_nm=None, n_points=None):
        """
        Пустой спектр на равномерной сетке.

        Используй ровно один из: step_nm, n_points.

        Внимание (вариант со step_nm): внутри используется np.arange
        с плавающим шагом. Если (wl_max−wl_min) не кратно step_nm,
        правая граница может слегка «прыгать» из-за ошибок округления
        (число точек ±1 относительно ожидаемого). Если важна точная длина
        и попадание в обе границы — используй n_points.
        """
        if (step_nm is None) == (n_points is None):
            raise ValueError("Задай ровно один из: step_nm или n_points")
        if step_nm is not None:
            wl = np.arange(wl_min_nm, wl_max_nm + step_nm / 2, step_nm)
        else:
            wl = np.linspace(wl_min_nm, wl_max_nm, n_points)
        spec = cls(wl)
        spec.history.append(
            f"from_range({wl_min_nm}, {wl_max_nm}, "
            f"{'step_nm=' + str(step_nm) if step_nm else 'n_points=' + str(n_points)})"
        )
        return spec

    def copy(self):
        """Глубокая копия (для веток вычислений)."""
        new = Spectrum(self.wavelength_nm.copy(),
                       self._clean_optical_depth.copy())
        new._noise_T = self._noise_T.copy()
        new._noise_OD = self._noise_OD.copy()
        new.molecules = [dict(m) for m in self.molecules]
        new.ils = dict(self.ils) if self.ils else None
        new.noise = dict(self.noise) if self.noise else None
        new.history = list(self.history)
        new.meta = dict(self.meta)
        return new

    # -----------------------------------------------------------------
    # Свойства: разные представления
    # -----------------------------------------------------------------

    @property
    def optical_depth(self):
        """OD = чистая_OD + шум_OD − ln(1 + шум_T / T_чистое)."""
        # Чистое пропускание
        T_clean = np.exp(-self._clean_optical_depth)
        # Зашумлённое пропускание (с защитой от <=0)
        T_noisy = np.clip(T_clean + self._noise_T, 1e-12, None)
        return -np.log(T_noisy) + self._noise_OD

    @property
    def transmittance(self):
        """Наблюдаемое пропускание (с шумом)."""
        return np.exp(-self.optical_depth)

    @property
    def absorbance(self):
        """A = OD / ln(10) (наблюдаемая, с шумом)."""
        return self.optical_depth / np.log(10)

    @property
    def true_optical_depth(self):
        """Чистая OD без шума и дрейфа — для оценки качества препроцессинга."""
        return self._clean_optical_depth.copy()

    @property
    def true_transmittance(self):
        """Чистое пропускание без шума."""
        return np.exp(-self._clean_optical_depth)

    @property
    def true_absorbance(self):
        """Чистая absorbance без шума."""
        return self._clean_optical_depth / np.log(10)

    # -----------------------------------------------------------------
    # Метаданные
    # -----------------------------------------------------------------

    @property
    def metadata(self):
        """
        Полный словарь метаданных спектра — для сохранения вместе
        с массивами и для воспроизводимости.

        Внимание: поле 'step_nm' — это np.mean(np.diff(wavelength_nm)).
        Конструктор Spectrum проверяет монотонность, но НЕ равномерность
        сетки; для неравномерных сеток это значение становится
        средним шагом и может ввести в заблуждение. Если ты собрал
        спектр на нестандартной сетке — смотри сами массивы, а не это поле.
        """
        return {
            'wavelength_range_nm': (float(self.wavelength_nm[0]),
                                     float(self.wavelength_nm[-1])),
            'n_points': len(self.wavelength_nm),
            'step_nm': float(np.mean(np.diff(self.wavelength_nm))),
            'molecules': [dict(m) for m in self.molecules],
            'ils': dict(self.ils) if self.ils else None,
            'noise': dict(self.noise) if self.noise else None,
            'history': list(self.history),
            'user_meta': dict(self.meta),
        }

    # -----------------------------------------------------------------
    # Добавление молекулярного поглощения
    # -----------------------------------------------------------------

    def add_molecule(self, name, c_ppm, L_cm,
                     T_K=296, p_atm=1.0,
                     table_name=None, profile='voigt',
                     wing_cm=10.0, diluent=None,
                     step_cm=None,
                     source=None, sources=None,
                     sigma_T_K=None, air_to_vacuum=False,
                     verbose=False):
        """
        Добавляет вклад молекулы в чистую OD спектра.

        Сечение σ [см²/молекула] берётся из одного из трёх источников
        (см. spectrolib.databases): 'hitran' (line-by-line через hapi),
        'pnnl' (ИК-сечения VOC из PNNL IR Database) или 'mpi' (УФ/ВИД из
        MPI-Mainz Spectral Atlas). Источник выбирается через resolve_source:
        явный source= > локальная карта sources= > реестр MOLECULE_SOURCE >
        дефолт 'hitran'. Независимо от источника дальше всё одинаково:
        OD = σ·N(T,p)·L и интерполяция на сетку спектра — это гарантирует
        согласованность баз (общая единица σ, общая формула OD).

        Parameters
        ----------
        name : str
            Имя молекулы. Для HITRAN — см. MOLECULE_IDS; для PNNL/MPI — как
            зарегистрировано в соответствующем загрузчике.
        c_ppm : float
            Концентрация в ppm.
        L_cm : float
            Длина пути, см.
        T_K, p_atm : float
            Температура (K) и давление (атм). В законе Бугера–Ламберта–Бера
            числовая плотность N(T,p) считается ВСЕГДА по этим значениям.
            Для HITRAN T_K дополнительно влияет на форму/силу линий внутри
            hapi (пересчёт от 296 K через Q(T)).
        table_name : str, optional
            (HITRAN) имя локальной таблицы; None → fetch_molecule подберёт.
        profile : {'voigt', 'lorentz', 'gauss'}
            (HITRAN) профиль линии.
        wing_cm : float
            (HITRAN) запас по краям, см⁻¹.
        diluent : dict, optional
            (HITRAN) состав уширяющей среды; None → {'air': 1.0}.
        step_cm : float, optional
            (HITRAN) явный шаг сетки hapi, см⁻¹; None → авто [0.001, 0.01].
        source : {'hitran', 'pnnl', 'mpi'}, optional
            Явный источник сечений (высший приоритет).
        sources : dict, optional
            Локальная карта {имя: источник} (например, из GasMixture.sources).
        sigma_T_K : float, optional
            (PNNL) температура, к которой интерполируется ФОРМА σ по сетке.
            По умолчанию строго 310 K (температура выдоха) — требование
            диплома. Числовая плотность при этом всё равно берётся по T_K.
        air_to_vacuum : bool
            (MPI) переводить λ воздух→вакуум перед расчётом ν.
        verbose : bool
            Диагностический вывод.
        """
        src = resolve_source(name, source=source, overrides=sources)

        if src == DB_HITRAN:
            nu_grid, sigma, extra_meta = self._sigma_hitran(
                name, T_K=T_K, p_atm=p_atm, table_name=table_name,
                profile=profile, wing_cm=wing_cm, diluent=diluent,
                step_cm=step_cm, verbose=verbose,
            )
        elif src == DB_PNNL:
            # Форма σ интерполируется по T строго до 310 K (или sigma_T_K).
            T_shape = _pnnl.T_EXHALE_K if sigma_T_K is None else float(sigma_T_K)
            nu_grid, sigma, m = _pnnl.load_pnnl_sigma(
                name, T_target=T_shape, verbose=verbose,
            )
            extra_meta = {'pnnl': m, 'profile': None, 'table_name': None,
                          'diluent': None}
        elif src == DB_MPI:
            T_shape = T_K if sigma_T_K is None else float(sigma_T_K)
            nu_grid, sigma, m = _mpi.load_mpi_sigma(
                name, T_target=T_shape, air_to_vacuum=air_to_vacuum,
                verbose=verbose,
            )
            extra_meta = {'mpi': m, 'profile': None, 'table_name': None,
                          'diluent': None}
        elif src == DB_HITRAN_XSC:
            # Sharpe/PNNL VOC через HITRAN xsc. T-интерполяция на 310 K
            # (или sigma_T_K), как и для родного PNNL.
            T_shape = _hxsc.T_EXHALE_K if sigma_T_K is None else float(sigma_T_K)
            nu_grid, sigma, m = _hxsc.load_xsc_sigma(
                name, T_target=T_shape, verbose=verbose,
            )
            extra_meta = {'hitran_xsc': m, 'profile': None,
                          'table_name': None, 'diluent': None}
        else:  # pragma: no cover — resolve_source гарантирует допустимость
            raise ValueError(f"Неизвестный источник '{src}'")

        self._accumulate_sigma(nu_grid, sigma, c_ppm=c_ppm, L_cm=L_cm,
                               T_K=T_K, p_atm=p_atm)

        record = {
            'name': name, 'c_ppm': c_ppm, 'L_cm': L_cm,
            'T_K': T_K, 'p_atm': p_atm, 'source': src,
            'profile': extra_meta.get('profile'),
            'table_name': extra_meta.get('table_name'),
            'diluent': extra_meta.get('diluent'),
        }
        if 'pnnl' in extra_meta:
            record['pnnl'] = extra_meta['pnnl']
        if 'mpi' in extra_meta:
            record['mpi'] = extra_meta['mpi']
        if 'hitran_xsc' in extra_meta:
            record['hitran_xsc'] = extra_meta['hitran_xsc']
        self.molecules.append(record)

        self.history.append(
            f"add_molecule({name}, c={c_ppm} ppm, L={L_cm} cm, "
            f"T={T_K} K, p={p_atm} atm, source={src})"
        )
        return self

    # -----------------------------------------------------------------
    # Получение σ из источников и общий хвост σ→OD→сетка
    # -----------------------------------------------------------------

    def _sigma_hitran(self, name, T_K, p_atm, table_name, profile,
                      wing_cm, diluent, step_cm, verbose):
        """
        Рассчитать σ(ν) молекулы через hapi (line-by-line).

        Returns
        -------
        nu_grid : np.ndarray   (см⁻¹, по возрастанию)
        sigma : np.ndarray     (см²/молекула)
        meta : dict            (profile, table_name, diluent)
        """
        init_db()
        wl_min = float(self.wavelength_nm.min())
        wl_max = float(self.wavelength_nm.max())

        if table_name is None:
            table_name = fetch_molecule(name, wl_min_nm=wl_min,
                                        wl_max_nm=wl_max)

        nu_min = nm_to_wavenumber(wl_max)   # меньшее ν
        nu_max = nm_to_wavenumber(wl_min)   # большее ν

        # Шаг сетки hapi: min(d_nu_native/5, 0.01), пол 0.001 см⁻¹.
        if step_cm is None:
            d_nu_native = float(np.abs(np.mean(np.diff(self.wavenumber_cm))))
            step_cm_use = max(0.001, min(0.01, d_nu_native / 5))
        else:
            step_cm_use = float(step_cm)

        profile_func = {
            'voigt':   hapi.absorptionCoefficient_Voigt,
            'lorentz': hapi.absorptionCoefficient_Lorentz,
            'gauss':   hapi.absorptionCoefficient_Doppler,
        }[profile]

        if diluent is None:
            diluent = {'air': 1.0}

        # HITRAN_units=True → σ в см²/молекула. verbose=False глушит вывод hapi.
        import io as _io
        import contextlib as _ctx
        _sink = _io.StringIO() if not verbose else None
        _ctx_mgr = _ctx.redirect_stdout(_sink) if not verbose else _ctx.nullcontext()
        with _ctx_mgr:
            nu_grid, sigma = profile_func(
                SourceTables=table_name,
                Environment={'T': T_K, 'p': p_atm},
                WavenumberRange=[nu_min, nu_max],
                WavenumberStep=step_cm_use,
                WavenumberWing=wing_cm,
                Diluent=diluent,
                HITRAN_units=True,
            )
        return nu_grid, sigma, {'profile': profile, 'table_name': table_name,
                                'diluent': dict(diluent)}

    def _accumulate_sigma(self, nu_grid, sigma, c_ppm, L_cm, T_K, p_atm):
        """
        Общий хвост для всех источников: σ → OD (закон БЛБ) → интерполяция
        на сетку спектра → накопление в _clean_optical_depth.

        nu_grid должна быть по возрастанию. За пределами диапазона данных
        вклад зануляется (left=0, right=0) — это важно для PNNL/MPI, чьи
        измеренные диапазоны уже сетки спектра: нельзя «размазывать» краевое
        σ на всю сетку.
        """
        nu_grid = np.asarray(nu_grid, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        # OD = σ · N_target · L,  N_target = (c_ppm·1e-6) · N_total(T, p)
        od_native = beer_lambert(sigma, c_ppm=c_ppm, L_cm=L_cm,
                                  T_K=T_K, p_atm=p_atm)

        od_on_our_grid = np.interp(
            self.wavenumber_cm[::-1], nu_grid, od_native,
            left=0.0, right=0.0,
        )[::-1]

        self._clean_optical_depth = self._clean_optical_depth + od_on_our_grid

    # -----------------------------------------------------------------
    # Аппаратная функция
    # -----------------------------------------------------------------

    def convolve_ils(self, ils):
        """
        Свёртка с произвольной ILS (Gauss / Lorentz / Voigt / FromFile).

        Применяется к **чистой** OD (до шума). Это правильный порядок:
        ILS — характеристика прибора, она действует на физический
        спектр перед тем, как сигнал попадёт на детектор и наберёт шум.

        Parameters
        ----------
        ils : ILS
            Объект ILS (см. spectrolib.ils).
        """
        if not isinstance(ils, ILS):
            raise TypeError(f"Ожидался объект ILS, получено {type(ils)}")
        self._clean_optical_depth = ils.convolve(
            self._clean_optical_depth, self.wavelength_nm
        )
        self.ils = {'type': type(ils).__name__, 'repr': repr(ils),
                    'fwhm': float(ils.fwhm)}
        self.history.append(f"convolve_ils({ils!r})")
        return self

    def convolve_gauss(self, fwhm_nm):
        """
        Свёртка с гауссовой ILS — шорткат для частого случая.
        Эквивалент: convolve_ils(GaussILS(fwhm_nm)).
        """
        return self.convolve_ils(GaussILS(fwhm_nm))

    # -----------------------------------------------------------------
    # Шумы
    # -----------------------------------------------------------------

    def add_noise_model(self, model: NoiseModel, seed=None):
        """
        Применить полную модель шума (NoiseModel из spectrolib.noise).

        Шум **добавляется** к существующему (если уже был добавлен через
        предыдущий вызов — добавки суммируются). Это полезно для проверки
        робастности препроцессинга к разным режимам шума, но если тебе
        нужен «чистый» режим — вызови .reset_noise() перед этим.

        Parameters
        ----------
        model : NoiseModel
        seed : int, optional
            Для воспроизводимости.
        """
        rng = np.random.default_rng(seed)
        T_clean = self.true_transmittance
        delta_T, delta_OD, contributions = model.apply(
            T_clean, self.wavelength_nm, rng
        )
        self._noise_T = self._noise_T + delta_T
        self._noise_OD = self._noise_OD + delta_OD

        self.noise = model.to_dict()
        self.noise['seed'] = seed
        self.history.append(
            f"add_noise_model({model!r}, seed={seed})"
        )
        return self

    def add_noise(self, snr=None, sigma=None, kind='gauss', seed=None):
        """
        Простой гауссов шум — обёртка над add_noise_model.
        Сохранена для обратной совместимости.

        Parameters
        ----------
        snr : float, optional
            Signal-to-noise. σ = 1/snr (относительно T=1).
        sigma : float, optional
            СКО шума напрямую.
        """
        if (snr is None) == (sigma is None):
            raise ValueError("Задай ровно один из: snr или sigma")
        if sigma is None:
            sigma = 1.0 / snr
        if kind != 'gauss':
            raise ValueError(
                f"add_noise поддерживает только kind='gauss'. "
                f"Для других видов используй add_noise_model."
            )
        return self.add_noise_model(NoiseModel(thermal_sigma=sigma), seed=seed)

    def reset_noise(self):
        """Обнулить шумовые добавки, сохранив чистую OD."""
        self._noise_T = np.zeros_like(self.wavelength_nm)
        self._noise_OD = np.zeros_like(self.wavelength_nm)
        self.noise = None
        self.history.append("reset_noise()")
        return self

    # -----------------------------------------------------------------
    # Полиномиальный дрейф (отдельно от NoiseModel — для совместимости
    # и потому что иногда нужен детерминированный наклон)
    # -----------------------------------------------------------------

    def add_baseline(self, slope=0.0, offset=0.0, curvature=0.0):
        """
        Детерминированный полиномиальный дрейф базовой линии в OD.

        baseline(λ) = offset + slope·(λ−λ₀) + curvature·(λ−λ₀)²,
        где λ₀ — центр диапазона.

        В отличие от NoiseModel.drift_amplitude (который случайный),
        этот дрейф детерминированный и идёт в **чистую** OD — то есть
        считается частью «истинной» картины (например, известный наклон
        отклика прибора).
        """
        wl0 = (self.wavelength_nm[0] + self.wavelength_nm[-1]) / 2
        dx = self.wavelength_nm - wl0
        baseline = offset + slope * dx + curvature * dx ** 2
        self._clean_optical_depth = self._clean_optical_depth + baseline
        self.history.append(
            f"add_baseline(slope={slope}, offset={offset}, curvature={curvature})"
        )
        return self

    # -----------------------------------------------------------------
    # Тестовые / отладочные пики
    # -----------------------------------------------------------------

    def add_gauss_peak(self, center_nm, fwhm_nm, amplitude,
                       in_what='optical_depth'):
        """
        Добавить аналитический гауссов пик. Полезно для тестов и
        проверки ILS на известном профиле.

        Идёт в чистую OD (или в чистое T → пересчёт в OD).
        """
        sigma = fwhm_nm / (2 * np.sqrt(2 * np.log(2)))
        g = amplitude * np.exp(
            -(self.wavelength_nm - center_nm) ** 2 / (2 * sigma ** 2)
        )
        if in_what == 'optical_depth':
            self._clean_optical_depth = self._clean_optical_depth + g
        elif in_what == 'transmittance':
            T_old = self.true_transmittance
            T_new = np.clip(T_old + g, 1e-12, None)
            self._clean_optical_depth = -np.log(T_new)
        else:
            raise ValueError(
                f"in_what должен быть 'optical_depth' или 'transmittance', "
                f"получено {in_what!r}"
            )
        self.history.append(
            f"add_gauss_peak(center_nm={center_nm}, fwhm_nm={fwhm_nm}, "
            f"amplitude={amplitude}, in_what={in_what!r})"
        )
        return self

    # -----------------------------------------------------------------
    # Утилиты
    # -----------------------------------------------------------------

    def reset(self):
        """
        Сброс OD/шумов и связанных метаданных молекул, ILS и шума.

        Что обнуляется:
            _clean_optical_depth, _noise_T, _noise_OD, molecules, ils, noise.

        Что **не** обнуляется (сохраняется как есть):
            - wavelength_nm / wavenumber_cm (сетка)
            - history — в неё дописывается строка "reset()", прошлые
              записи сохраняются (журнал операций, а не текущее состояние)
            - meta — произвольный пользовательский словарь

        Если нужен полностью «голый» спектр, проще создать новый через
        Spectrum.from_range(...).
        """
        self._clean_optical_depth = np.zeros_like(self.wavelength_nm)
        self._noise_T = np.zeros_like(self.wavelength_nm)
        self._noise_OD = np.zeros_like(self.wavelength_nm)
        self.molecules = []
        self.ils = None
        self.noise = None
        self.history.append("reset()")
        return self

    # -----------------------------------------------------------------
    # Многоканальная регистрация
    # -----------------------------------------------------------------

    def to_channels(self, channels):
        """
        Имитация регистрации спектра прибором с дискретными каналами.

        Возвращает ChannelizedSpectrum: массив значений по каналам
        (вместо тысяч точек тонкой сетки — N интегралов по полосам
        каналов прибора).

        Parameters
        ----------
        channels : ChannelSet
            Набор каналов прибора (см. spectrolib.channels.ChannelSet).

        Returns
        -------
        ChannelizedSpectrum

        Notes
        -----
        Это и есть «как видит твой реальный прибор с OLED/QD-пикселями»,
        в отличие от тонкой сетки, имитирующей FTIR высокого разрешения.
        Используй этот шаг последним в пайплайне (после ILS и шума).
        """
        from .channels import channelize
        return channelize(self, channels)

    # -----------------------------------------------------------------
    # Визуализация
    # -----------------------------------------------------------------

    def plot(self, kind='transmittance', which='auto', ax=None,
             figsize=None, title=None, show_legend=True, **kwargs):
        """
        Построить график спектра одной командой.

        Parameters
        ----------
        kind : {'transmittance', 'absorbance', 'optical_depth'}
        which : {'auto', 'observed', 'true', 'compare'}
            'auto' (дефолт): если есть шум — рисует обе кривые
            (истина пунктиром, наблюдаемое сплошным), иначе только истину.
        ax : matplotlib.axes.Axes, optional
        figsize : tuple, optional
        title : str, optional
            Если None — собирается автоматически из метаданных
            (молекулы, T, p, L).
        **kwargs : передаются в matplotlib для основной линии.

        Returns
        -------
        (fig, ax)
        """
        from .plotting import plot_spectrum
        return plot_spectrum(self, kind=kind, which=which, ax=ax,
                              figsize=figsize, title=title,
                              show_legend=show_legend, **kwargs)

    def plot_clean_vs_noisy(self, kind='transmittance', figsize=None):
        """
        Двухпанельный график: сверху — истина и наблюдаемое,
        снизу — разница (с RMS-аннотацией).

        Полезно для оценки шума и эталонной величины RMSE препроцессинга.
        """
        from .plotting import plot_clean_vs_noisy
        return plot_clean_vs_noisy(self, kind=kind, figsize=figsize)

    def save(self, filepath, kind='transmittance', fmt=None, include_true=True):
        """
        Сохранить спектр в файл.

        Формат определяется по расширению (.csv/.txt — CSV с шапкой;
        .npz — бинарный архив со всеми массивами). См. io.save_spectrum.
        """
        from .io import save_spectrum
        return save_spectrum(self, filepath, kind=kind, fmt=fmt,
                             include_true=include_true)

    def __repr__(self):
        wl0, wl1 = self.wavelength_nm[0], self.wavelength_nm[-1]
        n = len(self.wavelength_nm)
        mols = ', '.join(m['name'] for m in self.molecules) or '-'
        od_max = self._clean_optical_depth.max()
        ils = self.ils['type'] if self.ils else '-'
        noise = 'yes' if self.noise else 'no'
        return (f"Spectrum({wl0:.1f}–{wl1:.1f} нм, {n} точек, "
                f"молекулы: {mols}, OD_max={od_max:.4g}, "
                f"ILS={ils}, noise={noise})")
