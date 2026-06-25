"""
spectrolib.audit
================
Покомпонентный информационный аудит спектральных каналов для проектирования
QD-платформы. Реализация алгоритма из главы 4 диплома.

Кратко:
    Для одного целевого биомаркера m* (или группы перекрытых биомаркеров T_r)
    и заданного спектрального блока подбирается локальная конфигурация
    каналов K* — список центральных длин волн квантовых точек, который
    максимизирует J*_block = min_m J*_m (E-оптимальность в J*-метрике).

Опорные формулы:
    Маргинальная Фишера:  I_{c|d} = N * A^T C_n^{-1} P_B^perp A
    P_B^perp = I - B (B^T C_n^{-1} B)^{-1} B^T C_n^{-1}    (косой проектор)
    J*_m = delta_m^2 * sigma_b_m^2 / sigma_i_m^2
    sigma_i_m^2 = [I_{c|d}^{-1}]_{mm}

Жадная процедура: на каждом шаге добавляется канал, дающий максимальное
уменьшение Psi(K) = max_m 1/J*_m(K); останов — относительный выигрыш
ниже epsilon_stop в течение двух шагов подряд.

Пример использования (минимальный):

    from spectrolib import (
        MixturePanel, NoiseModel, ChannelSet, Instrument, SpectrumGenerator,
        GasMixture, preconcentrate, GaussILS,
    )
    from spectrolib.audit import LocalBlock, audit_block

    panel0 = MixturePanel.from_file('example_panels/lung_cancer_control_mu0.yaml')
    panel1 = MixturePanel.from_file('example_panels/lung_cancer_disease_mu1.yaml')
    noise = NoiseModel.from_file('example_noise_models/table_2_6.yaml')

    block = LocalBlock(
        name='HCHO_UV',
        wavelength_range_nm=(260.0, 350.0),
        targets=['CH2O'],
        interferents=['acetone', 'isoprene', 'benzene', 'toluene'],
        cv_table={'CH2O': 0.4, 'acetone': 0.3, 'isoprene': 0.3,
                  'benzene': 0.5, 'toluene': 0.5},
        fwhm_nm=25.0,
        kappa=200.0,
    )
    result = audit_block(block, panel0, panel1, noise)
    print(result.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Iterable
import numpy as np
from scipy import stats

from .channels import Channel, ChannelSet, channelize
from .noise import NoiseModel
from .panels import MixturePanel
from .api import Instrument, GasMixture, SpectrumGenerator
from .ils import GaussILS
from .protocol import preconcentrate


# ============================================================================
# Расчётные концентрации (design point)
# ============================================================================

def lognormal_percentile(c_median: float, cv: float, percentile: float) -> float:
    """
    Процентиль логнормального распределения с медианой c_median и
    коэффициентом вариации cv.

    c_p = c_median * exp(sigma_log * z_p),
    sigma_log = sqrt(ln(1 + CV^2)),
    z_p = квантиль стандартного нормального.

    Parameters
    ----------
    c_median : float
        Медиана концентрации (любые единицы).
    cv : float
        Коэффициент вариации, CV > 0.
    percentile : float
        Процентиль в [0, 1]. 0.5 — медиана; 0.30 — нижний типичный.
    """
    if cv <= 0:
        raise ValueError(f"CV must be > 0, got {cv}")
    if not 0 < percentile < 1:
        raise ValueError(f"percentile must be in (0, 1), got {percentile}")
    sigma_log = np.sqrt(np.log(1 + cv ** 2))
    z = stats.norm.ppf(percentile)
    return float(c_median * np.exp(sigma_log * z))


def design_concentrations(
    targets: Sequence[str],
    interferents: Sequence[str],
    panel0: MixturePanel,
    panel1: MixturePanel,
    cv_table: Dict[str, float],
    p_lo: float = 0.30,
) -> Dict[str, float]:
    """
    Расчётные концентрации для аудита (раздел 4.1.2 диплома).

    Для целевых биомаркеров берётся MIN по p_lo-процентилю обеих субпопуляций
    (наихудший случай для дробового шума). Для интерферентов — медиана.

    Returns
    -------
    dict
        Маппинг имя_молекулы → c_design в ppm.
    """
    by_name_0 = {b.name: b for b in panel0.biomarkers}
    by_name_1 = {b.name: b for b in panel1.biomarkers}

    c_design: Dict[str, float] = {}

    for m in targets:
        if m not in cv_table:
            raise KeyError(f"CV для {m} не задан в cv_table")
        if m not in by_name_0 or m not in by_name_1:
            raise KeyError(f"{m} отсутствует в одной из панелей")
        c0 = by_name_0[m].c_ppm
        c1 = by_name_1[m].c_ppm
        c0_p = lognormal_percentile(c0, cv_table[m], p_lo)
        c1_p = lognormal_percentile(c1, cv_table[m], p_lo)
        c_design[m] = min(c0_p, c1_p)

    for m in interferents:
        if m not in by_name_0:
            raise KeyError(f"{m} отсутствует в panel0")
        # Медиана = c_ppm из панели (она уже задана как μ)
        c_design[m] = by_name_0[m].c_ppm

    return c_design


# ============================================================================
# Клинический порог J*_min из (tau*, epsilon, M)
# ============================================================================

def compute_J_star_min(
    tau_star: float = 0.60,
    epsilon: float = 0.05,
    M: int = 11,
) -> float:
    """
    Клинический порог информационного показателя из eq:Jstar-min.

    Формула:
        J*_min = (2 * 1.96 * f_chi2_M(D*) * D* / epsilon)^2,
        D*^2 = chi2_M.ppf(1 - tau*).

    Parameters
    ----------
    tau_star : float, default 0.60
        Клинически выбранный порог типичности.
    epsilon : float, default 0.05
        Допустимая полуширина 95% CI типичности в окрестности tau*.
    M : int, default 11
        Размерность вектора целевых биомаркеров (число молекул в направлении
        различия). Для одиночного аудита (Case A/B) M=1 даёт J*_min ≈ 744.

    Returns
    -------
    float
        Численное значение порога.
    """
    if not 0 < tau_star < 1:
        raise ValueError(f"tau_star must be in (0, 1), got {tau_star}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    D_star_sq = float(stats.chi2(df=M).ppf(1.0 - tau_star))
    D_star = np.sqrt(D_star_sq)
    f_chi2 = float(stats.chi2(df=M).pdf(D_star_sq))
    return float((2.0 * 1.96 * f_chi2 * D_star / epsilon) ** 2)


# ============================================================================
# Авто-расчёт kappa из условия линейности (раздел 4.2 диплома)
# ============================================================================

def compute_kappa_max(
    wavelength_range_nm: Tuple[float, float],
    molecules: Sequence[str],
    panel0: MixturePanel,
    panel1: MixturePanel,
    cv_table: Dict[str, float],
    L_cm: float = 15.0,
    tau_max_lin: float = 0.3,
    p_hi: float = 0.95,
    grid_sampling_nm: float = 0.5,
    margin_nm: float = 50.0,
    sources_override: Optional[Dict[str, List[str]]] = None,
) -> Tuple[float, float, str]:
    """
    Максимально допустимое κ из условия линейности (4.2 диплома).

    Шаги:
      1. Для каждой молекулы: c_lin = max по верхнему процентилю обеих
         субпопуляций.
      2. Генерируется тонкосетный спектр с этими концентрациями и κ=1.
      3. τ_peak = max τ(λ) по всему рабочему диапазону.
      4. κ_max = τ_max_lin / τ_peak_at_κ_1.

    Returns
    -------
    kappa_max, tau_peak_at_kappa_1, dominant_molecule
        Дробное κ_max, пиковое τ при κ=1, имя доминирующей молекулы.
    """
    by0 = {b.name: b.c_ppm for b in panel0.biomarkers}
    by1 = {b.name: b.c_ppm for b in panel1.biomarkers}
    c_hi: Dict[str, float] = {}
    for m in molecules:
        if m not in by0 or m not in by1 or m not in cv_table:
            # Молекула без полной информации — берём среднее
            c_hi[m] = max(by0.get(m, 0.0), by1.get(m, 0.0))
            continue
        c_hi[m] = max(
            lognormal_percentile(by0[m], cv_table[m], p_hi),
            lognormal_percentile(by1[m], cv_table[m], p_hi),
        )

    cache = _build_block_cache(
        wavelength_range_nm=wavelength_range_nm,
        molecules=molecules,
        panel_template=panel0,
        L_cm=L_cm,
        grid_sampling_nm=grid_sampling_nm,
        margin_nm=margin_nm,
        sources_override=sources_override,
    )

    # Полный τ(λ) на тонкой сетке при κ=1
    grid = cache.wavelength_nm
    tau = np.zeros_like(grid)
    contrib_max: Dict[str, float] = {}
    for m in molecules:
        od_m = cache.od_per_unit_c.get(m)
        if od_m is None:
            continue
        contrib = c_hi[m] * od_m
        tau += contrib
        contrib_max[m] = float(contrib.max())

    # Ограничиваем рабочим диапазоном (без margin)
    lo, hi = wavelength_range_nm
    mask = (grid >= lo) & (grid <= hi)
    if not mask.any():
        mask = np.ones_like(grid, dtype=bool)
    tau_peak = float(tau[mask].max())
    dominant = max(contrib_max, key=contrib_max.get) if contrib_max else ''
    kappa_max = float(tau_max_lin / max(tau_peak, 1e-30))
    return kappa_max, tau_peak, dominant


def round_kappa_floor(
    kappa_max: float, kappa_tech_cap: float = 1000.0,
) -> float:
    """
    Округление вниз по технологической шкале (раздел 4.2 диплома).

    Технологический потолок сорбционного преконцентрирования
    (Tenax/Carbopack, Lawal 2017): κ ≤ 10^3. Применяется как min(κ_max, cap).
    Шкала ступеней: {10, 20, 50, 100, 200, 500, 1000}.
    """
    scale = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    effective_max = min(float(kappa_max), float(kappa_tech_cap))
    fit = [s for s in scale if s <= effective_max]
    if not fit:
        return 1.0
    return float(fit[-1])


# ============================================================================
# Локальный блок и результат
# ============================================================================

@dataclass
class LocalBlock:
    """
    Описание локального блока для аудита (раздел 4.1 диплома).

    Parameters
    ----------
    name : str
        Имя блока для отчётов (например, "HCHO_UV").
    wavelength_range_nm : (float, float)
        Спектральные границы блока G_r.
    targets : list[str]
        Целевые биомаркеры T_r (имена для HITRAN/MOLECULE_IDS).
        Один элемент — изолированный блок (Case A/B), несколько — перекрытый
        блок (Case C) с E-оптимальным критерием.
    interferents : list[str]
        Локально перекрывающиеся интерференты по правилу eq:audit-eta-rule.
        Включает H2O, CO2 для БИК-блоков по необходимости.
    cv_table : dict[str, float]
        CV логнормали для всех молекул из targets ∪ interferents.
        Берётся из таблицы 2.2 практической части.
    fwhm_nm : float
        FWHM канала в поддиапазоне (25 / 30 / 150 для УФ/видимый/БИК).
    L_cm : float, default 10.0
        Длина оптического пути кюветы.
    kappa : float, default 100.0
        Коэффициент предконцентрирования; должен быть определён глобально
        по разделу 4.2.
    N_breaths : int, default 3
        Расчётное число выдохов (эксплуатация: 5).
    drift_degree : int, default 2
        Степень полиномиального базиса дрейфа (B^(r)); dim = degree + 1.
    grid_step_factor : float, default 0.10
        Шаг сетки кандидатных центров: delta_lambda = fwhm * grid_step_factor.
    min_dist_factor : float, default 0.5
        Минимальное расстояние между центрами: fwhm * min_dist_factor.
    p_lo : float, default 0.30
        Нижний процентиль для design-концентрации целевых биомаркеров.
    """
    name: str
    wavelength_range_nm: Tuple[float, float]
    targets: List[str]
    interferents: List[str]
    cv_table: Dict[str, float]
    fwhm_nm: float
    L_cm: float = 15.0
    kappa: float = 100.0
    N_breaths: int = 3
    drift_degree: int = 2
    grid_step_factor: float = 0.10
    min_dist_factor: float = 0.30
    p_lo: float = 0.30
    sources_override: Optional[Dict[str, List[str]]] = None
    """
    Если задан, перекрывает поле `sources` молекул панели при построении
    кэша спектров. Используется, чтобы выбрать только MPI для УФ-блоков
    или только HITRAN/XSC для ИК-блоков, избегая лишних расчётов.

    Пример:
        sources_override={'HCHO': ['mpi'], 'acetone': ['mpi']}
    """
    kappa_per_molecule: Optional[Dict[str, float]] = None
    """
    Если задан, перекрывает глобальный `kappa` для отдельных молекул.
    Физически: селективное сорбционное преконцентрирование удерживает
    VOC и пропускает H2O/CO2/O2 (по сути κ для них ≈ 1).

    Пример (БИК-блок):
        kappa=1000, kappa_per_molecule={'H2O': 1.0, 'CO2': 1.0, 'O2': 1.0}

    Не указанные молекулы используют глобальный `kappa`.
    """

    def kappa_of(self, m: str) -> float:
        """Эффективное κ для молекулы m."""
        if self.kappa_per_molecule and m in self.kappa_per_molecule:
            return float(self.kappa_per_molecule[m])
        return float(self.kappa)

    @property
    def molecules(self) -> List[str]:
        return list(self.targets) + list(self.interferents)

    @property
    def grid_step_nm(self) -> float:
        return self.fwhm_nm * self.grid_step_factor

    @property
    def min_dist_nm(self) -> float:
        return self.fwhm_nm * self.min_dist_factor

    @property
    def drift_dim(self) -> int:
        return self.drift_degree + 1

    def candidate_grid(self) -> np.ndarray:
        """Дискретная сетка допустимых центров каналов Λ_r."""
        lo, hi = self.wavelength_range_nm
        step = self.grid_step_nm
        return np.arange(lo, hi + step / 2, step)


@dataclass
class AuditResult:
    """
    Результат покомпонентного аудита блока (раздел 4.5 диплома).

    Поля
    ----
    block : LocalBlock
    lambdas_star : np.ndarray
        Центры выбранных каналов K* (в порядке добавления).
    K_star : int
        Количество каналов.
    sigma_i : dict[str, float]
        Инструментальное СКО оценки для каждого целевого биомаркера, ppm.
    J_star_m : dict[str, float]
        Покомпонентный показатель информации для каждого целевого биомаркера.
    delta_rel : dict[str, float]
        Относительная точность на c^design в %, для каждого целевого биомаркера.
    verdict : dict[str, str]
        PASS+ / PASS / FAIL для каждого целевого биомаркера.
    J_star_block : float
        min_m J*_m — E-оптимальный показатель блока.
    history_psi : list[float]
        Значения Psi = max_m 1/J*_m по шагам жадной процедуры.
    history_lambda : list[float]
        Добавленные центры по шагам.
    history_gain : list[float]
        Относительный выигрыш g^(t) по шагам.
    c_design : dict[str, float]
        Расчётные концентрации, использованные в аудите.
    J_star_min : float
        Порог, использованный для классификации.
    """
    block: LocalBlock
    lambdas_star: np.ndarray
    K_star: int
    sigma_i: Dict[str, float]
    J_star_m: Dict[str, float]
    delta_rel: Dict[str, float]
    verdict: Dict[str, str]
    J_star_block: float
    history_psi: List[float]
    history_lambda: List[float]
    history_gain: List[float]
    c_design: Dict[str, float]
    J_star_min: float

    def summary(self) -> str:
        """Человеко-читаемый отчёт."""
        lines = [
            f"=== Audit: {self.block.name} ===",
            f"K* = {self.K_star} каналов в [{self.block.wavelength_range_nm[0]:.0f},"
            f" {self.block.wavelength_range_nm[1]:.0f}] нм, FWHM={self.block.fwhm_nm:.0f} нм",
            f"kappa = {self.block.kappa:.0f}, N = {self.block.N_breaths}",
            f"J*_block = {self.J_star_block:.3f} (порог {self.J_star_min:.3f})",
            "",
            f"{'Биомаркер':<14} {'sigma_i, ppm':>12} {'delta_rel, %':>13} "
            f"{'J*_m':>8} {'вердикт':>9}",
        ]
        for m in self.block.targets:
            lines.append(
                f"{m:<14} {self.sigma_i[m]:>12.4e} {self.delta_rel[m]*100:>13.2f} "
                f"{self.J_star_m[m]:>8.3f} {self.verdict[m]:>9}"
            )
        lines.append("")
        lines.append(f"История относительного выигрыша:")
        for t, (lam, gain) in enumerate(
            zip(self.history_lambda, self.history_gain), 1
        ):
            lines.append(f"  шаг {t:2d}: lambda={lam:.1f} нм, g={gain*100:.2f}%")
        return "\n".join(lines)


# ============================================================================
# Построение матрицы отклика A^(r)
# ============================================================================

def _make_channel_set(centers_nm: Iterable[float], fwhm_nm: float) -> ChannelSet:
    """Утилитарный конструктор ChannelSet из списка центров."""
    centers = list(centers_nm)
    if len(centers) == 0:
        raise ValueError("Пустой набор центров")
    return ChannelSet(
        name=f"audit_{len(centers)}ch_fwhm{fwhm_nm:.0f}",
        channels=[
            Channel(center_nm=float(c), fwhm_nm=fwhm_nm, shape='gauss',
                    name=f"ch_{i:02d}")
            for i, c in enumerate(sorted(centers))
        ],
    )


@dataclass
class _BlockSpectraCache:
    """
    Кэш тонкосетных OD-спектров по молекулам для одного блока.

    Идея: спектры HITRAN/MPI/XSC для каждой молекулы пересчитываются один
    раз на единичной концентрации, после чего любая конфигурация каналов
    интегрируется быстро.
    """
    wavelength_nm: np.ndarray
    od_per_unit_c: Dict[str, np.ndarray]  # OD/ppm для unit концентрации
    panel_template: MixturePanel


# Модуль-уровневый кэш тонкосетных спектров. Используется чтобы не пересчитывать
# HITRAN/MPI/XSC при повторных вызовах audit_block с теми же молекулами/диапазоном.
_GLOBAL_SPECTRA_CACHE: Dict[Tuple, '_BlockSpectraCache'] = {}


def clear_audit_cache() -> None:
    """Очистить модуль-уровневый кэш тонкосетных спектров аудита."""
    _GLOBAL_SPECTRA_CACHE.clear()


def _cache_key(wavelength_range, molecules, L_cm, grid_sampling,
               margin, sources_override) -> Tuple:
    src_key = tuple(sorted(
        (m, tuple(v)) for m, v in (sources_override or {}).items()
    ))
    return (
        float(wavelength_range[0]), float(wavelength_range[1]),
        tuple(sorted(molecules)),
        float(L_cm), float(grid_sampling), float(margin),
        src_key,
    )


def _build_block_cache(
    wavelength_range_nm: Tuple[float, float],
    molecules: Sequence[str],
    panel_template: MixturePanel,
    L_cm: float,
    grid_sampling_nm: float,
    unit_c_ppm: float = 1e-3,
    margin_nm: float = 100.0,
    sources_override: Optional[Dict[str, List[str]]] = None,
) -> _BlockSpectraCache:
    """
    Один проход HITRAN/MPI/XSC по молекулам блока.

    Результат кэшируется на уровне модуля; повторные вызовы с теми же
    параметрами возвращают сохранённый объект.
    """
    key = _cache_key(wavelength_range_nm, molecules, L_cm,
                     grid_sampling_nm, margin_nm, sources_override)
    cached = _GLOBAL_SPECTRA_CACHE.get(key)
    if cached is not None:
        return cached
    lo, hi = wavelength_range_nm
    lo_ext, hi_ext = lo - margin_nm, hi + margin_nm

    conditions = dict(panel_template.conditions or {})
    T_K = float(conditions.get('T_K', 310.0))
    p_atm = float(conditions.get('p_atm', 1.0))
    diluent = conditions.get('diluent', {'air': 1.0})
    sources_map = {b.name: (b.sources or [b.source] if b.source else None)
                   for b in panel_template.biomarkers}
    if sources_override:
        for k, v in sources_override.items():
            sources_map[k] = list(v)

    inst = Instrument(
        wavelength_range=(lo_ext, hi_ext),
        sampling_step=grid_sampling_nm,
        ils=None,
    )
    gen = SpectrumGenerator(instrument=inst, noise_model=None, seed=None)

    grid: Optional[np.ndarray] = None
    od_map: Dict[str, np.ndarray] = {}

    for m_name in molecules:
        try:
            mix = GasMixture(
                composition={m_name: unit_c_ppm},
                T_K=T_K, p_atm=p_atm, L_cm=L_cm,
                sources={m_name: sources_map.get(m_name)}
                    if sources_map.get(m_name) else None,
                diluent=diluent,
            )
            spec = gen.generate(mix)
            if grid is None:
                grid = spec.wavelength_nm.copy()
            od_per_unit = -np.log(np.clip(spec.true_transmittance,
                                          1e-12, 1.0)) / unit_c_ppm
            od_map[m_name] = od_per_unit
        except Exception:
            if grid is None:
                grid = np.arange(lo_ext, hi_ext + grid_sampling_nm / 2,
                                  grid_sampling_nm)
            od_map[m_name] = np.zeros_like(grid)

    result = _BlockSpectraCache(
        wavelength_nm=grid,
        od_per_unit_c=od_map,
        panel_template=panel_template,
    )
    _GLOBAL_SPECTRA_CACHE[key] = result
    return result


def _channel_response(
    centers_nm: Sequence[float], fwhm_nm: float, grid_nm: np.ndarray
) -> np.ndarray:
    """
    Матрица канальных функций R[k, i] = phi_k(lambda_i), нормированных
    так, что sum_i R[k, i] * d_lambda = 1.
    """
    centers = np.asarray(list(centers_nm), dtype=float)
    K = len(centers)
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    d_lambda = float(np.mean(np.diff(grid_nm)))
    diff = grid_nm[None, :] - centers[:, None]
    R = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    norm = R.sum(axis=1) * d_lambda
    norm = np.where(norm > 0, norm, 1.0)
    R = R / norm[:, None]
    return R


def build_response_matrix_from_cache(
    centers_nm: Sequence[float],
    fwhm_nm: float,
    molecules: Sequence[str],
    c_design: Dict[str, float],
    kappa: float,
    cache: _BlockSpectraCache,
    kappa_per_molecule: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, ChannelSet]:
    """
    Быстрая сборка A^(r) и T^design через кэш тонкосетных спектров.

    A[k, m] = κ_m * integ_k(OD_per_unit_c[m])
    T_design[k] = exp(-integ_k(sum_m κ_m * c_design[m] * OD_per_unit_c[m]))

    Тут κ_m = kappa_per_molecule.get(m, kappa) — селективное
    преконцентрирование (например, H2O/CO2 не задерживаются сорбентом).

    integ_k — свёртка с гауссовой инструментальной функцией.
    """
    grid = cache.wavelength_nm
    R = _channel_response(centers_nm, fwhm_nm, grid)
    d_lambda = float(np.mean(np.diff(grid)))

    centers_sorted = sorted(set(float(c) for c in centers_nm))
    channel_set = _make_channel_set(centers_sorted, fwhm_nm)

    def _kappa_for(m: str) -> float:
        if kappa_per_molecule and m in kappa_per_molecule:
            return float(kappa_per_molecule[m])
        return float(kappa)

    K = len(centers_sorted)
    M = len(molecules)
    A = np.zeros((K, M), dtype=float)
    od_total = np.zeros_like(grid)
    for j, m_name in enumerate(molecules):
        od_m = cache.od_per_unit_c.get(m_name)
        if od_m is None:
            continue
        k_eff = _kappa_for(m_name)
        per_unit_in_channels = (R * od_m[None, :]).sum(axis=1) * d_lambda
        A[:, j] = k_eff * per_unit_in_channels
        od_total = od_total + k_eff * float(c_design.get(m_name, 0.0)) * od_m

    T_design = np.array([
        float((R[k] * np.exp(-od_total)).sum() * d_lambda)
        for k in range(K)
    ])
    return A, T_design, channel_set


def build_response_matrix(
    centers_nm: Sequence[float],
    fwhm_nm: float,
    molecules: Sequence[str],
    c_design: Dict[str, float],
    L_cm: float,
    kappa: float,
    panel_template: MixturePanel,
    grid_sampling_nm: float = 0.05,
    unit_c_ppm: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, ChannelSet]:
    """
    Построить матрицу отклика A^(r) в OD-пространстве и вектор T^design.

    Метод: для каждой молекулы m генерируется спектр на тонкой сетке с
    единичной концентрацией c0 = unit_c_ppm, обрабатывается каналами;
    отклик A_{km} = κ * (-ln T_k) / c0. Это даёт линеаризованную колонку
    под молекулу m. Полный вектор y = A * c (при подстановке c = κ c_real)
    даёт OD по каналам.

    Параллельно генерируется спектр со всей design-смесью и обрабатывается
    каналами — это T^design, нужный для шумовой модели.

    Parameters
    ----------
    centers_nm : sequence of float
        Центры каналов.
    fwhm_nm : float
        FWHM каналов (одинаковая для всех в блоке).
    molecules : sequence of str
        Имена молекул в порядке столбцов A.
    c_design : dict[str, float]
        Расчётные концентрации в ppm для всех molecules (без κ).
    L_cm, kappa : float
        Геометрия кюветы и предконцентрирование.
    panel_template : MixturePanel
        Источник conditions (T, p, источники сечений). Концентрации
        биомаркеров в template игнорируются — берётся только sources.
    grid_sampling_nm : float, default 0.05
        Шаг тонкой сетки для генерации спектров. 0.05 нм достаточно для
        большинства полос в УФ/видимом; для тонких ИК-линий нужно мельче.
    unit_c_ppm : float, default 1e-3
        Концентрация на молекулу для извлечения колонки A. Должна быть
        достаточно мала для линеаризации (τ < 0.1 при единичной κ).

    Returns
    -------
    A : np.ndarray, shape (K, M)
        Матрица отклика в OD-пространстве, A_{km} = κ * L * <α_m>_k.
    T_design : np.ndarray, shape (K,)
        Канализованное прохождение при design-смеси с учётом κ.
    channel_set : ChannelSet
        Сформированный набор каналов (на случай если нужен далее).
    """
    centers_list = sorted(set(float(c) for c in centers_nm))
    channel_set = _make_channel_set(centers_list, fwhm_nm)

    lo, hi = min(centers_list) - 2 * fwhm_nm, max(centers_list) + 2 * fwhm_nm

    conditions = dict(panel_template.conditions or {})
    T_K = float(conditions.get('T_K', 310.0))
    p_atm = float(conditions.get('p_atm', 1.0))
    diluent = conditions.get('diluent', {'air': 1.0})

    inst = Instrument(
        wavelength_range=(lo, hi),
        sampling_step=grid_sampling_nm,
        ils=None,  # ILS отдельный — каналы делают свою свёртку
    )
    gen = SpectrumGenerator(instrument=inst, noise_model=None, seed=None)

    sources_map = {b.name: (b.sources or [b.source] if b.source else None)
                   for b in panel_template.biomarkers}

    # --- A: по молекуле колонкой ---
    K = len(channel_set)
    M = len(molecules)
    A = np.zeros((K, M), dtype=float)

    for j, m_name in enumerate(molecules):
        mix = GasMixture(
            composition={m_name: unit_c_ppm},
            T_K=T_K, p_atm=p_atm, L_cm=L_cm,
            sources={m_name: sources_map.get(m_name)} if sources_map.get(m_name) else None,
            diluent=diluent,
        )
        spec = gen.generate(mix)
        ch_spec = channelize(spec, channel_set)
        # OD от единичной концентрации
        od_unit = ch_spec.true_optical_depth
        # Колонка A с учётом κ: A_{km} = κ * (OD_k / c0)
        A[:, j] = kappa * od_unit / unit_c_ppm

    # --- T^design: всё вместе при design ---
    mix_full = GasMixture(
        composition={m: float(c_design[m]) * kappa for m in molecules},
        T_K=T_K, p_atm=p_atm, L_cm=L_cm,
        sources={m: sources_map.get(m) for m in molecules
                 if sources_map.get(m) is not None} or None,
        diluent=diluent,
    )
    spec_full = gen.generate(mix_full)
    ch_full = channelize(spec_full, channel_set)
    T_design = ch_full.values_T_true.copy()

    return A, T_design, channel_set


# ============================================================================
# Шумовая ковариация и базис дрейфа
# ============================================================================

def build_noise_covariance_OD(
    T_design: np.ndarray,
    channel_centers_nm: np.ndarray,
    noise_model: NoiseModel,
) -> np.ndarray:
    """
    Шумовая ковариация в OD-пространстве при заданных T^design.

    Преобразование T → OD = -ln T линеаризуется как δOD_k = -δT_k / T_k.
    Поэтому Cov_OD[k,l] = Cov_T[k,l] / (T_k * T_l).

    Включает три стационарных вклада (таблица 2.6):
      1. thermal:  δ_kl σ_T²              (в T)
      2. shot:     δ_kl T_k / N_0          (в T)
      3. colored:  σ_col² ρ^|k-l|          (в T, AR(1))

    Дрейф и spikes сюда не входят — дрейф обрабатывается через B^(r),
    spikes — отдельным шагом препроцессинга.

    Parameters
    ----------
    T_design : np.ndarray, shape (K,)
        Прохождение при design-смеси.
    channel_centers_nm : np.ndarray, shape (K,)
        Центры каналов (для AR(1) по индексу канала).
    noise_model : NoiseModel
        Параметры шумовой модели.

    Returns
    -------
    C_n_OD : np.ndarray, shape (K, K)
        Симметричная положительно определённая ковариация.
    """
    K = len(T_design)
    T = np.asarray(T_design, dtype=float)
    T = np.clip(T, 1e-9, None)  # защита от деления на 0

    # === Ковариация в T-пространстве ===
    C_T = np.zeros((K, K), dtype=float)

    # 1. Тепловой шум: σ_T² δ_kl
    if noise_model.thermal_sigma is not None:
        sig = float(noise_model.thermal_sigma)
        C_T += np.eye(K) * sig ** 2

    # 2. Дробовой шум: T_k / N_0 на диагонали
    if noise_model.shot_n_photons_max is not None:
        N0 = float(noise_model.shot_n_photons_max)
        C_T += np.diag(T / N0)

    # 3. Цветной AR(1): σ_col² ρ^|k-l| по индексу канала
    if noise_model.colored_sigma is not None:
        sig_c = float(noise_model.colored_sigma)
        rho = float(noise_model.colored_ar)
        k_idx = np.arange(K)
        lag = np.abs(k_idx[:, None] - k_idx[None, :])
        C_T += sig_c ** 2 * (rho ** lag)

    # === Линеаризация в OD: δOD_k = -δT_k / T_k ===
    inv_T = 1.0 / T
    C_OD = C_T * np.outer(inv_T, inv_T)
    # Симметризуем после численных операций
    C_OD = 0.5 * (C_OD + C_OD.T)
    return C_OD


def build_drift_basis(channel_centers_nm: np.ndarray, degree: int) -> np.ndarray:
    """
    Полиномиальный базис дрейфа B^(r) по центрам каналов.

    B[k, j] = c_k^j, j = 0..degree, где c_k — центр k-го канала,
    нормированный на [-1, 1] для численной устойчивости.

    Возвращает матрицу (K, degree+1).
    """
    c = np.asarray(channel_centers_nm, dtype=float)
    K = len(c)
    if K < degree + 1:
        raise ValueError(
            f"Для полинома степени {degree} нужно ≥ {degree+1} каналов, "
            f"передано {K}"
        )
    c_mid = c.mean()
    c_half = max(0.5 * (c.max() - c.min()), 1e-6)
    c_norm = (c - c_mid) / c_half
    B = np.column_stack([c_norm ** j for j in range(degree + 1)])
    return B


# ============================================================================
# Маргинальная информация Фишера и косой проектор
# ============================================================================

def oblique_projector_perp(B: np.ndarray, C_inv: np.ndarray) -> np.ndarray:
    """
    Косой проектор P_B^⊥ в метрике C_inv (раздел 2.6 диплома, eq:oblique-projector).

        P_B^⊥ = I - B (B^T C_inv B)^{-1} B^T C_inv

    P_B^⊥ B = 0; P_B^⊥ — идемпотент (P² = P).
    Не симметричен, но P_B^⊥ C_inv симметричен (sym_projector_identity).
    """
    BtCinv = B.T @ C_inv
    Gram = BtCinv @ B
    K = B.shape[0]
    # Численно устойчиво через solve вместо inv
    P_perp = np.eye(K) - B @ np.linalg.solve(Gram, BtCinv)
    return P_perp


def marginal_fisher(
    A: np.ndarray, C_n_OD: np.ndarray, B: np.ndarray, N_breaths: int = 1
) -> np.ndarray:
    """
    Маргинальная Фишера для блока:
        I_{c|d} = N * A^T C_n^{-1} P_B^⊥ A   (eq:fisher-marginal)

    Линейный рост по N через постоянство C_n при усреднении.
    """
    C_inv = np.linalg.inv(C_n_OD)
    P_perp = oblique_projector_perp(B, C_inv)
    I_marg = N_breaths * (A.T @ C_inv @ P_perp @ A)
    # Симметризация после численных операций
    I_marg = 0.5 * (I_marg + I_marg.T)
    return I_marg


# ============================================================================
# Целевой функционал и J*_m
# ============================================================================

def population_params(
    targets: Sequence[str],
    panel0: MixturePanel,
    panel1: MixturePanel,
    cv_table: Dict[str, float],
) -> Dict[str, Tuple[float, float]]:
    """
    Для каждой целевой молекулы вычислить (delta_m, sigma_b_m) в ppm.

      delta_m   = |μ₁ − μ₀|        (популяционный сдвиг центров, ppm)
      sigma_b_m = sqrt(μ₀ · μ₁) · CV   (популяционное СКО, ppm)
                  — биологическое СКО в логнормали при CV порядка ≤0.5
                  даёт sigma_b ≈ μ·CV в линеаризации.

    В формуле J*_m используется delta_m / sigma_b_m (см. eq:Jstar-m-def).
    """
    by0 = {b.name: b.c_ppm for b in panel0.biomarkers}
    by1 = {b.name: b.c_ppm for b in panel1.biomarkers}
    out: Dict[str, Tuple[float, float]] = {}
    for m in targets:
        c0, c1, cv = by0[m], by1[m], cv_table[m]
        delta = abs(c1 - c0)
        sigma_b = float(np.sqrt(c0 * c1) * cv)
        out[m] = (delta, sigma_b)
    return out


def compute_J_star_m(
    I_marg: np.ndarray, m_idx: int, delta_m: float, sigma_b_m: float,
    ridge: float = 1e-12,
) -> Tuple[float, float]:
    """
    Покомпонентный одномерный показатель информации (eq:Jstar-m-def):

        J*_m = (delta_m / sigma_b_m)² * sigma_b_m² / sigma_i_m²
             = delta_m² / sigma_i_m²,

    где sigma_i_m² = [I_marg^{-1}]_{mm} — инструментальная дисперсия оценки c_m.

    При близкой к сингулярной I_marg (что может произойти, когда столбцы
    A^(r) близки к коллинеарным после проекции на ortho-dop B^(r))
    добавляется небольшая регуляризация ridge·trace(I)/M, чтобы получить
    численно стабильную, но честно «плохую» оценку (большая sigma_i).
    Это предотвращает возврат заведомо нефизичных значений типа sigma=1e-15.

    Returns
    -------
    J_star_m, sigma_i_m : float
    """
    M = I_marg.shape[0]
    # Добавляем гребневую регуляризацию: масштабируем по среднему диагональному
    # элементу, чтобы не сместить нормально обусловленные матрицы.
    diag_mean = float(np.trace(I_marg) / max(M, 1))
    reg = ridge * max(diag_mean, 1e-30)
    I_reg = I_marg + reg * np.eye(M)
    try:
        I_inv = np.linalg.inv(I_reg)
    except np.linalg.LinAlgError:
        # Полностью сингулярная — возвращаем большую sigma
        return 0.0, float('inf')
    sigma_i_sq = float(I_inv[m_idx, m_idx])
    if not np.isfinite(sigma_i_sq) or sigma_i_sq <= 0:
        return 0.0, float('inf')
    sigma_i = float(np.sqrt(sigma_i_sq))
    delta_z = delta_m / sigma_b_m  # стандартизированный эффект-сайз
    J_star = float(delta_z ** 2 * (sigma_b_m ** 2 / sigma_i_sq))
    return J_star, sigma_i


# ============================================================================
# Стартовый каркас
# ============================================================================

def build_scaffold(
    block: LocalBlock,
    panel_template: MixturePanel,
    grid_sampling_nm: float = 0.05,
    sources_override: Optional[Dict[str, List[str]]] = None,
) -> List[float]:
    """
    Начальный каркас K^(0) для жадного алгоритма (раздел 4.4.1 диплома).

    Правила:
      1. По одному каналу на каждую молекулу из M_r — у пика сечения.
      2. Три канала по краям блока (для базиса дрейфа).

    Дубликаты по min_dist_nm объединяются.
    """
    lo, hi = block.wavelength_range_nm

    # 1. Для каждой молекулы — найти пик сечения
    conditions = dict(panel_template.conditions or {})
    T_K = float(conditions.get('T_K', 310.0))
    diluent = conditions.get('diluent', {'air': 1.0})
    sources_map = {b.name: (b.sources or [b.source] if b.source else None)
                   for b in panel_template.biomarkers}
    if sources_override:
        for k, v in sources_override.items():
            sources_map[k] = list(v)

    inst = Instrument(
        wavelength_range=(lo, hi),
        sampling_step=grid_sampling_nm,
        ils=None,
    )
    gen = SpectrumGenerator(instrument=inst, noise_model=None, seed=None)

    peak_centers: List[float] = []
    for m_name in block.molecules:
        try:
            mix = GasMixture(
                composition={m_name: 1e-3},
                T_K=T_K, L_cm=block.L_cm,
                sources={m_name: sources_map.get(m_name)} if sources_map.get(m_name) else None,
                diluent=diluent,
            )
            spec = gen.generate(mix)
            od = -np.log(np.clip(spec.true_transmittance, 1e-12, 1.0))
            k_max = int(np.argmax(od))
            peak_centers.append(float(spec.wavelength_nm[k_max]))
        except Exception:
            # Если молекула отсутствует в данных — берём середину блока
            peak_centers.append(0.5 * (lo + hi))

    # 2. Краевые/средний каналы для дрейфа
    drift_centers = [lo, 0.5 * (lo + hi), hi]

    # Объединение с фильтром по минимальному расстоянию
    all_candidates = sorted(set(peak_centers + drift_centers))
    pruned: List[float] = []
    for c in all_candidates:
        if not pruned or (c - pruned[-1]) >= block.min_dist_nm:
            pruned.append(c)
        else:
            # Слишком близко — заменим предыдущим на среднее
            pruned[-1] = 0.5 * (pruned[-1] + c)

    # Прижимаем к сетке Λ_r
    grid = block.candidate_grid()
    snapped = sorted(set(
        float(grid[np.argmin(np.abs(grid - c))]) for c in pruned
    ))

    # Гарантируем минимальный размер каркаса для невырожденности I_marg:
    # K >= |M_r| + dim(B). Заполняем наибольший зазор последовательно.
    min_K = len(block.molecules) + block.drift_dim
    # Если геометрически нельзя — снижаем требование, попутно сохраняя как
    # можно больше каналов
    max_feasible = int(np.floor((hi - lo) / block.min_dist_nm)) + 1
    if min_K > max_feasible:
        min_K = max_feasible

    safety_iter = 100
    while len(snapped) < min_K and safety_iter > 0:
        safety_iter -= 1
        if len(snapped) < 2:
            mid = 0.5 * (lo + hi)
            c_snap = float(grid[np.argmin(np.abs(grid - mid))])
            if c_snap not in snapped:
                snapped.append(c_snap)
                snapped.sort()
            else:
                break
            continue

        gaps = [(snapped[i+1] - snapped[i], i) for i in range(len(snapped) - 1)]
        gaps.sort(reverse=True)
        added = False
        for gap_size, idx in gaps:
            mid = 0.5 * (snapped[idx] + snapped[idx+1])
            # Перебираем по сетке от середины наружу — ищем валидную точку
            grid_local = grid[(grid > snapped[idx]) & (grid < snapped[idx+1])]
            if len(grid_local) == 0:
                continue
            # Сортируем по близости к середине
            order = np.argsort(np.abs(grid_local - mid))
            for c_candidate in grid_local[order]:
                c_snap = float(c_candidate)
                if c_snap in snapped:
                    continue
                if all(abs(c_snap - s) >= block.min_dist_nm - 1e-9
                       for s in snapped):
                    snapped.append(c_snap)
                    snapped.sort()
                    added = True
                    break
            if added:
                break
        if not added:
            break
    return snapped


# ============================================================================
# Жадная процедура forward selection
# ============================================================================

def _compute_psi_cached(
    centers: Sequence[float],
    block: LocalBlock,
    cache: _BlockSpectraCache,
    noise_model: NoiseModel,
    pop_params: Dict[str, Tuple[float, float]],
    c_design: Dict[str, float],
    target_indices: List[int],
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """
    Внутренняя функция: вычислить Psi = max_m 1/J*_m для конфигурации centers
    с использованием закэшированных тонкосетных спектров.
    """
    A, T_design, ch_set = build_response_matrix_from_cache(
        centers_nm=centers,
        fwhm_nm=block.fwhm_nm,
        molecules=block.molecules,
        c_design=c_design,
        kappa=block.kappa,
        cache=cache,
        kappa_per_molecule=block.kappa_per_molecule,
    )
    C_n = build_noise_covariance_OD(T_design, ch_set.centers, noise_model)
    B = build_drift_basis(ch_set.centers, block.drift_degree)
    I_marg = marginal_fisher(A, C_n, B, N_breaths=block.N_breaths)

    Js: Dict[str, float] = {}
    sigmas: Dict[str, float] = {}
    psi = 0.0
    for j, m_name in zip(target_indices, block.targets):
        delta_m, sigma_b = pop_params[m_name]
        J, sigma_i = compute_J_star_m(I_marg, j, delta_m, sigma_b)
        Js[m_name] = J
        sigmas[m_name] = sigma_i
        psi = max(psi, 1.0 / max(J, 1e-30))
    return psi, Js, sigmas


def _compute_psi(
    centers: Sequence[float],
    block: LocalBlock,
    panel_template: MixturePanel,
    noise_model: NoiseModel,
    pop_params: Dict[str, Tuple[float, float]],
    c_design: Dict[str, float],
    target_indices: List[int],
    grid_sampling_nm: float = 0.05,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """Историческая обёртка для обратной совместимости."""
    cache = _build_block_cache(
        wavelength_range_nm=block.wavelength_range_nm,
        molecules=block.molecules,
        panel_template=panel_template,
        L_cm=block.L_cm,
        grid_sampling_nm=grid_sampling_nm,
    )
    return _compute_psi_cached(
        centers, block, cache, noise_model, pop_params, c_design, target_indices,
    )


def audit_block(
    block: LocalBlock,
    panel0: MixturePanel,
    panel1: MixturePanel,
    noise_model: NoiseModel,
    panel_template: Optional[MixturePanel] = None,
    J_star_min: float = 1.0,
    epsilon_stop: float = 0.05,
    max_K: int = 30,
    grid_sampling_nm: float = 0.05,
    verbose: bool = False,
) -> AuditResult:
    """
    Главная функция: жадный forward selection для покомпонентного аудита блока.

    Алгоритм (раздел 4.4 диплома):
      1. Стартовый каркас K^(0) (минимальный размер для разрешимости).
      2. На каждом шаге: добавить канал, минимизирующий
         Psi = max_m 1/J*_m (E-оптимальный критерий по T_r).
      3. Останов: относительный выигрыш < epsilon_stop два шага подряд
         или достигнут max_K.

    Parameters
    ----------
    block : LocalBlock
        Описание блока с целевыми, интерферентами, FWHM и κ.
    panel0, panel1 : MixturePanel
        Субпопуляции «норма» и «болезнь». Различие центров даёт delta_m.
    noise_model : NoiseModel
        Параметры C_n.
    panel_template : MixturePanel, optional
        Источник `conditions` и `sources` для генерации спектров. По умолчанию
        берётся panel0.
    J_star_min : float, default 1.0
        Порог для классификации PASS+/PASS/FAIL.
        Будет уточнён в главе 5; для аудита служит ориентиром.
    epsilon_stop : float, default 0.05
        Порог плато: 5% относительного выигрыша.
    max_K : int, default 30
        Аварийный предел числа каналов.
    grid_sampling_nm : float, default 0.05
        Шаг тонкой сетки для генерации спектров.
    verbose : bool
        Печатать прогресс по шагам.

    Returns
    -------
    AuditResult
    """
    if panel_template is None:
        panel_template = panel0

    # Расчётные концентрации
    c_design = design_concentrations(
        block.targets, block.interferents, panel0, panel1,
        block.cv_table, block.p_lo,
    )

    # Популяционные параметры для J*_m
    pop = population_params(block.targets, panel0, panel1, block.cv_table)

    # Индексы целевых биомаркеров в molecules
    target_indices = [block.molecules.index(m) for m in block.targets]

    # Кэш тонкосетных OD по молекулам — один проход по HITRAN/MPI/XSC
    if verbose:
        print(f"[{block.name}] построение кэша тонкосетных спектров...")
    cache = _build_block_cache(
        wavelength_range_nm=block.wavelength_range_nm,
        molecules=block.molecules,
        panel_template=panel_template,
        L_cm=block.L_cm,
        grid_sampling_nm=grid_sampling_nm,
        sources_override=block.sources_override,
    )

    # Стартовый каркас
    current_centers = list(build_scaffold(block, panel_template,
                                            grid_sampling_nm,
                                            sources_override=block.sources_override))
    grid = block.candidate_grid()

    if verbose:
        print(f"[{block.name}] стартовый каркас: {len(current_centers)} каналов")
        print(f"   {[f'{c:.1f}' for c in current_centers]}")

    psi_curr, Js_curr, sigs_curr = _compute_psi_cached(
        current_centers, block, cache, noise_model, pop,
        c_design, target_indices,
    )

    history_psi = [psi_curr]
    history_lambda: List[float] = []
    history_gain: List[float] = []

    low_gain_streak = 0  # сколько шагов подряд g < epsilon

    while len(current_centers) < max_K:
        # Кандидаты: точки сетки, не вошедшие в текущую конфигурацию
        # и удовлетворяющие ограничению на минимальное расстояние
        used = set(round(c, 6) for c in current_centers)
        candidates: List[float] = []
        for lam in grid:
            if round(float(lam), 6) in used:
                continue
            if all(abs(lam - c) >= block.min_dist_nm - 1e-9
                   for c in current_centers):
                candidates.append(float(lam))

        if not candidates:
            break

        # Перебор кандидатов
        best_lam = None
        best_psi = np.inf
        best_Js = None
        best_sigs = None
        for lam in candidates:
            trial = sorted(current_centers + [lam])
            psi_t, Js_t, sigs_t = _compute_psi_cached(
                trial, block, cache, noise_model, pop,
                c_design, target_indices,
            )
            if psi_t < best_psi:
                best_psi = psi_t
                best_lam = lam
                best_Js = Js_t
                best_sigs = sigs_t

        if best_lam is None:
            break

        gain = (psi_curr - best_psi) / max(psi_curr, 1e-30)
        psi_curr = best_psi
        Js_curr = best_Js
        sigs_curr = best_sigs
        current_centers = sorted(current_centers + [best_lam])

        history_psi.append(psi_curr)
        history_lambda.append(float(best_lam))
        history_gain.append(float(gain))

        if verbose:
            print(f"[{block.name}] +{best_lam:.1f}нм, K={len(current_centers)}, "
                  f"Psi={psi_curr:.4f}, gain={gain*100:.2f}%")

        if gain < epsilon_stop:
            low_gain_streak += 1
            if low_gain_streak >= 2:
                # Откатываем последние 2 шага (плато)
                if len(current_centers) >= 2:
                    current_centers = current_centers[:-2] \
                        if len(history_lambda) >= 2 else current_centers
                # ... фактически, оставляем последнюю конфигурацию
                break
        else:
            low_gain_streak = 0

    # Финальные значения после возможного отката
    psi_final, Js_final, sigs_final = _compute_psi_cached(
        current_centers, block, cache, noise_model, pop,
        c_design, target_indices,
    )

    # Вердикты
    verdict: Dict[str, str] = {}
    delta_rel: Dict[str, float] = {}
    for m in block.targets:
        J = Js_final[m]
        if J >= 2.0 * J_star_min:
            verdict[m] = 'PASS+'
        elif J >= J_star_min:
            verdict[m] = 'PASS'
        else:
            verdict[m] = 'FAIL'
        delta_rel[m] = sigs_final[m] / max(c_design[m], 1e-30)

    J_star_block = min(Js_final.values()) if Js_final else 0.0

    return AuditResult(
        block=block,
        lambdas_star=np.array(current_centers),
        K_star=len(current_centers),
        sigma_i=sigs_final,
        J_star_m=Js_final,
        delta_rel=delta_rel,
        verdict=verdict,
        J_star_block=J_star_block,
        history_psi=history_psi,
        history_lambda=history_lambda,
        history_gain=history_gain,
        c_design=c_design,
        J_star_min=J_star_min,
    )


# ============================================================================
# Визуализация
# ============================================================================

def plot_audit_trajectory(result: AuditResult, ax=None, **kwargs):
    """
    График динамики жадной процедуры: относительный выигрыш по шагам.

    Горизонтальная линия — порог плато epsilon_stop = 5%.
    """
    from .plotstyle import plot, new_figure, DEFAULTS, PALETTE

    if ax is None:
        fig, ax = new_figure(figsize=kwargs.pop('figsize', (8, 4.5)))
    else:
        fig = ax.figure

    if not result.history_gain:
        ax.text(0.5, 0.5, 'нет шагов жадной процедуры',
                ha='center', va='center', transform=ax.transAxes)
        return ax

    steps = np.arange(1, len(result.history_gain) + 1)
    gains_pct = np.array(result.history_gain) * 100
    ax.bar(steps, gains_pct, color=PALETTE['primary'], alpha=0.7,
           edgecolor=PALETTE['primary'], width=0.7,
           label='отн. выигрыш')
    ax.axhline(5.0, color=PALETTE['accent'], linestyle='--', lw=1.5,
               label=r'$\varepsilon_\mathrm{stop}=5\%$')

    ax.set_xlabel('Шаг жадной процедуры $t$')
    ax.set_ylabel(r'Относительный выигрыш $g^{(t)}$, \%')
    ax.set_title(f'Аудит блока {result.block.name}: K* = {result.K_star}')
    ax.legend(framealpha=DEFAULTS['legend_framealpha'])
    return ax


def plot_audit_configuration(
    result: AuditResult,
    panel_template: MixturePanel,
    ax=None,
    grid_sampling_nm: float = 0.05,
    **kwargs,
):
    """
    Итоговая конфигурация K* на фоне сечений молекул блока.
    """
    from .plotstyle import new_figure, DEFAULTS, PALETTE

    if ax is None:
        fig, ax = new_figure(figsize=kwargs.pop('figsize', (10, 5)))
    else:
        fig = ax.figure

    lo, hi = result.block.wavelength_range_nm
    inst = Instrument(
        wavelength_range=(lo - 2 * result.block.fwhm_nm,
                          hi + 2 * result.block.fwhm_nm),
        sampling_step=grid_sampling_nm,
    )
    gen = SpectrumGenerator(instrument=inst)

    conditions = dict(panel_template.conditions or {})
    T_K = float(conditions.get('T_K', 310.0))
    sources_map = {b.name: (b.sources or [b.source] if b.source else None)
                   for b in panel_template.biomarkers}

    cycle = DEFAULTS['color_cycle']
    max_od = 0.0
    spectra = []
    for i, m in enumerate(result.block.molecules):
        try:
            # Используем design-концентрацию из результата аудита
            c_des = result.c_design.get(m, 1e-3)
            mix = GasMixture(
                composition={m: c_des * result.block.kappa},
                T_K=T_K, L_cm=result.block.L_cm,
                sources={m: sources_map.get(m)} if sources_map.get(m) else None,
            )
            spec = gen.generate(mix)
            od = -np.log(np.clip(spec.true_transmittance, 1e-12, 1.0))
            spectra.append((m, spec.wavelength_nm, od))
            max_od = max(max_od, od.max())
        except Exception:
            pass

    for i, (m, wl, od) in enumerate(spectra):
        color = cycle[i % len(cycle)]
        is_target = m in result.block.targets
        ax.plot(wl, od, color=color, lw=2.0 if is_target else 1.0,
                alpha=1.0 if is_target else 0.6,
                label=f'{m} (c={result.c_design.get(m, 0)*1e3:.2g} ppb)'
                      + (' [цель]' if is_target else ''))

    marker_y = max_od * 1.05 if max_od > 0 else 1.0
    for lam in result.lambdas_star:
        ax.axvline(lam, color=PALETTE['neutral'], lw=0.6, alpha=0.4)
    ax.scatter(result.lambdas_star,
               np.full(len(result.lambdas_star), marker_y),
               color=PALETTE['accent'], marker='v', s=80, zorder=5,
               label=f'QD-каналы (K*={result.K_star})')

    ax.set_xlim(lo, hi)
    ax.set_xlabel('Длина волны, нм')
    ax.set_ylabel(r'OD при $c_m^\mathrm{design} \cdot \kappa$')
    ax.set_title(f'Конфигурация {result.block.name}, '
                 f'κ={result.block.kappa:.0f}')
    ax.legend(framealpha=DEFAULTS['legend_framealpha'], fontsize=8, loc='best')
    return ax
