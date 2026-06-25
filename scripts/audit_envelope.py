#!/usr/bin/env python
"""
scripts/audit_envelope.py
=========================
Огибающая возможностей платформы: J_possible(α_peak) на текущей платформе
(L = 25 см, κ = 10³, N = 30, FWHM поддиапазонные) для синтетического
Gauss-биомаркера в каждом из трёх поддиапазонов (УФ/Vis/БИК).

Идея. Для произвольной молекулы с известным α_peak (пик сечения, cm²/molec)
платформа обеспечивает инструментальную точность σ_i(α_peak), не зависящую от
конкретной молекулы (в пределах Gauss-приближения формы полосы). Биомаркер
проходит PASS, если δ_m² · (1/σ_i²) ≥ J*_min, т. е. J_possible(α_peak) ≥ J*_min/δ_m².

Что считается:
  1. σ_i(α_peak) для Gauss-биомаркера с FWHM_line = band-typical,
     в двух сценариях — изолированный и с band-typical интерферентами
     (для БИК — H2O+CO2 с κ=1; для УФ/Vis значимых атмосферных интерферентов нет,
     добавляется синтетический интерферент с ρ=0,6 для иллюстрации штрафа).
  2. J_possible(α_peak) := 1/σ_i² — платформенная Фишера на единицу δ².
  3. Точки 11 биомаркеров рака лёгких как overlay: их α_peak и σ_i из
     основной audit-программы.

Запуск:
    conda run -n spectro python scripts/audit_envelope.py

Выход:
    results/audit_envelope.json
    latex_doc/figures/audit_envelope.pdf
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent))

from spectrolib import MixturePanel, NoiseModel, mpi, hitran, hitran_xsc

from spectrolib.audit import (
    _build_block_cache, build_response_matrix_from_cache,
    build_noise_covariance_OD, build_drift_basis, marginal_fisher,
    compute_J_star_m, compute_J_star_min,
)
from spectrolib.plotstyle import save as save_figure, new_figure, PALETTE


PANEL_MU0 = REPO / 'example_panels/lung_cancer_control_mu0.yaml'
PANEL_MU1 = REPO / 'example_panels/lung_cancer_disease_mu1.yaml'
NOISE_FILE = REPO / 'example_noise_models/table_2_6.yaml'
RESULTS_DIR = REPO / 'results'
FIG_DIR_REPO = RESULTS_DIR
FIG_DIR_LATEX = REPO.parent / 'latex_doc' / 'figures'


# ----- параметры платформы -----
L_CM = 25.0
KAPPA = 1000.0
N_BREATHS = 30
DRIFT_DEG = 1

# Лошмидтова плотность при T=310 K, p=1 атм, per ppm
N_LOSCHMIDT_PER_PPM = 2.367e13  # molec/cm³ per ppm


# ----- определение поддиапазонов -----
# Каждый band: (имя, λ_peak центр, FWHM_channel, FWHM_line синтетического биомаркера,
#               блок [λ_lo, λ_hi], рабочая концентрация ppb)
BANDS = {
    'UV': dict(
        lambda0=275.0, fwhm_channel=25.0, fwhm_line=10.0,
        block=(220.0, 340.0), c_design_ppb=10.0,
        title='УФ-поддиапазон (FWHM кан. = 25 нм)',
        interferent='synthetic_overlap',
    ),
    'Vis': dict(
        lambda0=550.0, fwhm_channel=30.0, fwhm_line=30.0,
        block=(450.0, 650.0), c_design_ppb=10.0,
        title='Видимый поддиапазон (FWHM кан. = 30 нм)',
        interferent='synthetic_overlap',
    ),
    'NIR': dict(
        lambda0=2300.0, fwhm_channel=150.0, fwhm_line=80.0,
        block=(2050.0, 2500.0), c_design_ppb=10.0,
        title='БИК-поддиапазон (FWHM кан. = 150 нм)',
        interferent='H2O_CO2',
    ),
}


# ----- свип по α_peak -----
ALPHA_PEAK_GRID = np.logspace(-22, -16, 25)  # cm²/molec
# Опорный α для разовой greedy-расстановки каналов (середина сетки, линейный режим)
ALPHA_REF = 1e-19  # cm²/molec


def synthetic_alpha_profile(grid_nm: np.ndarray, lambda0: float,
                             fwhm_line: float, alpha_peak: float) -> np.ndarray:
    """Gaussian-форма сечения с пиком α_peak в λ0 и шириной FWHM_line."""
    sigma = fwhm_line / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return alpha_peak * np.exp(-0.5 * ((grid_nm - lambda0) / sigma) ** 2)


def build_cache_with_interferents(band_name: str, band_cfg: dict,
                                    panel_template: MixturePanel) -> '_BlockSpectraCache':
    """Построить кэш с band-typical интерферентами (для NIR — H2O+CO2; иначе пусто)."""
    if band_cfg['interferent'] == 'H2O_CO2':
        molecules = ['H2O', 'CO2']
        sources_override = {'H2O': ['hitran'], 'CO2': ['hitran']}
    else:
        # Для УФ/Vis значимых атмосферных нет. Кэш строим только с одной "затравкой"
        # — это нужно, чтобы получить wavelength_nm сетку. Затем мы её заменим
        # на ручную регулярную сетку.
        molecules = ['H2O']  # пустышка, не используется в дальнейшем
        sources_override = {'H2O': ['hitran']}
    cache = _build_block_cache(
        wavelength_range_nm=band_cfg['block'],
        molecules=molecules,
        panel_template=panel_template,
        L_cm=L_CM,
        grid_sampling_nm=min(0.5, band_cfg['fwhm_line'] / 4),
        sources_override=sources_override,
    )
    return cache


def _inject_synthetic_biomarker(
    alpha_peak: float, band_cfg: dict, cache,
    include_real_interferents: bool, include_synthetic_interferent: bool,
) -> Tuple[List[str], Dict[str, float], dict]:
    """
    Инжектировать в кэш OD синтетического Gauss-биомаркера (SYNTH, idx 0) с
    пиком α_peak и, при необходимости, интерферентов. Возвращает
    (molecules, c_design, kappa_per_mol). SYNTH всегда в позиции 0.
    """
    grid = cache.wavelength_nm
    lambda0 = band_cfg['lambda0']
    fwhm_line = band_cfg['fwhm_line']
    c_des_ppm = band_cfg['c_design_ppb'] * 1e-3  # ppb -> ppm

    alpha_target = synthetic_alpha_profile(grid, lambda0, fwhm_line, alpha_peak)
    cache.od_per_unit_c['SYNTH'] = L_CM * alpha_target * N_LOSCHMIDT_PER_PPM

    molecules = ['SYNTH']
    c_design = {'SYNTH': c_des_ppm}
    kappa_per_mol = None

    # Реальные интерференты (только для БИК — H2O/CO2 с κ=1)
    if include_real_interferents and band_cfg['interferent'] == 'H2O_CO2':
        molecules += ['H2O', 'CO2']
        c_design['H2O'] = 4.5e4   # ppm (4.5%)
        c_design['CO2'] = 4.0e4   # ppm (4%)
        kappa_per_mol = {'H2O': 1.0, 'CO2': 1.0}

    # Синтетический перекрывающийся интерферент при ρ ≈ 0,6
    if include_synthetic_interferent:
        lambda_interf = lambda0 + 0.7 * fwhm_line
        alpha_interf = synthetic_alpha_profile(grid, lambda_interf, fwhm_line, alpha_peak)
        cache.od_per_unit_c['INTERF_SYNTH'] = L_CM * alpha_interf * N_LOSCHMIDT_PER_PPM
        molecules.append('INTERF_SYNTH')
        c_design['INTERF_SYNTH'] = c_des_ppm

    return molecules, c_design, kappa_per_mol


def _sigma_i_ppb_for_centers(
    centers: Sequence[float], band_cfg: dict, molecules: List[str],
    c_design: Dict[str, float], kappa_per_mol, cache, noise: NoiseModel,
) -> float:
    """σ_i (ppb) целевого SYNTH (idx 0) для заданного набора каналов; inf если вырождено."""
    fwhm_ch = band_cfg['fwhm_channel']
    centers = sorted(centers)
    cen = np.asarray(centers)
    A, T_design, _ = build_response_matrix_from_cache(
        centers_nm=centers, fwhm_nm=fwhm_ch, molecules=molecules,
        c_design=c_design, kappa=KAPPA, cache=cache,
        kappa_per_molecule=kappa_per_mol,
    )
    C_n = build_noise_covariance_OD(T_design, cen, noise)
    try:
        B = build_drift_basis(cen, degree=DRIFT_DEG)
    except ValueError:
        return float('inf')
    I_marg = marginal_fisher(A, C_n, B, N_breaths=N_BREATHS)
    # «условный» δ=1, σ_b=1 ppb → чистый σ_i (в ppm, т.к. c_design в ppm)
    _, sigma_i_ppm = compute_J_star_m(I_marg, m_idx=0, delta_m=1.0, sigma_b_m=1.0)
    return sigma_i_ppm * 1000.0  # ppm -> ppb


# ----- параметры greedy-оптимизатора каналов (как для реальных биомаркеров) -----
GRID_STEP_FACTOR = 0.10   # δλ_r = fwhm_ch · 0.10  (audit.LocalBlock default)
MIN_DIST_FACTOR = 0.30    # |λ_k1 − λ_k2| ≥ fwhm_ch · 0.30
EPS_STOP = 0.05
MAX_K = 20


def greedy_select_centers(
    band_cfg: dict, molecules: List[str], c_design: Dict[str, float],
    kappa_per_mol, cache, noise: NoiseModel, verbose: bool = False,
) -> List[float]:
    """
    Жадная forward-selection расстановка каналов для SYNTH (idx 0), та же
    процедура, что применяется к 11 реальным биомаркерам (audit.audit_block):
    сетка центров Λ_r с шагом fwhm·0.10, ограничение |Δλ| ≥ fwhm·0.30,
    стартовый каркас «пики молекул + края блока для дрейфа», шаг минимизирует
    σ_i (= Ψ при одной целевой молекуле), останов по плато (ε=0.05) два шага.
    Канальная геометрия не зависит от α_peak в линейном по шуму режиме, поэтому
    выбор делается один раз на (band, сценарий).
    """
    lo, hi = band_cfg['block']
    fwhm_ch = band_cfg['fwhm_channel']
    lambda0 = band_cfg['lambda0']
    step = fwhm_ch * GRID_STEP_FACTOR
    min_dist = fwhm_ch * MIN_DIST_FACTOR
    grid = np.arange(lo, hi + step / 2, step)
    drift_dim = DRIFT_DEG + 1

    def snap(x: float) -> float:
        return float(grid[int(np.argmin(np.abs(grid - x)))])

    # --- стартовый каркас: пик каждой молекулы (argmax OD) + края блока ---
    grid_fine = cache.wavelength_nm
    in_block = (grid_fine >= lo) & (grid_fine <= hi)
    peak_centers = []
    for m in molecules:
        od = np.asarray(cache.od_per_unit_c[m])
        od_masked = np.where(in_block, od, -np.inf)
        peak_centers.append(snap(float(grid_fine[int(np.argmax(od_masked))])))
    drift_centers = [snap(lo), snap(0.5 * (lo + hi)), snap(hi)]

    scaffold = sorted(set(peak_centers + drift_centers))
    # прорядить по min_dist
    pruned: List[float] = []
    for c in scaffold:
        if not pruned or (c - pruned[-1]) >= min_dist - 1e-9:
            pruned.append(c)
    centers = sorted(set(pruned))

    # добить до min_K = |molecules| + drift_dim заполнением наибольших зазоров
    min_K = len(molecules) + drift_dim
    max_feasible = int(np.floor((hi - lo) / min_dist)) + 1
    min_K = min(min_K, max_feasible)
    guard = 100
    while len(centers) < min_K and guard > 0:
        guard -= 1
        gaps = sorted(((centers[i + 1] - centers[i], i)
                       for i in range(len(centers) - 1)), reverse=True)
        added = False
        for _, i in gaps:
            local = grid[(grid > centers[i]) & (grid < centers[i + 1])]
            if len(local) == 0:
                continue
            mid = 0.5 * (centers[i] + centers[i + 1])
            for cand in local[np.argsort(np.abs(local - mid))]:
                cc = float(cand)
                if all(abs(cc - s) >= min_dist - 1e-9 for s in centers):
                    centers.append(cc)
                    centers.sort()
                    added = True
                    break
            if added:
                break
        if not added:
            break

    sigma_curr = _sigma_i_ppb_for_centers(
        centers, band_cfg, molecules, c_design, kappa_per_mol, cache, noise)

    # --- forward selection ---
    low_gain_streak = 0
    while len(centers) < MAX_K:
        used = set(round(c, 6) for c in centers)
        candidates = [float(lam) for lam in grid
                      if round(float(lam), 6) not in used
                      and all(abs(lam - c) >= min_dist - 1e-9 for c in centers)]
        if not candidates:
            break
        best_lam, best_sigma = None, np.inf
        for lam in candidates:
            s = _sigma_i_ppb_for_centers(
                centers + [lam], band_cfg, molecules, c_design,
                kappa_per_mol, cache, noise)
            if s < best_sigma:
                best_sigma, best_lam = s, lam
        if best_lam is None or not np.isfinite(best_sigma):
            break
        gain = (sigma_curr - best_sigma) / max(sigma_curr, 1e-30)
        centers = sorted(centers + [best_lam])
        sigma_curr = best_sigma
        if verbose:
            print(f'      +{best_lam:.1f}нм K={len(centers)} σ={sigma_curr:.3g} '
                  f'gain={gain*100:.2f}%')
        if gain < EPS_STOP:
            low_gain_streak += 1
            if low_gain_streak >= 2:
                break
        else:
            low_gain_streak = 0
    return centers


def compute_sigma_i_for_alpha(
    alpha_peak: float, band_cfg: dict, cache, noise: NoiseModel,
    include_real_interferents: bool, include_synthetic_interferent: bool,
    panel0: MixturePanel, panel1: MixturePanel,
    centers: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """
    Для заданного α_peak построить синтетический биомаркер с Gauss-полосой
    шириной band.fwhm_line в центре block и вычислить σ_i, J_possible := 1/σ_i².

    Расстановка каналов:
      * centers is None → K=5 равномерно в окне ±1,5·FWHM_line (огибающая —
        строго платформенная характеристика, без оптимизатора);
      * centers задан → использовать готовый набор (результат greedy).
    """
    molecules, c_design, kappa_per_mol = _inject_synthetic_biomarker(
        alpha_peak, band_cfg, cache,
        include_real_interferents, include_synthetic_interferent,
    )

    if centers is None:
        # Каналы: K=5 равномерно в окне ±1,5 FWHM_line вокруг пика
        K = 5
        lambda0 = band_cfg['lambda0']
        halfspan = 1.5 * band_cfg['fwhm_line']
        centers = list(np.linspace(lambda0 - halfspan, lambda0 + halfspan, K))

    sigma_i_ppb = _sigma_i_ppb_for_centers(
        centers, band_cfg, molecules, c_design, kappa_per_mol, cache, noise)
    J_possible_per_ppb2 = 1.0 / max(sigma_i_ppb, 1e-30) ** 2
    return sigma_i_ppb, J_possible_per_ppb2


def lung_cancer_overlay() -> Dict[str, List[dict]]:
    """
    Для каждого биомаркера панели рака лёгких:
      - α_peak: максимум сечения в его рабочем блоке (из MPI или HITRAN-XSC).
      - σ_i: из results/audit_full_results.json (запуск audit_full.py).
    """
    audit_path = REPO / 'results' / 'audit_full_results.json'
    if not audit_path.exists():
        print(f'WARN: {audit_path} не найден, overlay будет пустой.')
        return {'UV': [], 'NIR': []}
    audit = json.loads(audit_path.read_text())
    rows = audit['current']['rows']

    # Загрузим спектры из MPI/XSC и найдём α_peak в нужных диапазонах
    # Биомаркеры и их рабочие блоки (как в audit_full.py)
    BIOMARKERS_UV = {
        'benzene': (235.0, 270.0),
        'toluene': (215.0, 245.0),
        'ethylbenzene': (245.0, 280.0),
        'styrene': (220.0, 260.0),
        'HCHO': (270.0, 340.0),
        'acetone': (260.0, 320.0),
        'butanone': (255.0, 295.0),
        'hexanal': (260.0, 305.0),
        'isoprene': (250.0, 280.0),
    }
    BIOMARKERS_NIR = {
        'pentane': (2150.0, 2400.0),
        'hexane': (2150.0, 2400.0),
        'isoprene': (2150.0, 2400.0),  # NIR-вариант
        'HCHO': (2100.0, 2300.0),     # NIR-вариант (HITRAN)
    }

    overlay = {'UV': [], 'NIR': []}

    def alpha_peak_in_block(loader, name, lo_nm, hi_nm):
        """loader возвращает (nu_cm⁻¹, sigma_cm², meta). Конвертим в λ_nm и ищем max."""
        nu_cm, sigma, _ = loader(name, T_target=310.0)
        wl_nm = 1e7 / nu_cm
        mask = (wl_nm >= lo_nm) & (wl_nm <= hi_nm)
        if not mask.any():
            return None
        return float(sigma[mask].max())

    # UV — MPI
    for name, (lo, hi) in BIOMARKERS_UV.items():
        try:
            alpha_peak = alpha_peak_in_block(mpi.load_mpi_sigma, name, lo, hi)
            if alpha_peak is None:
                continue
        except Exception as e:
            print(f'WARN UV {name}: {e}')
            continue
        sigma_i_ppb = None
        for r in rows:
            if r['biomarker'] == name and r['band'] == 'UV':
                sigma_i_ppb = float(r['sigma_ppb'])
                break
        if sigma_i_ppb is None or sigma_i_ppb > 1e10:
            continue
        overlay['UV'].append(dict(name=name, alpha_peak=alpha_peak,
                                   sigma_i_ppb=sigma_i_ppb))

    # NIR — XSC (для пентана/гексана/изопрена); HCHO пропускаем
    for name, (lo, hi) in BIOMARKERS_NIR.items():
        if name == 'HCHO':
            continue  # HITRAN line-by-line: α_peak плохо сопоставим с XSC
        try:
            alpha_peak = alpha_peak_in_block(hitran_xsc.load_xsc_sigma, name, lo, hi)
            if alpha_peak is None:
                continue
        except Exception as e:
            print(f'WARN NIR {name}: {e}')
            continue
        # σ_i из audit (с пометкой "_NIR")
        sigma_i_ppb = None
        target_bio = name + '_NIR'
        for r in rows:
            if r['biomarker'] == target_bio and r['band'] == 'NIR':
                sigma_i_ppb = float(r['sigma_ppb'])
                break
        if sigma_i_ppb is None or sigma_i_ppb > 1e10:
            continue
        overlay['NIR'].append(dict(name=name, alpha_peak=alpha_peak,
                                    sigma_i_ppb=sigma_i_ppb))

    return overlay


def main():
    # init_db нужен только для overlay реальных биомаркеров (mpi/xsc-сечения).
    # На некоторых машинах mpi_data/hitran_xsc_data недоступны как папки —
    # тогда overlay переиспользуется из кэша results/audit_envelope.json,
    # а сами огибающие/greedy от этих БД не зависят (band-cache на HITRAN).
    overlay_db_ok = True
    try:
        mpi.init_db(str(REPO / 'mpi_data'))
        hitran_xsc.init_db(str(REPO / 'hitran_xsc_data'))
    except Exception as exc:  # noqa: BLE001
        overlay_db_ok = False
        print(f'[warn] init_db недоступен ({exc!r}); overlay будет взят из кэша.')
    panel0 = MixturePanel.from_file(PANEL_MU0)
    panel1 = MixturePanel.from_file(PANEL_MU1)
    noise = NoiseModel.from_file(NOISE_FILE)

    J_min = compute_J_star_min(M=1)
    print(f'J*_min(M=1) = {J_min:.1f}')
    print(f'Свип α_peak: {ALPHA_PEAK_GRID[0]:.1e} ... {ALPHA_PEAK_GRID[-1]:.1e} cm²/molec')
    print(f'Платформа: L={L_CM}см, κ={KAPPA}, N={N_BREATHS}\n')

    results = {'J_star_min_M1': J_min, 'platform': {
        'L_cm': L_CM, 'kappa': KAPPA, 'N_breaths': N_BREATHS,
    }, 'bands': {}, 'lung_cancer_overlay': {}}

    for band_name, cfg in BANDS.items():
        print(f'=== Band: {band_name} ({cfg["title"]}) ===')
        cache = build_cache_with_interferents(band_name, cfg, panel0)

        scenarios = ['isolated']
        if cfg['interferent'] == 'H2O_CO2':
            scenarios.append('with_real_interferents')
        else:
            scenarios.append('with_synthetic_overlap')

        band_data = {'config': dict(cfg), 'scenarios': {}}
        for scen in scenarios:
            incl_real = (scen == 'with_real_interferents')
            incl_synth = (scen == 'with_synthetic_overlap')

            # --- 1) равномерная огибающая (K=5, без оптимизатора) ---
            sigma_arr, J_arr = [], []
            for alpha in ALPHA_PEAK_GRID:
                s, J = compute_sigma_i_for_alpha(
                    alpha, cfg, cache, noise,
                    include_real_interferents=incl_real,
                    include_synthetic_interferent=incl_synth,
                    panel0=panel0, panel1=panel1,
                )
                sigma_arr.append(s)
                J_arr.append(J)

            # --- 2) greedy-оптимизатор каналов (как для реальных биомаркеров) ---
            # Расстановка не зависит от α_peak в линейном по шуму режиме, поэтому
            # выбираем каналы один раз при опорном α, затем считаем σ_i по всей сетке.
            molecules_g, c_des_g, kappa_g = _inject_synthetic_biomarker(
                ALPHA_REF, cfg, cache,
                include_real_interferents=incl_real,
                include_synthetic_interferent=incl_synth,
            )
            greedy_centers = greedy_select_centers(
                cfg, molecules_g, c_des_g, kappa_g, cache, noise)
            sigma_g_arr, J_g_arr = [], []
            for alpha in ALPHA_PEAK_GRID:
                s, J = compute_sigma_i_for_alpha(
                    alpha, cfg, cache, noise,
                    include_real_interferents=incl_real,
                    include_synthetic_interferent=incl_synth,
                    panel0=panel0, panel1=panel1,
                    centers=greedy_centers,
                )
                sigma_g_arr.append(s)
                J_g_arr.append(J)

            band_data['scenarios'][scen] = {
                'alpha_peak': ALPHA_PEAK_GRID.tolist(),
                'sigma_i_ppb': sigma_arr,
                'J_possible_per_ppb2': J_arr,
                'sigma_i_ppb_greedy': sigma_g_arr,
                'J_possible_per_ppb2_greedy': J_g_arr,
                'greedy_centers_nm': [round(c, 1) for c in greedy_centers],
            }
            ratio = sigma_arr[12] / max(sigma_g_arr[12], 1e-30)
            print(f'  [{scen}] K=5: σ={sigma_arr[0]:.2g}..{sigma_arr[-1]:.2g} ppb | '
                  f'greedy K={len(greedy_centers)}: σ={sigma_g_arr[0]:.2g}..'
                  f'{sigma_g_arr[-1]:.2g} ppb (×{ratio:.2f} лучше)')
            print(f'        каналы greedy: '
                  f'{[round(c, 1) for c in greedy_centers]}')

        results['bands'][band_name] = band_data

    # Overlay
    print('\n=== Overlay биомаркеров рака лёгких ===')
    if overlay_db_ok:
        overlay = lung_cancer_overlay()
    else:
        cached = RESULTS_DIR / 'audit_envelope.json'
        overlay = json.loads(cached.read_text(encoding='utf-8')).get(
            'lung_cancer_overlay', {})
        print(f'  (overlay переиспользован из {cached.name})')
    for band, points in overlay.items():
        print(f'  {band}: {len(points)} точек')
        for p in points:
            print(f'    {p["name"]:<14s} α_peak = {p["alpha_peak"]:.2e}, σ_i = {p["sigma_i_ppb"]:.2g} ppb')
    results['lung_cancer_overlay'] = overlay

    # JSON
    out_path = RESULTS_DIR / 'audit_envelope.json'
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2,
                                     default=float), encoding='utf-8')
    print(f'\nJSON: {out_path.relative_to(REPO)}')

    # Plot
    make_plot(results, overlay, J_min)


def make_plot(results: dict, overlay: dict, J_min: float):
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    band_order = ['UV', 'Vis', 'NIR']

    delta_refs = [1.0, 3.0, 5.0]   # ppb
    delta_colors = ['#888888', '#555555', '#222222']

    for ax, band in zip(axes, band_order):
        bd = results['bands'][band]
        cfg = bd['config']
        for i, (scen, scen_data) in enumerate(bd['scenarios'].items()):
            alpha = np.asarray(scen_data['alpha_peak'])
            sigma = np.asarray(scen_data['sigma_i_ppb'])
            label = 'изолированный' if scen == 'isolated' else (
                'с H₂O+CO₂' if scen == 'with_real_interferents'
                else 'с интерферентом (ρ≈0,6)'
            )
            color = PALETTE['primary'] if scen == 'isolated' else PALETTE['warning']
            # Сплошная — равномерные K=5 (огибающая-характеристика платформы)
            ax.loglog(alpha, sigma, lw=2.0, marker='o', ms=4,
                       color=color, label=f'{label}: K=5 равном.')
            # Штриховая — greedy-оптимизатор каналов (та же расстановка, что для
            # реальных биомаркеров)
            sigma_g = scen_data.get('sigma_i_ppb_greedy')
            if sigma_g is not None:
                Kg = len(scen_data.get('greedy_centers_nm', []))
                ax.loglog(alpha, np.asarray(sigma_g), lw=1.6, ls='--',
                           marker='s', ms=3, color=color, alpha=0.85,
                           label=f'{label}: greedy (K={Kg})')

        # Горизонтальные пороги: σ_max = δ/√J_min для нескольких δ.
        # Три порога лежат в пределах ~0,7 декады у нижнего края, поэтому
        # подписи разносим по оси x (левая треть, где кривые ушли далеко
        # вверх и место свободно), а не ставим стопкой друг на друга.
        x_lo = ax.get_xlim()[0]
        for i, (d, c) in enumerate(zip(delta_refs, delta_colors)):
            sigma_threshold = d / np.sqrt(J_min)
            ax.axhline(sigma_threshold, color=c, ls='--', lw=1.2, alpha=0.8)
            x_label = x_lo * 10 ** (0.3 + 1.7 * i)
            ax.text(x_label, sigma_threshold * 1.15,
                     f'PASS δ={d:.0f} ppb',
                     fontsize=8.5, color=c, va='bottom', ha='left',
                     bbox=dict(boxstyle='round,pad=0.15', fc='white',
                               ec='none', alpha=0.7))

        # Overlay
        pts = overlay.get(band, [])
        if pts:
            ax.scatter([p['alpha_peak'] for p in pts],
                        [p['sigma_i_ppb'] for p in pts],
                        s=60, color='red', zorder=5, edgecolor='black',
                        linewidth=0.7, label='панель рака лёгких')
            for p in pts:
                ax.annotate(p['name'], (p['alpha_peak'], p['sigma_i_ppb']),
                              xytext=(6, 5), textcoords='offset points',
                              fontsize=9, color='red')

        if band == band_order[-1]:
            ax.set_xlabel(r'$\alpha_{\mathrm{peak}}$, cm²/molec', fontsize=11)
        ax.set_ylabel(r'$\sigma_i$, ppb (CRLB)', fontsize=11)
        ax.set_title(cfg['title'], fontsize=11)
        ax.grid(True, which='both', alpha=0.3)
        # Легенду — в верхний правый угол: кривые идут с наклоном −1
        # (из верх-лев в ниж-прав), поэтому верх-прав пуст, а ниж-лев занят
        # подписями PASS-порогов.
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        ax.set_xlim(1e-22, 1e-16)
        ax.set_ylim(1e-3, 1e9)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        f'Огибающая возможностей платформы (L={int(L_CM)}см, κ={int(KAPPA)}, N={N_BREATHS})',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    for out_dir in (FIG_DIR_REPO, FIG_DIR_LATEX):
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / 'audit_envelope.pdf'
        fig.savefig(out_path, bbox_inches='tight')
        print(f'PDF: {out_path}')
    plt.close(fig)


def replot_from_cache():
    """Быстрая перерисовка рисунка из results/audit_envelope.json без пересчёта."""
    cache_path = RESULTS_DIR / 'audit_envelope.json'
    results = json.loads(cache_path.read_text(encoding='utf-8'))
    overlay = results.get('lung_cancer_overlay', {})
    J_min = results.get('J_star_min_M1', compute_J_star_min(M=1))
    print(f'Перерисовка из кэша {cache_path.name} (J*_min = {J_min:.1f})')
    make_plot(results, overlay, J_min)


if __name__ == '__main__':
    if '--replot' in sys.argv:
        replot_from_cache()
    else:
        main()
