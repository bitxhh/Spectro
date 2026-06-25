#!/usr/bin/env python
"""
scripts/audit_full.py
=====================
Полный аудит всех 11 биомаркеров рака лёгких в их оптимальных спектральных
поддиапазонах. Объединяет цели глав 4 (аудит) и 5 (рекомендации) диплома.

Аппаратные сценарии (длина оптического пути зафиксирована L = 25 см
во всей работе; платформы различаются только FWHM и числом выдохов N):
  - CURRENT  : L = 25 см, FWHM поддиапазонная (УФ 25 нм, БИК 150 нм),
               κ = 1000, N = 6...30.
  - NEXTGEN  : L = 25 см, узкие QD-каналы FWHM = ?, свип по FWHM, N = 20.

Биомаркеры группированы по подDIAP:
  - UV блок: 9 биомаркеров (ароматики + карбонилы + изопрен)
  - NIR блок: 5 биомаркеров (пентан, гексан, гексаналь, изопрен, HCHO)
    + интерференты H2O, CO2 с κ=1 (не преконцентрируются сорбентом)

Запуск:
    conda run -n spectro python scripts/audit_full.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent  # корень репозитория spectrolib
sys.path.insert(0, str(REPO.parent))  # чтобы `from spectrolib import ...` работало без pip install

from spectrolib import MixturePanel, NoiseModel, mpi, hitran_xsc
mpi.init_db(str(REPO / 'mpi_data'))
hitran_xsc.init_db(str(REPO / 'hitran_xsc_data'))

from spectrolib.audit import (
    LocalBlock, audit_block, compute_J_star_min,
)
from spectrolib.plotstyle import save as save_figure, new_figure, PALETTE


PANEL_MU0 = REPO / 'example_panels/lung_cancer_control_mu0.yaml'
PANEL_MU1 = REPO / 'example_panels/lung_cancer_disease_mu1.yaml'
NOISE_FILE = REPO / 'example_noise_models/table_2_6.yaml'
RESULTS_DIR = REPO / 'results'
FIG_DIR = RESULTS_DIR  # PDF-графики тоже сохраняем сюда, всё в одном каталоге

# Параметры по разделу 4.2 диплома
KAPPA = 1000.0          # реалистичный максимум сорбентов
L_CURRENT_CM = 25.0     # текущая платформа (длина кюветы зафиксирована 25 см)
L_NEXTGEN_CM = 25.0     # перспективная платформа (та же кювета)

# Поддиапазонные FWHM по таблице 2.5 (раздел subsec:qd-platform)
FWHM_UV_CURRENT = 25.0
FWHM_VIS_CURRENT = 30.0
FWHM_NIR_CURRENT = 150.0


CV = {
    'benzene': 0.5, 'toluene': 0.5, 'ethylbenzene': 0.5, 'styrene': 0.5,
    'HCHO': 0.4, 'acetone': 0.3, 'butanone': 0.4, 'hexanal': 0.4,
    'pentane': 0.4, 'hexane': 0.4, 'isoprene': 0.3,
    'H2O': 0.1, 'CO2': 0.1, 'O2': 0.1, 'ethanol': 0.4,
}

# Источники сечений: УФ-биомаркеры через MPI, БИК через HITRAN-XSC,
# вода/углекислый газ через HITRAN line-by-line.
SRC_UV = {m: ['mpi'] for m in ('HCHO', 'acetone', 'butanone', 'hexanal',
                                'isoprene', 'benzene', 'toluene',
                                'ethylbenzene', 'styrene')}
SRC_NIR_VOC = {m: ['hitran_xsc'] for m in
               ('pentane', 'hexane', 'isoprene', 'acetone',
                'butanone', 'hexanal', 'ethanol')}
SRC_NIR_VOC['HCHO'] = ['hitran']      # HCHO в HITRAN line-by-line
SRC_NIR_INTERF = {'H2O': ['hitran'], 'CO2': ['hitran'], 'O2': ['hitran']}

# Селективное преконцентрирование: сорбенты пропускают воду/CO2/O2
KAPPA_NATURAL = {'H2O': 1.0, 'CO2': 1.0, 'O2': 1.0}


# Биомаркеры и их рабочие диапазоны (один основной за биомаркер)
UV_BIOMARKERS: List[Tuple[str, Tuple[float, float], List[str]]] = [
    # (имя, спектральный блок, главные интерференты в блоке)
    ('benzene',      (235.0, 270.0), ['toluene']),
    ('toluene',      (215.0, 245.0), ['benzene', 'ethylbenzene']),
    ('ethylbenzene', (245.0, 280.0), ['benzene', 'toluene']),
    ('styrene',      (220.0, 260.0), ['benzene']),
    ('HCHO',         (270.0, 340.0), ['acetone']),
    ('acetone',      (260.0, 320.0), ['HCHO', 'isoprene']),
    ('butanone',     (255.0, 295.0), ['acetone']),
    ('hexanal',      (260.0, 305.0), ['acetone']),
    ('isoprene',     (250.0, 280.0), ['acetone']),
]

NIR_BIOMARKERS: List[Tuple[str, Tuple[float, float], List[str]]] = [
    # Пентан/гексан: CH-обертоны в районе 1700, 2200, 2300 нм
    # H2O имеет полосы 1380, 1870, 2700 нм; CO2 — 1600, 2000, 2700 нм
    # Блок 2150-2400 нм относительно чистый от воды
    ('pentane',  (2150.0, 2400.0), ['hexane', 'H2O', 'CO2']),
    ('hexane',   (2150.0, 2400.0), ['pentane', 'H2O', 'CO2']),
    ('isoprene', (2150.0, 2400.0), ['pentane', 'hexane', 'H2O', 'CO2']),
    # HCHO в БИК ~2194 нм — слабее чем УФ, но проверим
    ('HCHO',     (2100.0, 2300.0), ['H2O', 'CO2']),
]


def make_block(name, wl_range, targets, interferents,
               L_cm, fwhm_nm, kappa, N,
               sources_override, kappa_per_mol=None,
               min_dist_factor=0.3, drift_degree=1):
    mols = list(targets) + list(interferents)
    return LocalBlock(
        name=name,
        wavelength_range_nm=wl_range,
        targets=list(targets), interferents=list(interferents),
        cv_table={k: CV[k] for k in mols},
        fwhm_nm=fwhm_nm, L_cm=L_cm, kappa=kappa, N_breaths=N,
        drift_degree=drift_degree,
        min_dist_factor=min_dist_factor,
        p_lo=0.30,
        sources_override=sources_override,
        kappa_per_molecule=kappa_per_mol,
    )


def run_one(name, wl_range, targets, interferents, *,
            L_cm, fwhm_nm, kappa, N, src, kappa_per_mol=None,
            drift_degree=1):
    panel0 = MixturePanel.from_file(PANEL_MU0)
    panel1 = MixturePanel.from_file(PANEL_MU1)
    noise = NoiseModel.from_file(NOISE_FILE)

    # min_dist подбираем под FWHM
    min_dist_factor = 0.3 if fwhm_nm >= 10 else (0.5 if fwhm_nm >= 3 else 1.0)
    block = make_block(
        f'{name}_L{L_cm}_F{fwhm_nm}_k{kappa}_N{N}',
        wl_range, targets, interferents,
        L_cm=L_cm, fwhm_nm=fwhm_nm, kappa=kappa, N=N,
        sources_override=src, kappa_per_mol=kappa_per_mol,
        min_dist_factor=min_dist_factor,
        drift_degree=drift_degree,
    )
    J_min = compute_J_star_min(M=len(targets))
    r = audit_block(
        block, panel0, panel1, noise,
        J_star_min=J_min, max_K=30, verbose=False,
        grid_sampling_nm=min(0.5, fwhm_nm / 4),
    )
    return r, J_min


def biomarker_set(L_cm, fwhm_uv, fwhm_nir, kappa, N):
    """
    Возвращает массив строк-результатов аудита по всем 11 биомаркерам
    для данных аппаратных параметров.
    """
    rows = []
    print(f'  [биомаркер-сет L={L_cm}, F_UV={fwhm_uv}, F_NIR={fwhm_nir}, '
          f'κ={kappa}, N={N}]', flush=True)
    # UV
    for name, wl, interf in UV_BIOMARKERS:
        r, J_min = run_one(name, wl, [name], interf,
                            L_cm=L_cm, fwhm_nm=fwhm_uv,
                            kappa=kappa, N=N, src=SRC_UV)
        print(f'    UV {name:<14s}: K*={r.K_star:>2d}, '
              f'σ={r.sigma_i[name]*1000:>9.3g} ppb, '
              f'J*={r.J_star_m[name]:>9.3g}, {r.verdict[name]}',
              flush=True)
        rows.append({
            'biomarker': name, 'band': 'UV', 'wl_nm': wl,
            'K_star': r.K_star,
            'sigma_ppb': r.sigma_i[name] * 1000,
            'J_star_m': r.J_star_m[name],
            'J_star_min': J_min,
            'delta_rel_pct': r.delta_rel[name] * 100,
            'verdict': r.verdict[name],
        })
    # NIR
    for name, wl, interf in NIR_BIOMARKERS:
        src = {**SRC_NIR_VOC, **SRC_NIR_INTERF}
        kp = {k: v for k, v in KAPPA_NATURAL.items() if k in interf}
        r, J_min = run_one(name, wl, [name], interf,
                            L_cm=L_cm, fwhm_nm=fwhm_nir,
                            kappa=kappa, N=N, src=src, kappa_per_mol=kp)
        print(f'    NIR {name:<13s}: K*={r.K_star:>2d}, '
              f'σ={r.sigma_i[name]*1000:>9.3g} ppb, '
              f'J*={r.J_star_m[name]:>9.3g}, {r.verdict[name]}',
              flush=True)
        rows.append({
            'biomarker': name + '_NIR', 'band': 'NIR', 'wl_nm': wl,
            'K_star': r.K_star,
            'sigma_ppb': r.sigma_i[name] * 1000,
            'J_star_m': r.J_star_m[name],
            'J_star_min': J_min,
            'delta_rel_pct': r.delta_rel[name] * 100,
            'verdict': r.verdict[name],
        })
    return rows


def print_table(rows, title):
    print()
    print('=' * 80)
    print(f'  {title}')
    print('=' * 80)
    print(f'  {"Биомаркер":<18s} {"Band":<5s} {"K*":>3s} {"σ ppb":>11s} '
          f'{"δ_rel %":>9s} {"J*_m":>11s}  Вердикт')
    for row in rows:
        ratio = row['J_star_m'] / row['J_star_min']
        print(f'  {row["biomarker"]:<18s} {row["band"]:<5s} '
              f'{row["K_star"]:>3d} '
              f'{row["sigma_ppb"]:>11.4g} '
              f'{row["delta_rel_pct"]:>9.2f} '
              f'{row["J_star_m"]:>11.4g}  {row["verdict"]} '
              f'(×{ratio:.2g} от порога)')


def fwhm_sweep(L_cm, kappa, N, fwhms_nm):
    """
    Свип по FWHM для всех 9 УФ-биомаркеров.
    NIR-сценарий вынесен отдельно (FWHM=150 нм фикс., нет смысла менять).
    """
    print()
    print('=' * 80)
    print(f'  FWHM-свип (УФ-биомаркеры): L={L_cm}см, κ={kappa}, N={N}')
    print(f'  FWHMs = {fwhms_nm}')
    print('=' * 80)
    data = {m: [] for m, _, _ in UV_BIOMARKERS}
    for fwhm in fwhms_nm:
        for name, wl, interf in UV_BIOMARKERS:
            r, J_min = run_one(name, wl, [name], interf,
                                L_cm=L_cm, fwhm_nm=fwhm,
                                kappa=kappa, N=N, src=SRC_UV)
            data[name].append({
                'fwhm': fwhm,
                'J_star_m': r.J_star_m[name],
                'J_star_min': J_min,
                'sigma_ppb': r.sigma_i[name] * 1000,
                'verdict': r.verdict[name],
            })
        print(f'  FWHM={fwhm:>5.1f}нм:  ' +
              ' '.join(f'{n[:4]}={d[-1]["J_star_m"]:>8.2g}'
                       for n, d in data.items()))
    return data


def plot_fwhm_sweep(data, kappa, L_cm, N, J_min, filename):
    fig, ax = new_figure(figsize=(11, 6.5))
    cycle = [PALETTE['primary'], PALETTE['accent'], PALETTE['success'],
             PALETTE['warning'], PALETTE['muted'], PALETTE['highlight'],
             PALETTE['soft'], PALETTE['neutral'], '#5577AA']
    for i, (name, traces) in enumerate(data.items()):
        fwhms = [t['fwhm'] for t in traces]
        Js = [max(t['J_star_m'], 1e-10) for t in traces]
        ax.loglog(fwhms, Js, marker='o', lw=1.8,
                  color=cycle[i % len(cycle)], label=name)
    ax.axhline(J_min, color='red', linestyle='--', lw=1.2,
               label=f'J*_min ≈ {J_min:.0f}')
    ax.set_xlabel('FWHM канала, нм')
    ax.set_ylabel('J*_m')
    ax.set_title(f'Зависимость J*_m от FWHM по 9 УФ-биомаркерам\n'
                 f'L={L_cm}см, κ={kappa}, N={N}, single-biomarker аудит')
    ax.legend(fontsize=8, loc='best', framealpha=0.85, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    save_figure(ax, FIG_DIR / filename)
    plt.close(fig)
    print(f'  Сохранено: latex_doc/figures/{filename}')


def n_sweep(L_cm, kappa, fwhm_uv, fwhm_nir, n_values):
    """Свип по N с фикс. FWHM поддиапазонов."""
    print()
    print('=' * 80)
    print(f'  N-свип: L={L_cm}см, κ={kappa}, FWHM(УФ)={fwhm_uv}, '
          f'FWHM(БИК)={fwhm_nir}')
    print(f'  N = {n_values}')
    print('=' * 80)
    data = {}
    for N in n_values:
        rows = biomarker_set(L_cm, fwhm_uv, fwhm_nir, kappa, N)
        data[N] = rows
    return data


def plot_n_sweep(data, J_min, kappa, L_cm, fwhm_uv, fwhm_nir, filename):
    fig, ax = new_figure(figsize=(11, 6.5))
    # По каждой молекуле — линия J* vs N
    n_values = sorted(data.keys())
    biomarkers = [row['biomarker'] for row in data[n_values[0]]]
    cycle = [PALETTE['primary'], PALETTE['accent'], PALETTE['success'],
             PALETTE['warning'], PALETTE['muted'], PALETTE['highlight'],
             PALETTE['soft'], PALETTE['neutral'], '#5577AA',
             '#AA7755', '#77AA55', '#5577AA', '#A050C0']
    for i, bm in enumerate(biomarkers):
        Js = []
        for N in n_values:
            for row in data[N]:
                if row['biomarker'] == bm:
                    Js.append(max(row['J_star_m'], 1e-10))
                    break
        ax.loglog(n_values, Js, marker='o', lw=1.8,
                  color=cycle[i % len(cycle)], label=bm)
    ax.axhline(J_min, color='red', linestyle='--',
               label=f'J*_min = {J_min:.0f}')
    ax.set_xlabel('Число выдохов N')
    ax.set_ylabel('J*_m')
    ax.set_title(f'Зависимость J*_m от N\n'
                 f'L={L_cm}см, κ={kappa}, FWHM(УФ)={fwhm_uv}нм, '
                 f'FWHM(БИК)={fwhm_nir}нм')
    ax.legend(fontsize=8, loc='best', framealpha=0.85, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    save_figure(ax, FIG_DIR / filename)
    plt.close(fig)
    print(f'  Сохранено: latex_doc/figures/{filename}')


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    J_min_M1 = compute_J_star_min(M=1)
    print(f'J*_min(M=1, τ*=0.6, ε=0.05) = {J_min_M1:.1f}')

    # === 1. Текущая платформа: фикс. FWHM, N=6 ===
    current = biomarker_set(
        L_cm=L_CURRENT_CM, fwhm_uv=FWHM_UV_CURRENT,
        fwhm_nir=FWHM_NIR_CURRENT, kappa=KAPPA, N=6,
    )
    print_table(current,
                f'ТЕКУЩАЯ ПЛАТФОРМА: L={L_CURRENT_CM}см, '
                f'FWHM(УФ)={FWHM_UV_CURRENT}нм, FWHM(БИК)={FWHM_NIR_CURRENT}нм, '
                f'κ={KAPPA}, N=6')

    # === 2. N-свип на текущей платформе ===
    n_data = n_sweep(L_cm=L_CURRENT_CM, kappa=KAPPA,
                      fwhm_uv=FWHM_UV_CURRENT, fwhm_nir=FWHM_NIR_CURRENT,
                      n_values=[6, 10, 15, 20, 30])
    plot_n_sweep(n_data, J_min_M1, KAPPA, L_CURRENT_CM,
                  FWHM_UV_CURRENT, FWHM_NIR_CURRENT,
                  'audit_full_n_sweep_current.pdf')

    # === 3. FWHM-свип на перспективной платформе (L=25, N=20) ===
    fwhms = [1.0, 2.0, 5.0, 10.0, 15.0, 25.0]
    fwhm_data = fwhm_sweep(L_cm=L_NEXTGEN_CM, kappa=KAPPA, N=20,
                            fwhms_nm=fwhms)
    plot_fwhm_sweep(fwhm_data, KAPPA, L_NEXTGEN_CM, 20, J_min_M1,
                     'audit_full_fwhm_sweep_nextgen.pdf')

    # === 4. Перспективная платформа: тонкие каналы, N=20 ===
    nextgen = biomarker_set(
        L_cm=L_NEXTGEN_CM, fwhm_uv=2.0, fwhm_nir=30.0,
        kappa=KAPPA, N=20,
    )
    print_table(nextgen,
                f'ПЕРСПЕКТИВНАЯ ПЛАТФОРМА: L={L_NEXTGEN_CM}см, '
                f'FWHM(УФ)=2нм, FWHM(БИК)=30нм, κ={KAPPA}, N=20')

    # === Сохранение JSON ===
    out = {
        'current': {
            'L_cm': L_CURRENT_CM, 'fwhm_uv_nm': FWHM_UV_CURRENT,
            'fwhm_nir_nm': FWHM_NIR_CURRENT, 'kappa': KAPPA, 'N': 6,
            'rows': current,
        },
        'nextgen': {
            'L_cm': L_NEXTGEN_CM, 'fwhm_uv_nm': 2.0,
            'fwhm_nir_nm': 30.0, 'kappa': KAPPA, 'N': 20,
            'rows': nextgen,
        },
        'n_sweep': {str(N): rows for N, rows in n_data.items()},
        'fwhm_sweep': fwhm_data,
        'J_star_min_M1': J_min_M1,
    }
    out_path = RESULTS_DIR / 'audit_full_results.json'
    out_path.write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8',
    )
    print()
    print(f'  JSON: {out_path.relative_to(REPO)}')

    # === Итоговое сравнение ===
    print()
    print('=' * 80)
    print('  СРАВНЕНИЕ ТЕКУЩАЯ vs ПЕРСПЕКТИВНАЯ')
    print('=' * 80)
    print(f'  {"Биомаркер":<18s}  Текущая (L=25, F=25/150)  vs  Перспективная (L=25, F=2/30)')
    for row_c, row_n in zip(current, nextgen):
        rc = row_c['J_star_m'] / row_c['J_star_min']
        rn = row_n['J_star_m'] / row_n['J_star_min']
        print(f'  {row_c["biomarker"]:<18s}  '
              f'J*/J*_min={rc:>10.3g} ({row_c["verdict"]:<5})  '
              f'→  {rn:>10.3g} ({row_n["verdict"]})')


if __name__ == '__main__':
    main()
