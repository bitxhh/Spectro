#!/usr/bin/env python
"""
scripts/audit_rescale_L25.py
============================
Пересчёт результатов полного аудита (scripts/audit_full.py) с длины
оптического пути текущей платформы L = 15 см на L = 25 см.

Обоснование точности пересчёта (а не повторного прогона):
  В линеаризованной модели наблюдаемая величина — оптическая глубина
  τ = α·c·κ·L — строго пропорциональна L. Поэтому матрица отклика A ∝ L,
  информация Фишера I ∝ L², маргинальная CRLB σ_i ∝ 1/L, а целевой
  скаляр J*_m = δ²·σ_b²/σ_i² ∝ L². Жадный выбор каналов K* и срабатывание
  плато-критерия (ε_stop = 0.05) инвариантны к глобальному множителю L²,
  поэтому K* НЕ меняется. Пересчёт сводится к точным множителям:
      J*_m         ×= (25/15)² = 2.77778
      sigma_ppb    ×= (15/25)  = 0.6
      delta_rel_%  ×= (15/25)  = 0.6      (∝ σ_i ∝ 1/L)
      K*, J*_min   — без изменений
      verdict      — пересчитывается по J*_m относительно J*_min
  Эквивалентно тому, что дал бы повторный прогон audit_full.py с
  L_CURRENT_CM = 25 (значение в скрипте уже изменено на 25).

Запуск:
    conda run -n spectro python scripts/audit_rescale_L25.py
(или любым python с matplotlib; init_db не требуется — сечения не считаются)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent          # spectrolib/
RESULTS = REPO / 'results' / 'audit_full_results.json'
BACKUP = REPO / 'results' / 'audit_full_results_L15_backup.json'
LATEX_FIG_DIR = REPO.parent / 'latex_doc' / 'figures'  # куда LaTeX берёт рисунок
FIG_NAME = 'audit_full_n_sweep_current.pdf'

L_OLD, L_NEW = 15.0, 25.0
F_J = (L_NEW / L_OLD) ** 2     # 2.77778
F_SIGMA = L_OLD / L_NEW        # 0.6  (σ_i ∝ 1/L)
F_DELTA = L_OLD / L_NEW        # 0.6  (relative uncertainty ∝ σ_i ∝ 1/L)


def verdict_of(J: float, J_min: float) -> str:
    if J >= 2.0 * J_min:
        return 'PASS+'
    if J >= J_min:
        return 'PASS'
    return 'FAIL'


def rescale_row(row: dict, J_min: float) -> dict:
    row['J_star_m'] = row['J_star_m'] * F_J
    if 'sigma_ppb' in row:
        row['sigma_ppb'] = row['sigma_ppb'] * F_SIGMA
    if 'delta_rel_pct' in row:
        row['delta_rel_pct'] = row['delta_rel_pct'] * F_DELTA
    if 'verdict' in row:
        row['verdict'] = verdict_of(row['J_star_m'], J_min)
    return row


def plot_n_sweep(n_data: dict, J_min: float, kappa: float, L_cm: float,
                 fwhm_uv: float, fwhm_nir: float, out_paths):
    """Реплика scripts/audit_full.py:plot_n_sweep без зависимости от
    spectrolib.plotstyle (чтобы не тянуть init_db). Тот же стиль кривых."""
    PALETTE = ['#2A4D69', '#C0504D', '#4F8A5B', '#E0A458', '#8C8C8C',
               '#7A5CA0', '#9CB4CC', '#6B6B6B', '#5577AA',
               '#AA7755', '#77AA55', '#5577AA', '#A050C0']
    fig, ax = plt.subplots(figsize=(11, 6.5))
    n_values = sorted(n_data.keys())
    biomarkers = [row['biomarker'] for row in n_data[n_values[0]]]
    for i, bm in enumerate(biomarkers):
        Js = []
        for N in n_values:
            for row in n_data[N]:
                if row['biomarker'] == bm:
                    Js.append(max(row['J_star_m'], 1e-10))
                    break
        ax.loglog(n_values, Js, marker='o', lw=1.8,
                  color=PALETTE[i % len(PALETTE)], label=bm)
    ax.axhline(J_min, color='red', linestyle='--',
               label=f'J*_min = {J_min:.0f}')
    ax.set_xlabel('Число выдохов N')
    ax.set_ylabel('J*_m')
    ax.set_title(f'Зависимость J*_m от N\n'
                 f'L={L_cm:g}см, κ={kappa:g}, FWHM(УФ)={fwhm_uv:g}нм, '
                 f'FWHM(БИК)={fwhm_nir:g}нм')
    ax.legend(fontsize=8, loc='best', framealpha=0.85, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches='tight')
        print(f'  Рисунок: {p}')
    plt.close(fig)


def main():
    data = json.loads(RESULTS.read_text(encoding='utf-8'))
    J_min = data['J_star_min_M1']

    if not BACKUP.exists():
        shutil.copy2(RESULTS, BACKUP)
        print(f'  Бэкап L=15: {BACKUP.name}')

    assert abs(data['current']['L_cm'] - L_OLD) < 1e-9, \
        f"current.L_cm = {data['current']['L_cm']} (ожидалось 15) — уже пересчитано?"

    # --- current ---
    data['current']['L_cm'] = L_NEW
    for row in data['current']['rows']:
        rescale_row(row, J_min)

    # --- n_sweep (тот же L текущей платформы) ---
    n_data_int = {}
    for N_str, rows in data['n_sweep'].items():
        for row in rows:
            rescale_row(row, J_min)
        n_data_int[int(N_str)] = rows

    # nextgen и fwhm_sweep уже при L=25 — не трогаем.

    # --- N_PASS (J*_m ∝ N точно): N_PASS = N·J_min / J*_m(N) ---
    def n_pass(bm: str) -> float:
        N0 = sorted(n_data_int)[-1]
        J0 = next(r['J_star_m'] for r in n_data_int[N0] if r['biomarker'] == bm)
        return N0 * J_min / J0

    np_sty = n_pass('styrene')
    np_iso = n_pass('isoprene')

    # --- рисунок ---
    cur = data['current']
    plot_n_sweep(
        n_data_int, J_min, cur['kappa'], L_NEW,
        cur['fwhm_uv_nm'], cur['fwhm_nir_nm'],
        out_paths=[LATEX_FIG_DIR / FIG_NAME, REPO / 'results' / FIG_NAME],
    )

    # --- сохранить пересчитанный JSON ---
    RESULTS.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=float),
        encoding='utf-8',
    )
    print(f'  JSON перезаписан: {RESULTS}')

    # --- печать новой таблицы текущей платформы ---
    print()
    print('=' * 78)
    print(f'  ТЕКУЩАЯ ПЛАТФОРМА ПОСЛЕ ПЕРЕСЧЁТА: L={L_NEW:g} см, '
          f'FWHM(УФ)={cur["fwhm_uv_nm"]:g} нм, FWHM(БИК)={cur["fwhm_nir_nm"]:g} нм, '
          f'κ={cur["kappa"]:g}, N={cur["N"]}')
    print('=' * 78)
    print(f'  {"Биомаркер":<14s}{"Band":<5s}{"K*":>3s}  {"σ,ppb":>12s}  '
          f'{"J*/J*_min":>11s}  Вердикт')
    rows_sorted = sorted(cur['rows'], key=lambda r: -r['J_star_m'])
    for r in rows_sorted:
        print(f'  {r["biomarker"]:<14s}{r["band"]:<5s}{r["K_star"]:>3d}  '
              f'{r["sigma_ppb"]:>12.4g}  {r["J_star_m"]/J_min:>11.4g}  '
              f'{r["verdict"]}')
    print()
    print(f'  N_PASS(стирол)  = {np_sty:.1f}  выдохов  (~{np_sty/3.0:.0f} мин)')
    print(f'  N_PASS(изопрен) = {np_iso:.1f}  выдохов  (~{np_iso/3.0:.0f} мин)')
    print()
    # точные пересчитанные отношения для правки LaTeX-таблицы
    print('  Точные J*/J*_min (для LaTeX):')
    for r in rows_sorted:
        print(f'    {r["biomarker"]:<14s}{r["band"]:<5s} '
              f'{r["J_star_m"]/J_min:.6e}  σ={r["sigma_ppb"]:.6g}')


if __name__ == '__main__':
    main()
