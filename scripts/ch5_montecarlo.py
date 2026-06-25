#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/ch5_montecarlo.py
=========================

Глава 5 диплома — Монте-Карло-валидация информационного аудита (глава 4).

Идея
----
Глава 4 даёт аналитический прогноз: при гибридной архитектуре спектрометра
(две конфигурации каналов) и усреднении N выдохов достижимая точность оценки
концентрации биомаркера выходит на нижнюю границу Крамера–Рао (CRLB), а
сравнение CRLB с проектным порогом J*_min даёт вердикт PASS+/PASS/FAIL.

Этот скрипт проверяет прогноз прямым моделированием:

  1. Берёт РЕАЛЬНЫЕ сечения поглощения (MPI) для четырёх веществ —
     толуол, бензол, стирол, изопрен — и UV-интерферентов из панели.
  2. Воспроизводит ДВИЖОК аудита главы 4 (`spectrolib.audit.audit_block`)
     ровно с теми же поддиапазонами, интерферентами, FWHM и κ, что и в
     численной программе scripts/audit_full.py (таблица tab:audit-nextgen-best),
     но при эксплуатационном N = 35 выдохов. Аудит выбирает жадной forward-
     процедурой оптимальные каналы и даёт аналитический CRLB σ_i и показатель
     информации J*_m с вердиктом.
  3. Поверх этого строит нелинейную прямую модель измерения OD = -ln T
     со стационарным шумом (тепловой + дробовой + цветной) и дрейфом базовой
     линии, применяет обеляющий фильтр (Холецкий истинной C_OD) и полную
     GLS/ML-оценку расширенного плана [A | B] (молекулы + дрейф) к N_MC
     реализациям, каждая из которых — среднее N выдохов.
  4. Показывает, что фильтрация достигает CRLB (σ_MC -> σ_CRLB), пересчитывает
     дисперсию в ширину 95% доверительного интервала и сравнивает с целевой
     точностью σ_target = δ_m / sqrt(J*_min): вердикт и расстояние до цели
     sqrt(J*_min / J*_m) = σ_i / σ_target.

Две конфигурации каналов (итоги аудита, гл. 4):
  - узкая  (FWHM = 2 нм)  — узкие QD-каналы для острых УФ-полос толуол/бензол;
  - широкая (FWHM = 25 нм) — стандартные каналы для широких полос стирол/изопрен.
Каждый биомаркер аудируется в своём поддиапазоне против локально перекрывающих
интерферентов — в точности как в численном аудите главы 4.

Запуск (на этой машине, окружение conda `spectro`, UTF-8):
    set PYTHONIOENCODING=utf-8 & set PYTHONUTF8=1
    conda run --no-capture-output -n spectro python scripts/ch5_montecarlo.py

Воспроизводимость: результаты -> results/ch5_montecarlo_results.json,
рисунки -> results/ и latex_doc/figures/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Пути и импорт библиотеки
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO.parent))

from spectrolib import MixturePanel, NoiseModel, mpi, hitran_xsc  # noqa: E402
from spectrolib.audit import (  # noqa: E402
    LocalBlock,
    audit_block,
    compute_J_star_min,
    compute_J_star_m,
    population_params,
    _build_block_cache,
    build_response_matrix_from_cache,
    build_noise_covariance_OD,
    build_drift_basis,
    marginal_fisher,
)

# scipy для треугольных решений (обеляющий фильтр)
from scipy.linalg import solve_triangular  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---------------------------------------------------------------------------
# Файлы данных / выходные каталоги
# ---------------------------------------------------------------------------
PANEL_MU0 = REPO / "example_panels/lung_cancer_control_mu0.yaml"
PANEL_MU1 = REPO / "example_panels/lung_cancer_disease_mu1.yaml"
NOISE_FILE = REPO / "example_noise_models/table_2_6.yaml"
MPI_DATA_DIR = r"C:/Users/ilya1/Documents/diploma/mpi_data"

RESULTS_DIR = REPO / "results"
FIG_DIR_LATEX = REPO.parent / "latex_doc" / "figures"

# ---------------------------------------------------------------------------
# Физические константы платформы (как в аудите, глава 4)
# ---------------------------------------------------------------------------
L_CM = 25.0          # длина оптического пути преконцентратора, см
KAPPA = 1000.0       # коэффициент преконцентрирования VOC
N_BREATHS = 35       # число усредняемых выдохов (решение для главы 5)
DRIFT_DEG = 1        # степень полиномиального базиса дрейфа (offset + наклон)
N_MC = 6000          # число Монте-Карло реализаций

# Амплитуда дрейфа базовой линии (СКО коэффициентов B), в единицах OD.
# Дрейф лежит в span(B) и точно снимается GLS-оценкой [A|B]; для «наивной»
# оценки (только A) он остаётся и раздувает дисперсию — это демонстрируется.
DRIFT_SIGMA = np.array([3.0e-3, 3.0e-3])  # [offset, наклон]

# Жадный выбор каналов (как в audit_block / audit_full)
EPS_STOP = 0.05
MAX_K = 30

# 95% двусторонний доверительный интервал
Z95 = 1.959963984540054

RNG_SEED = 20260609

# ---------------------------------------------------------------------------
# Две конфигурации каналов и четыре аудита (итоги главы 4)
# ---------------------------------------------------------------------------
# Окна и интерференты совпадают с численным аудитом scripts/audit_full.py
# (UV_BIOMARKERS) -> таблица tab:audit-nextgen-best. FWHM узкой конфигурации
# взят 2 нм (технологичный QD-канал, чуть шире 1-нм теоретического оптимума).
FWHM_NARROW = 2.0
FWHM_WIDE = 25.0

AUDITS = [
    dict(target="toluene", ru="толуол", config="narrow", fwhm=FWHM_NARROW,
         window=(215.0, 245.0), interferents=["benzene", "ethylbenzene"]),
    dict(target="benzene", ru="бензол", config="narrow", fwhm=FWHM_NARROW,
         window=(235.0, 270.0), interferents=["toluene"]),
    dict(target="styrene", ru="стирол", config="wide", fwhm=FWHM_WIDE,
         window=(220.0, 260.0), interferents=["benzene"]),
    dict(target="isoprene", ru="изопрен", config="wide", fwhm=FWHM_WIDE,
         window=(250.0, 280.0), interferents=["acetone"]),
]

# Коэффициенты вариации логнормали популяции (таблица 2.2 / audit_full)
CV_TABLE = {
    "benzene": 0.5, "toluene": 0.5, "ethylbenzene": 0.5, "styrene": 0.5,
    "acetone": 0.3, "isoprene": 0.3, "HCHO": 0.4, "butanone": 0.4,
    "hexanal": 0.4,
}

# Форсируем единственный источник сечений — MPI. В панели у VOC прописано
# [mpi, hitran_xsc], но БД hitran_xsc на этой машине отсутствует, а УФ-сечения
# (200–360 нм), на которых работает аудит, целиком берутся из MPI. Без этого
# мультиисточниковая генерация падает на втором источнике и обнуляет столбец.
def _sources_override(molecules: Sequence[str]) -> Dict[str, List[str]]:
    return {m: ["mpi"] for m in molecules}


def _audit_hparams(fwhm: float):
    """Шаг тонкой сетки и min_dist_factor — те же правила, что в audit_full.py."""
    grid_sampling = min(0.5, fwhm / 4.0)
    min_dist_factor = 0.3 if fwhm >= 10 else (0.5 if fwhm >= 3 else 1.0)
    return grid_sampling, min_dist_factor


# ===========================================================================
# Монте-Карло: обеляющий фильтр + GLS-оценка [A | B]
# ===========================================================================
def _safe_cholesky(C: np.ndarray) -> np.ndarray:
    C = 0.5 * (C + C.T)
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        jit = 1e-15 * float(np.trace(C)) / C.shape[0]
        return np.linalg.cholesky(C + jit * np.eye(C.shape[0]))


def run_mc(centers, fwhm, molecules, c_design, cache, noise, rng,
           n_mc=N_MC) -> Dict[str, object]:
    """
    Прямое моделирование оценки концентрации таргета.

    Истина задана в расчётной точке (нулевое отклонение), поэтому оценка
    theta_hat[0] напрямую является ОШИБКОЙ оценки; её эмпирическое СКО есть
    sigma_MC, которое должно совпасть с CRLB. Дополнительно пересчитывается
    аналитический CRLB через маргинальную информацию Фишера на ТЕХ ЖЕ каналах
    — для сверки с движком аудита (главой 4).
    """
    centers = sorted(centers)
    cen = np.asarray(centers, dtype=float)
    K = len(cen)
    A, T_design, _ = build_response_matrix_from_cache(
        centers_nm=centers, fwhm_nm=fwhm, molecules=molecules,
        c_design=c_design, kappa=KAPPA, cache=cache,
    )
    C_OD = build_noise_covariance_OD(T_design, cen, noise)   # на 1 выдох
    B = build_drift_basis(cen, degree=DRIFT_DEG)
    C_T = C_OD * np.outer(T_design, T_design)               # ковариация в T

    # --- аналитический CRLB на этих же каналах (сверка с аудитом) ---
    I_marg = marginal_fisher(A, C_OD, B, N_breaths=N_BREATHS)
    _, sigma_i_ppm = compute_J_star_m(I_marg, m_idx=0, delta_m=1.0,
                                      sigma_b_m=1.0)
    sigma_crlb_analytic_ppb = float(sigma_i_ppm * 1000.0)

    L_T = _safe_cholesky(C_T)
    L_OD = _safe_cholesky(C_OD)
    OD_ref = -np.log(np.clip(T_design, 1e-12, None))

    # --- генерация шума на уровне каналов (T-пространство) ---
    Z = rng.standard_normal((K, N_BREATHS * n_mc))
    eps_T = L_T @ Z                                         # (K, N*n_mc)
    T_meas = np.clip(T_design[:, None] + eps_T, 1e-9, None)
    OD = -np.log(T_meas)                                    # нелинейность -lnT
    OD = OD.reshape(K, N_BREATHS, n_mc)
    OD_bar = OD.mean(axis=1)                                # усреднение N выдохов
    z = OD_bar - OD_ref[:, None]                            # (K, n_mc)

    # --- дрейф базовой линии в span(B) ---
    theta_d = rng.standard_normal((B.shape[1], n_mc)) * DRIFT_SIGMA[:, None]
    z = z + B @ theta_d

    # --- обеляющий фильтр (Холецкий истинной C_OD) ---
    zw = solve_triangular(L_OD, z, lower=True)

    # --- полная GLS/ML-оценка расширенного плана [A | B] ---
    G = np.column_stack([A, B])
    Gw = solve_triangular(L_OD, G, lower=True)
    P_full = np.linalg.pinv(Gw)
    theta_full = P_full @ zw
    err_full = theta_full[0] * 1000.0                       # ppb

    # --- «наивная» оценка без модели дрейфа (только A) ---
    Aw = solve_triangular(L_OD, A, lower=True)
    P_naive = np.linalg.pinv(Aw)
    theta_naive = P_naive @ zw
    err_naive = theta_naive[0] * 1000.0                     # ppb

    return dict(
        sigma_crlb_analytic_ppb=sigma_crlb_analytic_ppb,
        sigma_full_ppb=float(np.std(err_full, ddof=1)),
        bias_full_ppb=float(np.mean(err_full)),
        sigma_naive_ppb=float(np.std(err_naive, ddof=1)),
        bias_naive_ppb=float(np.mean(err_naive)),
        err_full=err_full,
        n_mc=int(n_mc),
        K=K,
    )


# ===========================================================================
# main
# ===========================================================================
def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR_LATEX.mkdir(parents=True, exist_ok=True)

    # --- инициализация баз сечений (реальный каталог MPI) ---
    try:
        mpi.init_db(MPI_DATA_DIR)
        print(f"[init_db] MPI OK: {MPI_DATA_DIR}")
    except Exception as e:  # noqa: BLE001
        print(f"[init_db] MPI FAIL: {e!r}")
        sys.exit(1)
    try:
        hitran_xsc.init_db(str(REPO / "hitran_xsc_data"))
    except Exception:  # noqa: BLE001
        pass

    panel0 = MixturePanel.from_file(PANEL_MU0)
    panel1 = MixturePanel.from_file(PANEL_MU1)
    noise = NoiseModel.from_file(NOISE_FILE)
    J_min = compute_J_star_min(M=1)
    print(f"[J*_min] (M=1, tau*=0.6, eps=0.05) = {J_min:.1f}")

    rng = np.random.default_rng(RNG_SEED)

    results: List[Dict[str, object]] = []
    for au in AUDITS:
        target = au["target"]
        fwhm = au["fwhm"]
        window = au["window"]
        interf = list(au["interferents"])
        molecules = [target] + interf            # таргет — индекс 0
        grid_sampling, min_dist_factor = _audit_hparams(fwhm)
        src = _sources_override(molecules)

        print(f"\n=== Аудит: {au['ru']} ({target}) | конфиг {au['config']} "
              f"FWHM={fwhm:g} нм | окно {window} нм ===")
        print(f"    интерференты: {interf}")

        # --- движок аудита главы 4 (тот же, что в audit_full.py), но N=35 ---
        block = LocalBlock(
            name=f"{target}_F{fwhm:g}_N{N_BREATHS}",
            wavelength_range_nm=window,
            targets=[target], interferents=interf,
            cv_table={m: CV_TABLE[m] for m in molecules},
            fwhm_nm=fwhm, L_cm=L_CM, kappa=KAPPA, N_breaths=N_BREATHS,
            drift_degree=DRIFT_DEG, min_dist_factor=min_dist_factor,
            p_lo=0.30, sources_override=src,
        )
        res = audit_block(
            block, panel0, panel1, noise,
            J_star_min=J_min, epsilon_stop=EPS_STOP, max_K=MAX_K,
            grid_sampling_nm=grid_sampling, verbose=False,
        )
        centers = [float(c) for c in res.lambdas_star]
        sigma_crlb_ppb = float(res.sigma_i[target] * 1000.0)
        J_star_m = float(res.J_star_m[target])
        verdict = res.verdict[target]
        c_design = res.c_design
        ratio = J_star_m / J_min

        # популяционные параметры таргета (для целевого σ и ДИ)
        pp = population_params([target], panel0, panel1, CV_TABLE)
        delta_ppm, sigma_b_ppm = pp[target]
        delta_ppb = float(delta_ppm * 1000.0)
        sigma_target_ppb = delta_ppb / float(np.sqrt(J_min))
        distance = float(np.sqrt(J_min / J_star_m))   # sigma_i / sigma_target
        ci_crlb_ppb = 2.0 * Z95 * sigma_crlb_ppb
        ci_target_ppb = 2.0 * Z95 * sigma_target_ppb

        print(f"    K* = {len(centers)} каналов: "
              f"{[round(c, 1) for c in centers]}")
        print(f"    c_design(таргет) = {c_design[target]*1000:.3f} ppb, "
              f"delta = {delta_ppb:.3f} ppb")

        # --- Монте-Карло на тех же каналах ---
        cache = _build_block_cache(
            wavelength_range_nm=window, molecules=molecules,
            panel_template=panel0, L_cm=L_CM,
            grid_sampling_nm=grid_sampling, sources_override=src,
        )
        mc = run_mc(centers, fwhm, molecules, c_design, cache, noise, rng)
        sigma_mc_ppb = mc["sigma_full_ppb"]
        ci_mc_ppb = 2.0 * Z95 * sigma_mc_ppb
        mc_over_crlb = (sigma_mc_ppb / sigma_crlb_ppb
                        if sigma_crlb_ppb > 0 else float("nan"))
        naive_inflation = (mc["sigma_naive_ppb"] / sigma_mc_ppb
                           if sigma_mc_ppb > 0 else float("nan"))
        crlb_match = (mc["sigma_crlb_analytic_ppb"] / sigma_crlb_ppb
                      if sigma_crlb_ppb > 0 else float("nan"))

        print(f"    sigma_CRLB(аудит) = {sigma_crlb_ppb:.4f} ppb | "
              f"sigma_CRLB(MC-аналит.) = {mc['sigma_crlb_analytic_ppb']:.4f} "
              f"ppb (x{crlb_match:.3f})")
        print(f"    sigma_MC = {sigma_mc_ppb:.4f} ppb | "
              f"MC/CRLB = {mc_over_crlb:.3f}")
        print(f"    sigma_naive (без дрейфа в модели) = "
              f"{mc['sigma_naive_ppb']:.4f} ppb (x{naive_inflation:.2f})")
        print(f"    J*_m = {J_star_m:.1f}  (J*/J*_min = {ratio:.3f})  -> "
              f"{verdict}")
        print(f"    sigma_target = {sigma_target_ppb:.4f} ppb, "
              f"расстояние до цели = {distance:.3f}")
        print(f"    95% ДИ: CRLB ±{Z95*sigma_crlb_ppb:.3f} ppb (ширина "
              f"{ci_crlb_ppb:.3f}), цель ширина {ci_target_ppb:.3f} ppb")

        results.append(dict(
            target=target, ru=au["ru"], config=au["config"], fwhm=fwhm,
            window=list(window), interferents=interf,
            centers=[round(float(c), 3) for c in centers],
            K=len(centers), greedy_gains=list(res.history_gain),
            c_design_ppb=c_design[target] * 1000.0,
            delta_ppb=delta_ppb, sigma_b_ppb=sigma_b_ppm * 1000.0,
            sigma_crlb_ppb=sigma_crlb_ppb,
            sigma_crlb_analytic_ppb=mc["sigma_crlb_analytic_ppb"],
            crlb_audit_vs_mc=crlb_match,
            sigma_mc_ppb=sigma_mc_ppb,
            mc_over_crlb=mc_over_crlb,
            bias_mc_ppb=mc["bias_full_ppb"],
            sigma_naive_ppb=mc["sigma_naive_ppb"],
            naive_inflation=naive_inflation,
            J_star_m=J_star_m, J_star_min=J_min, ratio=ratio,
            verdict=verdict,
            sigma_target_ppb=sigma_target_ppb,
            distance_to_target=distance,
            ci95_crlb_ppb=ci_crlb_ppb,
            ci95_mc_ppb=ci_mc_ppb,
            ci95_target_ppb=ci_target_ppb,
            n_mc=mc["n_mc"],
            _err_sample=mc["err_full"],  # для гистограмм, не сериализуется
        ))

    # --- сводка ---
    print("\n" + "=" * 86)
    print(f"{'вещество':10s} {'конф':6s} {'sCRLB':>8s} {'sMC':>8s} "
          f"{'MC/CR':>6s} {'J*/Jmin':>8s} {'верд':>6s} {'дист':>6s} "
          f"{'наив':>6s}")
    print("-" * 86)
    for r in results:
        print(f"{r['ru']:10s} {r['config']:6s} {r['sigma_crlb_ppb']:8.3f} "
              f"{r['sigma_mc_ppb']:8.3f} {r['mc_over_crlb']:6.3f} "
              f"{r['ratio']:8.3f} {r['verdict']:>6s} "
              f"{r['distance_to_target']:6.2f} {r['naive_inflation']:6.2f}")
    print("=" * 86)

    # --- запись JSON (без массивов выборок) ---
    out = []
    for r in results:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        out.append(rr)
    payload = dict(
        meta=dict(
            L_cm=L_CM, kappa=KAPPA, N_breaths=N_BREATHS, n_mc=N_MC,
            drift_deg=DRIFT_DEG, drift_sigma=list(DRIFT_SIGMA),
            J_star_min=J_min, z95=Z95, seed=RNG_SEED,
            fwhm_narrow=FWHM_NARROW, fwhm_wide=FWHM_WIDE,
        ),
        audits=out,
    )
    out_path = RESULTS_DIR / "ch5_montecarlo_results.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"\n[json] {out_path}")

    # --- рисунки ---
    make_figures(results)
    print("[fig] готово")


# ===========================================================================
# Рисунки
# ===========================================================================
def _save(fig, stem: str) -> None:
    for d in (RESULTS_DIR, FIG_DIR_LATEX):
        fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(RESULTS_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_figures(results: List[Dict[str, object]]) -> None:
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3})
    C_MC = "#3b6ea5"
    C_CRLB = "#c0392b"
    C_TARGET = "#27ae60"

    # --- Рис. 1: гистограммы ошибок vs CRLB и целевой коридор ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, r in zip(axes.ravel(), results):
        err = np.asarray(r["_err_sample"])
        s_crlb = r["sigma_crlb_ppb"]
        s_tgt = r["sigma_target_ppb"]
        mu = float(np.mean(err))
        lim = max(4.0 * s_crlb, Z95 * s_tgt * 1.3)
        xs = np.linspace(mu - lim, mu + lim, 400)
        ax.hist(err, bins=60, density=True, color=C_MC, alpha=0.45,
                label="MC оценки")
        gauss = (np.exp(-0.5 * ((xs - mu) / s_crlb) ** 2)
                 / (s_crlb * np.sqrt(2 * np.pi)))
        ax.plot(xs, gauss, color=C_CRLB, lw=2.0,
                label=f"$\\mathcal{{N}}(0,\\sigma_{{CRLB}})$")
        # целевой 95% коридор (зелёный) и достигнутый (красный пунктир)
        ax.axvspan(mu - Z95 * s_tgt, mu + Z95 * s_tgt, color=C_TARGET,
                   alpha=0.12, label="целевой 95% ДИ")
        ax.axvline(mu - Z95 * s_crlb, color=C_CRLB, ls="--", lw=1.0)
        ax.axvline(mu + Z95 * s_crlb, color=C_CRLB, ls="--", lw=1.0)
        ax.set_xlim(mu - lim, mu + lim)
        ax.set_title(f"{r['ru']} ({r['config']}, FWHM={r['fwhm']:g} нм) — "
                     f"{r['verdict']}")
        ax.set_xlabel("ошибка оценки, ppb")
        ax.set_ylabel("плотность")
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Глава 5: распределение MC-оценок против CRLB и целевого "
                 "доверительного коридора", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "ch5_mc_histograms")

    # --- Рис. 2: sigma_MC vs sigma_CRLB (достижение CRLB) ---
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    s_crlb = np.array([r["sigma_crlb_ppb"] for r in results])
    s_mc = np.array([r["sigma_mc_ppb"] for r in results])
    lo = 0.8 * min(s_crlb.min(), s_mc.min())
    hi = 1.25 * max(s_crlb.max(), s_mc.max())
    ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1.0,
            label=r"$\sigma_{MC}=\sigma_{CRLB}$")
    ax.scatter(s_crlb, s_mc, s=70, color=C_MC, zorder=3)
    for r in results:
        ax.annotate(r["ru"], (r["sigma_crlb_ppb"], r["sigma_mc_ppb"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\sigma_{CRLB}$ (аналитич.), ppb")
    ax.set_ylabel(r"$\sigma_{MC}$ (обеляющий фильтр + GLS), ppb")
    ax.set_title("Обеляющий фильтр достигает CRLB")
    ax.legend()
    fig.tight_layout()
    _save(fig, "ch5_mc_crlb_validation")

    # --- Рис. 3: расстояние до цели / вердикт ---
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    names = [r["ru"] for r in results]
    ratios = np.array([r["ratio"] for r in results])
    colors = [C_TARGET if r["verdict"].startswith("PASS") else C_CRLB
              for r in results]
    bars = ax.bar(names, ratios, color=colors, alpha=0.8)
    ax.axhline(1.0, color="k", ls="-", lw=1.2, label=r"$J^*_{\min}$ (PASS)")
    ax.axhline(2.0, color="k", ls="--", lw=1.0, label=r"$2J^*_{\min}$ (PASS+)")
    ax.set_yscale("log")
    ax.set_ylabel(r"$J^*_m / J^*_{\min}$")
    ax.set_title("Вердикт аудита: запас по информации (MC-валидация)")
    for b, r in zip(bars, results):
        ax.annotate(f"{r['verdict']}\nдист {r['distance_to_target']:.2f}",
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "ch5_mc_verdict")


if __name__ == "__main__":
    main()
