"""
spectrolib.plotting
===================
Унифицированная визуализация спектров.

Стиль:
- ось X в нм, явная подпись
- ось Y подписывается в зависимости от kind
- заголовок собирается автоматически из метаданных Spectrum
- тонкая сетка для удобства чтения отсчётов
- minor ticks для масштабирования
- если есть и истина, и наблюдаемое — режим 'compare' рисует обе

Все функции возвращают (fig, ax), чтобы пользователь мог дотюнить.
"""

from __future__ import annotations

import numpy as np

from . import plotstyle
from .plotstyle import _import_mpl, style_axes, new_figure, SEMANTIC, DEFAULTS


def _build_title(spec, extra=None):
    """
    Собрать заголовок графика из метаданных спектра.

    Если в spec.meta есть 'panel_name' (например, при генерации из
    MixturePanel) — он становится главным заголовком, молекулы
    суммируются строкой "(N биомаркеров)".

    Иначе — перечисляются молекулы и условия.
    """
    panel_name = spec.meta.get('panel_name') if hasattr(spec, 'meta') else None
    n_mols = len(spec.molecules)

    if panel_name:
        # Заголовок от панели; молекулы — в подпись
        if n_mols > 5:
            head = f"{panel_name} ({n_mols} биомаркеров)"
        else:
            mol_str = ', '.join(m['name'] for m in spec.molecules)
            head = f"{panel_name}: {mol_str}" if mol_str else panel_name
    elif not spec.molecules:
        head = 'Empty spectrum'
    elif n_mols > 5:
        head = f"Смесь из {n_mols} молекул"
    else:
        head = ', '.join(
            f"{m['name']} ({m['c_ppm']:g} ppm)" for m in spec.molecules
        )

    # Условия (T, p, L) — как раньше
    cond = ''
    if spec.molecules:
        m0 = spec.molecules[0]
        same_T = all(m['T_K'] == m0['T_K'] for m in spec.molecules)
        same_p = all(m['p_atm'] == m0['p_atm'] for m in spec.molecules)
        same_L = all(m['L_cm'] == m0['L_cm'] for m in spec.molecules)
        parts = []
        if same_T:
            parts.append(f"T={m0['T_K']:g} K")
        if same_p:
            parts.append(f"p={m0['p_atm']:g} atm")
        if same_L:
            parts.append(f"L={m0['L_cm']:g} cm")
        cond = ', '.join(parts)

    if cond:
        title = f"{head}  |  {cond}"
    else:
        title = head
    if extra:
        title = f"{title}  |  {extra}"
    return title


def _ylabel_for(kind):
    return {
        'transmittance': 'Transmittance T',
        'absorbance':    r'Absorbance $A = -\log_{10}(T)$',
        'optical_depth': r'Optical depth $\tau = -\ln(T)$',
    }[kind]


def _values_for(spec, kind, which):
    """
    Достать массив значений нужного типа.

    which : {'observed', 'true'}
    kind  : {'transmittance', 'absorbance', 'optical_depth'}
    """
    if which == 'observed':
        return {
            'transmittance': spec.transmittance,
            'absorbance':    spec.absorbance,
            'optical_depth': spec.optical_depth,
        }[kind]
    elif which == 'true':
        return {
            'transmittance': spec.true_transmittance,
            'absorbance':    spec.true_absorbance,
            'optical_depth': spec.true_optical_depth,
        }[kind]
    raise ValueError(f"which должен быть 'observed' или 'true', получено {which!r}")


# ---------------------------------------------------------------------------
# Основная функция
# ---------------------------------------------------------------------------

def plot_spectrum(spec, kind='transmittance', which='auto',
                  ax=None, figsize=None, title=None,
                  show_legend=True, **plot_kwargs):
    """
    Построить график спектра.

    Parameters
    ----------
    spec : Spectrum
    kind : {'transmittance', 'absorbance', 'optical_depth'}
        Какую величину рисовать.
    which : {'auto', 'observed', 'true', 'compare'}
        - 'auto': если есть шум — 'compare', иначе только 'true'.
        - 'observed': только наблюдаемое (с шумом).
        - 'true': только истина (без шума).
        - 'compare': и истина (пунктир), и наблюдаемое (сплошное).
    ax : matplotlib.axes.Axes, optional
        Если не задано — создаётся новая фигура.
    figsize : tuple, optional
    title : str, optional
        Если None — собирается автоматически из метаданных.
    show_legend : bool
    **plot_kwargs : передаются в ax.plot для основной линии.

    Returns
    -------
    (fig, ax) : matplotlib Figure, Axes
    """
    plt = _import_mpl()

    if kind not in ('transmittance', 'absorbance', 'optical_depth'):
        raise ValueError(
            f"kind должен быть transmittance/absorbance/optical_depth, "
            f"получено {kind!r}"
        )

    has_noise = spec.noise is not None
    if which == 'auto':
        which = 'compare' if has_noise else 'true'

    if ax is None:
        fig, ax = new_figure(figsize=figsize)
    else:
        fig = ax.figure

    x = spec.wavelength_nm

    if which == 'compare':
        if not has_noise:
            # Нет шума — нет смысла в сравнении, рисуем только истину
            y = _values_for(spec, kind, 'true')
            ax.plot(x, y, label='истина', **plot_kwargs)
        else:
            y_true = _values_for(spec, kind, 'true')
            y_obs = _values_for(spec, kind, 'observed')
            ax.plot(x, y_obs, color=SEMANTIC['observed'], lw=1.0, alpha=0.85,
                    label='наблюдаемое', **plot_kwargs)
            ax.plot(x, y_true, color=SEMANTIC['true'], lw=1.5, ls='--',
                    label='истина')
    elif which in ('true', 'observed'):
        y = _values_for(spec, kind, which)
        label = {'true': 'истина', 'observed': 'наблюдаемое'}[which]
        ax.plot(x, y, label=label, **plot_kwargs)
    else:
        raise ValueError(
            f"which должен быть auto/observed/true/compare, получено {which!r}"
        )

    ax.set_xlabel('Длина волны, нм')
    ax.set_ylabel(_ylabel_for(kind))
    ax.set_title(title if title is not None else _build_title(spec))

    # Для transmittance ставим разумные пределы [0, 1.02], если линии слабые
    if kind == 'transmittance':
        y_min = float(np.min([_values_for(spec, kind, 'true').min(),
                              _values_for(spec, kind, 'observed').min()
                              if has_noise else np.inf]))
        if y_min > 0.95:
            # Слабое поглощение — даём чуть запаса по обе стороны
            margin = max(0.001, (1.0 - y_min) * 0.5)
            ax.set_ylim(y_min - margin, 1.0 + margin)
        else:
            ax.set_ylim(max(0.0, y_min - 0.05), 1.02)

    style_axes(ax)

    if show_legend:
        ax.legend(loc=DEFAULTS['legend_loc'],
                  framealpha=DEFAULTS['legend_framealpha'],
                  fontsize=DEFAULTS['legend_size'])

    if DEFAULTS['tight_layout']:
        fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Сравнительные/диагностические графики
# ---------------------------------------------------------------------------

def plot_clean_vs_noisy(spec, kind='transmittance', figsize=None):
    """
    Двухпанельный график: сверху — спектр, снизу — разница (шум/остаток).

    Полезно для эксперимента 1 диплома: визуальная оценка шума и того,
    насколько препроцессинг должен «вытаскивать» истину.
    """
    plt = _import_mpl()
    # 2-панельный лэйаут: используем DEFAULTS для ширины, высота — увеличена
    base_w, _ = DEFAULTS['figsize']
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=figsize or (base_w, base_w * 0.6),
        dpi=DEFAULTS['dpi'],
        sharex=True, gridspec_kw={'height_ratios': [3, 1]},
    )
    plot_spectrum(spec, kind=kind, which='compare',
                  ax=ax_top, show_legend=True, title=_build_title(spec))

    # Разница в том же пространстве
    obs = _values_for(spec, kind, 'observed')
    tru = _values_for(spec, kind, 'true')
    diff = obs - tru
    ax_bot.plot(spec.wavelength_nm, diff, color=SEMANTIC['diff'], lw=0.8)
    ax_bot.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax_bot.set_xlabel('Длина волны, нм')
    ax_bot.set_ylabel('Шум\n(набл. − истина)')
    style_axes(ax_bot)
    rms = float(np.sqrt(np.mean(diff ** 2)))
    ax_bot.text(0.99, 0.95, f'RMS = {rms:.4g}',
                transform=ax_bot.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    if DEFAULTS['tight_layout']:
        fig.tight_layout()
    return fig, (ax_top, ax_bot)


def plot_overlay(specs, labels=None, kind='transmittance',
                 which='true', figsize=None, title=None):
    """
    Несколько спектров на одной оси — для сравнения вариантов
    (разные L, разные концентрации, разные ILS).

    Parameters
    ----------
    specs : list[Spectrum]
    labels : list[str], optional
    kind : см. plot_spectrum
    which : {'true', 'observed'}
    """
    fig, ax = new_figure(figsize=figsize)
    if labels is None:
        labels = [f'spec {i}' for i in range(len(specs))]
    for s, lab in zip(specs, labels):
        y = _values_for(s, kind, which)
        ax.plot(s.wavelength_nm, y, label=lab, lw=1.2)
    ax.set_xlabel('Длина волны, нм')
    ax.set_ylabel(_ylabel_for(kind))
    if title:
        ax.set_title(title)
    style_axes(ax)
    ax.legend(loc=DEFAULTS['legend_loc'],
              framealpha=DEFAULTS['legend_framealpha'],
              fontsize=DEFAULTS['legend_size'])
    if DEFAULTS['tight_layout']:
        fig.tight_layout()
    return fig, ax


def plot_snr_vs_n(snr_data, figsize=None, title=None,
                  show_theoretical=True, log_log=True):
    """
    График SNR vs число усреднённых реализаций.

    Parameters
    ----------
    snr_data : dict
        Результат SpectrumGenerator.snr_vs_n_realizations().
    figsize : tuple, optional
    title : str, optional
        По умолчанию: 'SNR vs число усреднённых реализаций'.
    show_theoretical : bool
        Показывать ли теоретическую линию ∝ √N.
    log_log : bool
        Логарифмические оси (на них ∝ √N — прямая с наклоном 0.5).

    Returns
    -------
    (fig, ax)
    """
    n_values = snr_data['n_values']
    snr_mean = snr_data['snr_mean']
    snr_std = snr_data['snr_std']
    theoretical = snr_data['theoretical']

    fig, ax = new_figure(figsize=figsize)

    ax.errorbar(n_values, snr_mean, yerr=snr_std, fmt='o-',
                color=SEMANTIC['observed'],
                capsize=DEFAULTS['errorbar_capsize'],
                lw=DEFAULTS['lw'], markersize=6,
                label='измеренный SNR (среднее ± СКО)')

    if show_theoretical:
        ax.plot(n_values, theoretical, '--', color=SEMANTIC['theory'],
                lw=DEFAULTS['lw'],
                label=r'теория: SNR $\propto \sqrt{N}$')

    if log_log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    ax.set_xlabel('N — число усреднённых реализаций')
    ax.set_ylabel('SNR')
    ax.set_title(title or 'SNR vs число усреднённых реализаций')
    style_axes(ax)
    ax.legend(loc=DEFAULTS['legend_loc'],
              framealpha=DEFAULTS['legend_framealpha'],
              fontsize=DEFAULTS['legend_size'])
    if DEFAULTS['tight_layout']:
        fig.tight_layout()
    return fig, ax
