"""
spectrolib.plotstyle
====================
Универсальный модуль визуализации с едиными дефолтами.

Можно импортировать отдельно — он не зависит от остального spectrolib:

    from spectrolib.plotstyle import plot, scatter, bar, hist, errorbar
    from spectrolib.plotstyle import PALETTE, SEMANTIC, MARKERS, DEFAULTS

    plot(x, y, xlabel='λ, нм', ylabel='T', title='Spectrum', label='obs')

Дефолты (figsize, dpi, цвета, сетка, шрифты) лежат в `DEFAULTS` — можно
перезаписать разово в начале ноутбука:

    from spectrolib import plotstyle
    plotstyle.DEFAULTS['figsize'] = (10, 5)

Палитра — Okabe-Ito (colorblind-safe). Цвет можно задавать именем из
`PALETTE`/`SEMANTIC` или любым matplotlib-цветом:

    plot(x, y, color='accent')       # из палитры
    plot(x, y, color='#ff0033')      # hex
    plot(x, y, color=(0.1, 0.2, 0.7)) # RGB

Для сохранения в LaTeX используйте векторный формат:

    ax = plot(x, y, xlabel=..., ylabel=...)
    ax.figure.savefig('fig.pdf', bbox_inches='tight')
"""

from __future__ import annotations

import os
from typing import Any


# ---------------------------------------------------------------------------
# Палитра Okabe-Ito (colorblind-safe)
# ---------------------------------------------------------------------------

PALETTE = {
    'primary':   '#0072B2',  # синий
    'accent':    '#D55E00',  # вермильон — высоко контрастно к primary
    'success':   '#009E73',  # зелёный
    'warning':   '#E69F00',  # оранжевый
    'muted':     '#56B4E9',  # светло-голубой
    'highlight': '#CC79A7',  # розовато-пурпурный
    'soft':      '#F0E442',  # жёлтый
    'neutral':   '#000000',  # чёрный
}

# Семантические алиасы — для повторяющихся ролей в графиках
SEMANTIC = {
    'observed':    PALETTE['primary'],
    'true':        PALETTE['accent'],
    'diff':        PALETTE['success'],
    'theory':      PALETTE['highlight'],
    'theoretical': PALETTE['highlight'],
    'residual':    PALETTE['success'],
}

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


# ---------------------------------------------------------------------------
# Дефолтные настройки
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    # Размер
    'figsize':            (12, 6),   # пошире, чтобы хорошо ложилось в LaTeX
    'dpi':                130,       # дисплейный dpi
    'savefig_dpi':        300,       # dpi при сохранении PNG (для LaTeX)

    # Линии и маркеры
    'lw':                 1.8,
    'marker_size':        40,        # для scatter (s=)
    'errorbar_capsize':   3,
    'bar_width':          0.8,
    'bar_alpha':          0.85,
    'hist_alpha':         0.85,
    'hist_edgecolor':     'white',

    # Цвета
    'color_cycle': [
        PALETTE['primary'], PALETTE['accent'], PALETTE['success'],
        PALETTE['warning'], PALETTE['muted'], PALETTE['highlight'],
        PALETTE['soft'], PALETTE['neutral'],
    ],
    'default_color':      PALETTE['primary'],

    # Сетка
    'grid':               True,
    'grid_alpha':         0.3,
    'minor_ticks':        True,
    'minor_grid_alpha':   0.1,

    # Рамки (spines)
    'spine_top':          False,
    'spine_right':        False,

    # Шрифты
    'font_size':          11,
    'title_size':         13,
    'label_size':         12,
    'legend_size':        10,

    # Легенда
    'legend_loc':         'best',
    'legend_framealpha':  0.9,

    # Прочее
    'tight_layout':       True,
}


# ---------------------------------------------------------------------------
# Внутренние хелперы
# ---------------------------------------------------------------------------

def _import_mpl():
    """Ленивый импорт matplotlib — модуль должен импортироваться без него."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError(
            "Для plotstyle нужен matplotlib: pip install matplotlib"
        ) from e


def _resolve_color(color):
    """Имя из PALETTE/SEMANTIC -> hex; иначе пропускаем как есть."""
    if color is None:
        return None
    if isinstance(color, str):
        if color in PALETTE:
            return PALETTE[color]
        if color in SEMANTIC:
            return SEMANTIC[color]
    return color


def new_figure(figsize=None, dpi=None):
    """Создать фигуру + ось с дефолтным стилем (figsize, dpi, цикл цветов)."""
    plt = _import_mpl()
    from cycler import cycler
    fig, ax = plt.subplots(
        figsize=figsize or DEFAULTS['figsize'],
        dpi=dpi or DEFAULTS['dpi'],
    )
    ax.set_prop_cycle(cycler(color=DEFAULTS['color_cycle']))
    return fig, ax


def subplots(nrows=1, ncols=1, *, figsize=None, sharex=False, sharey=False,
             flatten=True, **kwargs):
    """
    Сетка подграфиков со стилем, применённым ко всем осям.

    Адаптивный figsize: масштабируется по числу столбцов/строк, чтобы каждый
    подграфик оставался читаемым (если не задан явно).

    Parameters
    ----------
    nrows, ncols : int
    figsize : (w, h), optional
        Если None — рассчитывается из DEFAULTS['figsize'] и сетки.
    sharex, sharey : bool
    flatten : bool, default True
        Для grid (n×m) возвращает axes как одномерный массив — удобно
        итерировать `for ax in axes:`. Поставьте False, если нужна
        2D-индексация `axes[i, j]`.

    Returns
    -------
    (fig, axes)
        Для 1×1 axes — одна ось, не массив.
    """
    plt = _import_mpl()
    from cycler import cycler
    import numpy as _np

    base_w, base_h = DEFAULTS['figsize']
    if figsize is None:
        # Логика: ширина растёт с ncols, но не линейно
        # (1 кол → base_w; 2 кол → 1.5*base_w; 3 кол → 2*base_w)
        w = base_w * (0.5 + 0.5 * ncols)
        h = base_h * (0.5 + 0.5 * nrows)
        figsize = (w, h)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, dpi=DEFAULTS['dpi'],
        sharex=sharex, sharey=sharey, **kwargs,
    )

    ax_list = [axes] if (nrows == 1 and ncols == 1) else _np.asarray(axes).ravel()
    for ax in ax_list:
        ax.set_prop_cycle(cycler(color=DEFAULTS['color_cycle']))
        style_axes(ax)

    if DEFAULTS['tight_layout']:
        fig.tight_layout()

    if nrows == 1 and ncols == 1:
        return fig, axes
    if flatten:
        return fig, _np.asarray(axes).ravel()
    return fig, axes


def save(target, path, *, dpi=None, bbox_inches='tight', **kwargs):
    """
    Сохранить фигуру (или ось — возьмёт её figure) в файл.

    По расширению выбирает оптимальные настройки:
    - .pdf/.svg/.eps  → векторный, dpi игнорируется
    - .png/.jpg/.tiff → растр, dpi = DEFAULTS['savefig_dpi'] (300 по дефолту)

    Для LaTeX рекомендуется PDF.

    Parameters
    ----------
    target : matplotlib.figure.Figure | matplotlib.axes.Axes
    path : str | os.PathLike
    dpi : int, optional
        Только для растровых форматов; переопределяет DEFAULTS.
    bbox_inches : str, default 'tight'
    **kwargs : пробрасываются в Figure.savefig.

    Returns
    -------
    str
        Абсолютный путь к сохранённому файлу.
    """
    fig = target.figure if hasattr(target, 'figure') else target
    path_str = os.fspath(path)
    ext = path_str.rsplit('.', 1)[-1].lower() if '.' in path_str else ''
    raster = ext in ('png', 'jpg', 'jpeg', 'tiff', 'tif', 'webp', 'bmp')
    save_kwargs = dict(bbox_inches=bbox_inches, **kwargs)
    if raster:
        save_kwargs.setdefault(
            'dpi', dpi if dpi is not None else DEFAULTS['savefig_dpi']
        )
    fig.savefig(path_str, **save_kwargs)
    return os.path.abspath(path_str)


def style_axes(ax):
    """Применить дефолтный стиль к существующей оси: сетка, рамки, шрифты."""
    if DEFAULTS['grid']:
        ax.grid(True, which='major', alpha=DEFAULTS['grid_alpha'])
        if DEFAULTS['minor_ticks']:
            ax.minorticks_on()
            ax.grid(True, which='minor', alpha=DEFAULTS['minor_grid_alpha'])

    ax.spines['top'].set_visible(DEFAULTS['spine_top'])
    ax.spines['right'].set_visible(DEFAULTS['spine_right'])

    ax.title.set_fontsize(DEFAULTS['title_size'])
    ax.xaxis.label.set_fontsize(DEFAULTS['label_size'])
    ax.yaxis.label.set_fontsize(DEFAULTS['label_size'])
    ax.tick_params(labelsize=DEFAULTS['font_size'])
    return ax


def _finalize(ax, *, xlabel=None, ylabel=None, title=None,
              log_x=False, log_y=False, want_legend=False):
    style_axes(ax)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if want_legend:
        _, lbls = ax.get_legend_handles_labels()
        if lbls:
            ax.legend(loc=DEFAULTS['legend_loc'],
                      framealpha=DEFAULTS['legend_framealpha'],
                      fontsize=DEFAULTS['legend_size'])
    if DEFAULTS['tight_layout']:
        ax.figure.tight_layout()


def _ensure_axes(ax, figsize):
    if ax is not None:
        return ax.figure, ax
    return new_figure(figsize=figsize)


# ---------------------------------------------------------------------------
# Публичные функции построения графиков
# ---------------------------------------------------------------------------

def plot(x, y=None, *, ax=None, label=None,
         color=None, linestyle=None, marker=None, lw=None,
         xlabel=None, ylabel=None, title=None,
         log_x=False, log_y=False,
         figsize=None, **kwargs):
    """
    Линейный график со стандартным стилем.

    Parameters
    ----------
    x, y : array-like
        Если y=None, то x трактуется как ординаты, абсциссы — индекс.
    ax : matplotlib.axes.Axes, optional
        Если не задано — создаётся новая фигура.
    label : str, optional
        Если передан — легенда включается автоматически.
    color : str, optional
        Имя из PALETTE/SEMANTIC ('primary', 'accent', 'observed'...) или
        любой цвет matplotlib (hex, RGB, имя).
    linestyle, marker, lw : параметры matplotlib (lw по умолчанию из DEFAULTS).
    xlabel, ylabel, title : подписи.
    log_x, log_y : bool
        Логарифмическая шкала по соответствующей оси.
    figsize : (w, h), optional
        Размер фигуры, если ax=None.
    **kwargs : пробрасываются в ax.plot().

    Returns
    -------
    matplotlib.axes.Axes
    """
    _, ax = _ensure_axes(ax, figsize)

    line_kwargs = dict(kwargs)
    if label is not None:
        line_kwargs['label'] = label
    resolved_color = _resolve_color(color)
    if resolved_color is not None:
        line_kwargs['color'] = resolved_color
    if linestyle is not None:
        line_kwargs['linestyle'] = linestyle
    if marker is not None:
        line_kwargs['marker'] = marker
    line_kwargs.setdefault('lw', lw if lw is not None else DEFAULTS['lw'])

    if y is None:
        ax.plot(x, **line_kwargs)
    else:
        ax.plot(x, y, **line_kwargs)

    _finalize(ax, xlabel=xlabel, ylabel=ylabel, title=title,
              log_x=log_x, log_y=log_y, want_legend=label is not None)
    return ax


def scatter(x, y, *, ax=None, label=None,
            color=None, marker=None, size=None,
            xlabel=None, ylabel=None, title=None,
            log_x=False, log_y=False,
            figsize=None, **kwargs):
    """
    Точечный график со стандартным стилем.

    `size` — площадь маркера (matplotlib `s=`), по умолчанию из DEFAULTS.
    """
    _, ax = _ensure_axes(ax, figsize)

    s_kwargs = dict(kwargs)
    if label is not None:
        s_kwargs['label'] = label
    resolved_color = _resolve_color(color)
    if resolved_color is not None:
        s_kwargs['color'] = resolved_color
    if marker is not None:
        s_kwargs['marker'] = marker
    s_kwargs.setdefault('s', size if size is not None else DEFAULTS['marker_size'])

    ax.scatter(x, y, **s_kwargs)

    _finalize(ax, xlabel=xlabel, ylabel=ylabel, title=title,
              log_x=log_x, log_y=log_y, want_legend=label is not None)
    return ax


def bar(x, height, *, ax=None, label=None,
        color=None, width=None, horizontal=False,
        xlabel=None, ylabel=None, title=None,
        log_x=False, log_y=False,
        figsize=None, **kwargs):
    """
    Столбчатый график. `horizontal=True` — горизонтальные столбцы.
    """
    _, ax = _ensure_axes(ax, figsize)

    b_kwargs = dict(kwargs)
    if label is not None:
        b_kwargs['label'] = label
    b_kwargs.setdefault('color', _resolve_color(color) or DEFAULTS['default_color'])
    b_kwargs.setdefault('alpha', DEFAULTS['bar_alpha'])
    w = width if width is not None else DEFAULTS['bar_width']

    if horizontal:
        ax.barh(x, height, height=w, **b_kwargs)
    else:
        ax.bar(x, height, width=w, **b_kwargs)

    _finalize(ax, xlabel=xlabel, ylabel=ylabel, title=title,
              log_x=log_x, log_y=log_y, want_legend=label is not None)
    return ax


def hist(data, *, ax=None, label=None,
         color=None, bins=30, density=False,
         xlabel=None, ylabel=None, title=None,
         log_x=False, log_y=False,
         figsize=None, **kwargs):
    """
    Гистограмма со стандартным стилем (белая обводка столбцов).
    """
    _, ax = _ensure_axes(ax, figsize)

    h_kwargs = dict(kwargs)
    if label is not None:
        h_kwargs['label'] = label
    h_kwargs.setdefault('color', _resolve_color(color) or DEFAULTS['default_color'])
    h_kwargs.setdefault('alpha', DEFAULTS['hist_alpha'])
    h_kwargs.setdefault('edgecolor', DEFAULTS['hist_edgecolor'])
    h_kwargs.setdefault('bins', bins)
    h_kwargs.setdefault('density', density)

    ax.hist(data, **h_kwargs)

    if ylabel is None:
        ylabel = 'Плотность' if density else 'Частота'

    _finalize(ax, xlabel=xlabel, ylabel=ylabel, title=title,
              log_x=log_x, log_y=log_y, want_legend=label is not None)
    return ax


def errorbar(x, y, yerr=None, xerr=None, *, ax=None, label=None,
             color=None, fmt='o-', capsize=None, markersize=6,
             xlabel=None, ylabel=None, title=None,
             log_x=False, log_y=False,
             figsize=None, **kwargs):
    """
    График с error bars. `fmt='o-'` по умолчанию: маркеры + линия.
    """
    _, ax = _ensure_axes(ax, figsize)

    e_kwargs = dict(kwargs)
    if label is not None:
        e_kwargs['label'] = label
    resolved_color = _resolve_color(color)
    if resolved_color is not None:
        e_kwargs['color'] = resolved_color
    e_kwargs.setdefault('capsize', capsize if capsize is not None
                        else DEFAULTS['errorbar_capsize'])
    e_kwargs.setdefault('lw', DEFAULTS['lw'])
    e_kwargs.setdefault('markersize', markersize)

    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=fmt, **e_kwargs)

    _finalize(ax, xlabel=xlabel, ylabel=ylabel, title=title,
              log_x=log_x, log_y=log_y, want_legend=label is not None)
    return ax


# ---------------------------------------------------------------------------
# Глобальное применение стиля (для случая «хочу единый вид и в plt.plot тоже»)
# ---------------------------------------------------------------------------

def apply_style(preset='default'):
    """
    Применить дефолты к глобальным rcParams matplotlib. После этого
    любой `plt.plot`/`plt.bar` в этом же процессе тоже подхватит стиль.

    Parameters
    ----------
    preset : {'default', 'mathtext', 'latex'}
        - 'default'  — системные шрифты matplotlib.
        - 'mathtext' — Computer Modern через mathtext (LaTeX устанавливать
          не нужно). Хороший выбор для тезисов без жёсткой типографики.
        - 'latex'    — `text.usetex=True`, рендеринг текста через настоящий
          LaTeX. Требует установленного LaTeX (texlive/MacTeX). Шрифты
          будут совпадать с текстом диплома.

    Returns
    -------
    dict
        Прежние значения rcParams — для отката через `plt.rcParams.update(prev)`.
    """
    plt = _import_mpl()
    from cycler import cycler

    overrides = {
        'figure.figsize':      DEFAULTS['figsize'],
        'figure.dpi':          DEFAULTS['dpi'],
        'savefig.dpi':         DEFAULTS['savefig_dpi'],
        'savefig.bbox':        'tight',
        'axes.prop_cycle':     cycler(color=DEFAULTS['color_cycle']),
        'axes.grid':           DEFAULTS['grid'],
        'grid.alpha':          DEFAULTS['grid_alpha'],
        'axes.spines.top':     DEFAULTS['spine_top'],
        'axes.spines.right':   DEFAULTS['spine_right'],
        'font.size':           DEFAULTS['font_size'],
        'axes.titlesize':      DEFAULTS['title_size'],
        'axes.labelsize':      DEFAULTS['label_size'],
        'legend.fontsize':     DEFAULTS['legend_size'],
        'legend.framealpha':   DEFAULTS['legend_framealpha'],
        'lines.linewidth':     DEFAULTS['lw'],
    }

    if preset == 'mathtext':
        overrides.update({
            'mathtext.fontset':  'cm',
            'font.family':       'serif',
            'font.serif':        ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
            'axes.unicode_minus': False,
        })
    elif preset == 'latex':
        overrides.update({
            'text.usetex':       True,
            'font.family':       'serif',
            'font.serif':        ['Computer Modern Roman'],
            'axes.unicode_minus': False,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        })
    elif preset != 'default':
        raise ValueError(
            f"preset должен быть 'default'/'mathtext'/'latex', получено {preset!r}"
        )

    prev = {k: plt.rcParams[k] for k in overrides}
    plt.rcParams.update(overrides)
    return prev


__all__ = [
    'PALETTE', 'SEMANTIC', 'MARKERS', 'DEFAULTS',
    'plot', 'scatter', 'bar', 'hist', 'errorbar',
    'new_figure', 'subplots', 'save', 'style_axes', 'apply_style',
]
