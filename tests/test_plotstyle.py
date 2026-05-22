"""Тесты на spectrolib.plotstyle."""

import os

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectrolib import plotstyle
from spectrolib.plotstyle import (
    PALETTE, SEMANTIC, MARKERS, DEFAULTS,
    plot, scatter, bar, hist, errorbar,
    new_figure, style_axes, apply_style,
)


class TestPalette:

    def test_palette_keys(self):
        # Базовые имена должны существовать
        for name in ('primary', 'accent', 'success', 'warning',
                     'muted', 'highlight', 'soft', 'neutral'):
            assert name in PALETTE
            assert PALETTE[name].startswith('#')

    def test_semantic_keys(self):
        for name in ('observed', 'true', 'diff', 'theory'):
            assert name in SEMANTIC

    def test_markers_nonempty(self):
        assert len(MARKERS) >= 4


class TestNewFigure:

    def test_returns_fig_ax(self):
        fig, ax = new_figure()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_uses_default_figsize(self):
        fig, ax = new_figure()
        assert tuple(fig.get_size_inches()) == tuple(DEFAULTS['figsize'])
        plt.close(fig)

    def test_custom_figsize(self):
        fig, ax = new_figure(figsize=(7, 3))
        assert tuple(fig.get_size_inches()) == (7, 3)
        plt.close(fig)


class TestStyleAxes:

    def test_spines_hidden(self):
        fig, ax = plt.subplots()
        style_axes(ax)
        assert not ax.spines['top'].get_visible()
        assert not ax.spines['right'].get_visible()
        plt.close(fig)

    def test_grid_on(self):
        fig, ax = plt.subplots()
        style_axes(ax)
        # просто проверяем, что не падает и сетка включена
        assert ax.xaxis.get_gridlines() != []
        plt.close(fig)


class TestPlot:

    def test_basic(self):
        x = np.linspace(0, 1, 50)
        y = np.sin(2 * np.pi * x)
        ax = plot(x, y)
        assert len(ax.lines) == 1
        plt.close(ax.figure)

    def test_into_existing_ax(self):
        fig, ax_in = plt.subplots()
        ax_out = plot([1, 2, 3], [4, 5, 6], ax=ax_in)
        assert ax_in is ax_out
        plt.close(fig)

    def test_label_triggers_legend(self):
        ax = plot([1, 2], [3, 4], label='foo')
        assert ax.get_legend() is not None
        plt.close(ax.figure)

    def test_no_label_no_legend(self):
        ax = plot([1, 2], [3, 4])
        assert ax.get_legend() is None
        plt.close(ax.figure)

    def test_color_by_palette_name(self):
        ax = plot([1, 2], [3, 4], color='accent')
        assert ax.lines[0].get_color() == PALETTE['accent']
        plt.close(ax.figure)

    def test_color_by_semantic_name(self):
        ax = plot([1, 2], [3, 4], color='observed')
        assert ax.lines[0].get_color() == SEMANTIC['observed']
        plt.close(ax.figure)

    def test_color_passthrough(self):
        ax = plot([1, 2], [3, 4], color='#112233')
        assert ax.lines[0].get_color() == '#112233'
        plt.close(ax.figure)

    def test_labels_and_title(self):
        ax = plot([1, 2], [3, 4], xlabel='X', ylabel='Y', title='T')
        assert ax.get_xlabel() == 'X'
        assert ax.get_ylabel() == 'Y'
        assert ax.get_title() == 'T'
        plt.close(ax.figure)

    def test_log_scales(self):
        ax = plot([1, 2, 3], [10, 100, 1000], log_x=True, log_y=True)
        assert ax.get_xscale() == 'log'
        assert ax.get_yscale() == 'log'
        plt.close(ax.figure)

    def test_y_none_uses_x_as_y(self):
        ax = plot([1, 4, 9, 16])
        assert len(ax.lines) == 1
        plt.close(ax.figure)


class TestScatter:

    def test_basic(self):
        ax = scatter([1, 2, 3], [4, 5, 6])
        assert len(ax.collections) == 1
        plt.close(ax.figure)

    def test_color_and_label(self):
        ax = scatter([1, 2], [3, 4], color='success', label='pts')
        assert ax.get_legend() is not None
        plt.close(ax.figure)


class TestBar:

    def test_basic(self):
        ax = bar([1, 2, 3], [4, 5, 6])
        assert len(ax.patches) == 3
        plt.close(ax.figure)

    def test_horizontal(self):
        ax = bar([1, 2, 3], [4, 5, 6], horizontal=True)
        assert len(ax.patches) == 3
        plt.close(ax.figure)


class TestHist:

    def test_basic(self):
        data = np.random.RandomState(0).randn(200)
        ax = hist(data, bins=10)
        # patches содержит столбцы гистограммы
        assert len(ax.patches) == 10
        plt.close(ax.figure)

    def test_density_label(self):
        data = np.random.RandomState(0).randn(100)
        ax = hist(data, density=True)
        assert 'плотность' in ax.get_ylabel().lower()
        plt.close(ax.figure)

    def test_count_label(self):
        data = np.random.RandomState(0).randn(100)
        ax = hist(data)
        assert 'частота' in ax.get_ylabel().lower()
        plt.close(ax.figure)


class TestErrorbar:

    def test_basic(self):
        x = [1, 2, 3]
        y = [2, 4, 6]
        e = [0.1, 0.2, 0.3]
        ax = errorbar(x, y, yerr=e, label='m')
        assert ax.get_legend() is not None
        plt.close(ax.figure)


class TestApplyStyle:

    def test_apply_and_restore(self):
        prev_size = plt.rcParams['figure.figsize']
        restored = apply_style()
        assert tuple(plt.rcParams['figure.figsize']) == tuple(DEFAULTS['figsize'])
        # Откат через возвращённый dict
        plt.rcParams.update(restored)
        assert tuple(plt.rcParams['figure.figsize']) == tuple(prev_size)

    def test_mathtext_preset(self):
        restored = apply_style(preset='mathtext')
        try:
            assert plt.rcParams['mathtext.fontset'] == 'cm'
            assert plt.rcParams['font.family'] == ['serif']
        finally:
            plt.rcParams.update(restored)

    def test_latex_preset_sets_usetex_flag(self):
        # Не рендерим — LaTeX может быть не установлен. Только проверяем,
        # что rcParam установлен.
        restored = apply_style(preset='latex')
        try:
            assert plt.rcParams['text.usetex'] is True
        finally:
            plt.rcParams.update(restored)

    def test_invalid_preset_raises(self):
        with pytest.raises(ValueError):
            apply_style(preset='nope')


class TestSubplots:

    def test_1x1_returns_single_ax(self):
        from spectrolib.plotstyle import subplots
        fig, ax = subplots(1, 1)
        # Не массив — одна ось
        assert not isinstance(ax, np.ndarray)
        plt.close(fig)

    def test_2x2_flattened(self):
        from spectrolib.plotstyle import subplots
        fig, axes = subplots(2, 2)
        assert isinstance(axes, np.ndarray)
        assert axes.ndim == 1
        assert len(axes) == 4
        plt.close(fig)

    def test_2x2_no_flatten(self):
        from spectrolib.plotstyle import subplots
        fig, axes = subplots(2, 2, flatten=False)
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_style_applied_to_all_axes(self):
        from spectrolib.plotstyle import subplots
        fig, axes = subplots(1, 3)
        for ax in axes:
            assert not ax.spines['top'].get_visible()
            assert not ax.spines['right'].get_visible()
        plt.close(fig)

    def test_figsize_scales_with_grid(self):
        from spectrolib.plotstyle import subplots
        fig1, _ = subplots(1, 1)
        fig2, _ = subplots(1, 3)
        w1 = fig1.get_size_inches()[0]
        w2 = fig2.get_size_inches()[0]
        assert w2 > w1
        plt.close(fig1)
        plt.close(fig2)


class TestSave:

    def test_save_pdf(self, tmp_path):
        from spectrolib.plotstyle import save
        ax = plot([1, 2, 3], [4, 5, 6])
        out = tmp_path / 'test.pdf'
        result = save(ax, out)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 0
        plt.close(ax.figure)

    def test_save_png_uses_dpi(self, tmp_path):
        from spectrolib.plotstyle import save
        ax = plot([1, 2, 3], [4, 5, 6])
        out = tmp_path / 'test.png'
        save(ax, out, dpi=72)  # маленький, чтобы был быстрый тест
        assert os.path.exists(out)
        plt.close(ax.figure)

    def test_save_accepts_figure(self, tmp_path):
        from spectrolib.plotstyle import save, new_figure
        fig, ax = new_figure()
        ax.plot([1, 2], [3, 4])
        out = tmp_path / 'test.svg'
        save(fig, out)
        assert os.path.exists(out)
        plt.close(fig)


class TestDefaultsMutable:

    def test_override_figsize(self):
        old = DEFAULTS['figsize']
        try:
            DEFAULTS['figsize'] = (5, 5)
            fig, ax = new_figure()
            assert tuple(fig.get_size_inches()) == (5, 5)
            plt.close(fig)
        finally:
            DEFAULTS['figsize'] = old
