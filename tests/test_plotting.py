"""Тесты визуализации и удобных методов GasMixture."""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')   # без дисплея
import matplotlib.pyplot as plt

from spectrolib import (
    Spectrum, GaussILS, NoiseModel, GasMixture,
)


@pytest.fixture
def simple_spec():
    return (Spectrum.from_range(750, 770, step_nm=0.01)
            .add_gauss_peak(760, fwhm_nm=0.5, amplitude=0.3))


@pytest.fixture
def noisy_spec(simple_spec):
    return simple_spec.add_noise(sigma=0.01, seed=42)


class TestPlotBasic:

    def test_plot_returns_fig_ax(self, simple_spec):
        fig, ax = simple_spec.plot()
        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_kinds(self, simple_spec):
        for kind in ('transmittance', 'absorbance', 'optical_depth'):
            fig, ax = simple_spec.plot(kind=kind)
            assert ax.get_ylabel()  # есть подпись Y
            plt.close(fig)

    def test_plot_invalid_kind_raises(self, simple_spec):
        with pytest.raises(ValueError):
            simple_spec.plot(kind='nonsense')

    def test_plot_no_noise_shows_only_truth(self, simple_spec):
        fig, ax = simple_spec.plot(which='auto')
        # Без шума — только одна линия
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_plot_with_noise_compares(self, noisy_spec):
        fig, ax = noisy_spec.plot(which='auto')
        # С шумом — две линии (compare)
        assert len(ax.lines) == 2
        plt.close(fig)

    def test_plot_explicit_observed(self, noisy_spec):
        fig, ax = noisy_spec.plot(which='observed')
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_plot_title_auto(self):
        s = (Spectrum.from_range(750, 770, step_nm=0.01)
             .add_gauss_peak(760, 0.5, 0.3))
        fig, ax = s.plot()
        # У этого спектра нет молекул — но есть пик; заголовок не падает
        assert ax.get_title()
        plt.close(fig)

    def test_plot_into_existing_ax(self, simple_spec):
        fig, ax = plt.subplots()
        fig2, ax2 = simple_spec.plot(ax=ax)
        assert ax is ax2
        assert fig2 is fig
        plt.close(fig)


class TestPlotCleanVsNoisy:

    def test_two_panels(self, noisy_spec):
        fig, axes = noisy_spec.plot_clean_vs_noisy()
        assert len(axes) == 2
        plt.close(fig)

    def test_rms_annotation(self, noisy_spec):
        fig, (ax_top, ax_bot) = noisy_spec.plot_clean_vs_noisy()
        texts = [t.get_text() for t in ax_bot.texts]
        assert any('RMS' in t for t in texts)
        plt.close(fig)


class TestGasMixtureHelpers:

    def test_with_L(self):
        m = GasMixture(composition={'O2': 1000}, T_K=300, p_atm=1.0, L_cm=10)
        m2 = m.with_L(50)
        assert m.L_cm == 10
        assert m2.L_cm == 50
        assert m2.composition == m.composition
        assert m2 is not m

    def test_with_composition_update(self):
        m = GasMixture(composition={'O2': 1000, 'H2O': 5000})
        m2 = m.with_composition(O2=2000)
        assert m.composition['O2'] == 1000
        assert m2.composition['O2'] == 2000
        assert m2.composition['H2O'] == 5000

    def test_with_composition_remove(self):
        m = GasMixture(composition={'O2': 1000, 'H2O': 5000})
        m2 = m.with_composition(H2O=None)
        assert 'H2O' not in m2.composition
        assert 'O2' in m2.composition

    def test_with_composition_add(self):
        m = GasMixture(composition={'O2': 1000})
        m2 = m.with_composition(CO2=400000)
        assert m2.composition == {'O2': 1000, 'CO2': 400000}
