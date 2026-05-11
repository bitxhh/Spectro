"""Тесты многоканальной регистрации."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectrolib import (
    Spectrum, GaussILS, NoiseModel,
    Channel, ChannelSet, ChannelizedSpectrum,
    channelize, load_channel_set,
    FromFileILS,
)


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class TestChannel:

    def test_basic_creation(self):
        ch = Channel(center_nm=760, fwhm_nm=25, shape='gauss')
        assert ch.center_nm == 760
        assert ch.fwhm_nm == 25
        assert ch.shape == 'gauss'

    def test_invalid_fwhm_raises(self):
        with pytest.raises(ValueError, match='> 0'):
            Channel(center_nm=760, fwhm_nm=0)
        with pytest.raises(ValueError, match='> 0'):
            Channel(center_nm=760, fwhm_nm=-1)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match='shape'):
            Channel(center_nm=760, fwhm_nm=25, shape='unknown')

    def test_response_normalized(self):
        ch = Channel(center_nm=760, fwhm_nm=10, shape='gauss')
        wl = np.linspace(700, 820, 12001)
        resp = ch.response(wl)
        # Площадь под нормированной функцией = 1
        area = np.trapezoid(resp, wl)
        assert area == pytest.approx(1.0, rel=1e-3)

    def test_response_peaks_at_center(self):
        ch = Channel(center_nm=760, fwhm_nm=10, shape='gauss')
        wl = np.linspace(700, 820, 12001)
        resp = ch.response(wl)
        idx_max = int(np.argmax(resp))
        assert wl[idx_max] == pytest.approx(760.0, abs=0.02)

    def test_lorentz_shape(self):
        ch = Channel(center_nm=760, fwhm_nm=10, shape='lorentz')
        wl = np.linspace(700, 820, 12001)
        resp = ch.response(wl)
        # Лоренцевы хвосты — широкие, в норму всё равно попадает
        assert np.trapezoid(resp, wl) == pytest.approx(1.0, rel=1e-2)

    def test_voigt_shape(self):
        ch = Channel(center_nm=760, fwhm_nm=10, shape='voigt')
        wl = np.linspace(700, 820, 12001)
        resp = ch.response(wl)
        assert np.trapezoid(resp, wl) == pytest.approx(1.0, rel=1e-2)

    def test_ils_object_as_shape(self):
        """Можно передать готовый ILS-объект (например, из FromFileILS)."""
        x = np.linspace(-15, 15, 301)
        intensity = np.exp(-x ** 2 / 8)  # ~гаусс с FWHM ~5 нм
        ils = FromFileILS(x, intensity)
        ch = Channel(center_nm=760, fwhm_nm=5, shape=ils)
        wl = np.linspace(700, 820, 12001)
        resp = ch.response(wl)
        # Должен интегрироваться в ~1 (может быть небольшое отклонение
        # из-за того, что область интерполяции FromFileILS ограничена)
        area = np.trapezoid(resp, wl)
        assert 0.8 < area < 1.2

    def test_to_dict_roundtrip(self):
        ch = Channel(center_nm=760, fwhm_nm=25, shape='gauss',
                     name='QD-760', notes='from batch 3')
        d = ch.to_dict()
        ch2 = Channel.from_dict(d)
        assert ch.center_nm == ch2.center_nm
        assert ch.fwhm_nm == ch2.fwhm_nm
        assert ch.name == ch2.name
        assert ch.notes == ch2.notes


# ---------------------------------------------------------------------------
# ChannelSet
# ---------------------------------------------------------------------------

class TestChannelSet:

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='пустым'):
            ChannelSet(name='Empty', channels=[])

    def test_uniform_constructor(self):
        cs = ChannelSet.uniform(start_nm=750, stop_nm=790, n=15, fwhm_nm=25)
        assert len(cs) == 15
        assert cs.centers[0] == pytest.approx(750)
        assert cs.centers[-1] == pytest.approx(790)
        assert all(c.fwhm_nm == 25 for c in cs)

    def test_from_centers_scalar_fwhm(self):
        cs = ChannelSet.from_centers([750, 760, 770], fwhm_nm=20)
        assert len(cs) == 3
        assert all(c.fwhm_nm == 20 for c in cs)

    def test_from_centers_vector_fwhm(self):
        cs = ChannelSet.from_centers([750, 760, 770], fwhm_nm=[10, 20, 30])
        assert cs.fwhms.tolist() == [10, 20, 30]

    def test_from_centers_mismatched_lengths(self):
        with pytest.raises(ValueError, match='не совпадает'):
            ChannelSet.from_centers([750, 760], fwhm_nm=[10, 20, 30])

    def test_iteration_indexing(self):
        cs = ChannelSet.uniform(750, 790, 5, fwhm_nm=20)
        chs = list(cs)
        assert len(chs) == 5
        assert cs[0].center_nm == 750
        assert cs[-1].center_nm == 790

    def test_save_load_yaml_roundtrip(self):
        cs = ChannelSet.uniform(750, 790, 10, fwhm_nm=25,
                                 name='Test 10ch')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cs.yaml'
            cs.save(path)
            loaded = ChannelSet.from_file(path)
        assert len(loaded) == 10
        assert loaded.name == 'Test 10ch'
        assert loaded.centers[0] == pytest.approx(750)

    def test_load_shortcut(self):
        cs = ChannelSet.uniform(750, 790, 5, fwhm_nm=20, name='T')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'cs.yaml'
            cs.save(path)
            loaded = load_channel_set(path)
        assert loaded.name == 'T'

    def test_example_set_loads(self):
        import spectrolib
        pkg_dir = Path(spectrolib.__file__).parent
        path = pkg_dir / 'example_channel_sets' / 'baseline_15ch.yaml'
        if not path.exists():
            pytest.skip(f'{path} not found')
        cs = load_channel_set(path)
        assert len(cs) == 15
        assert all(c.shape == 'gauss' for c in cs)


# ---------------------------------------------------------------------------
# channelize
# ---------------------------------------------------------------------------

class TestChannelize:

    def test_const_spectrum_gives_const_value(self):
        """T(λ) = 1 → I_k = 1 для всех каналов."""
        spec = Spectrum.from_range(700, 800, step_nm=0.05)
        cs = ChannelSet.uniform(720, 780, 10, fwhm_nm=15)
        ch_spec = spec.to_channels(cs)
        # T_clean = 1.0 везде → ∫1·φ_k dλ = 1
        assert np.allclose(ch_spec.values_T_true, 1.0, atol=1e-3)

    def test_gauss_peak_at_channel_center(self):
        """Если в спектре есть линия поглощения в центре канала,
        соответствующий канал даёт T < 1."""
        spec = Spectrum.from_range(700, 800, step_nm=0.05)
        spec.add_gauss_peak(center_nm=760, fwhm_nm=2, amplitude=0.5)

        cs = ChannelSet.uniform(720, 780, 7, fwhm_nm=10)
        ch_spec = spec.to_channels(cs)
        # Канал с центром 760 должен иметь самое низкое T
        idx_at_760 = int(np.argmin(np.abs(cs.centers - 760)))
        assert ch_spec.values_T_true[idx_at_760] == np.min(ch_spec.values_T_true)
        # И T в этом канале < 1
        assert ch_spec.values_T_true[idx_at_760] < 0.95

    def test_returns_correct_type(self):
        spec = Spectrum.from_range(700, 800, step_nm=0.1)
        cs = ChannelSet.uniform(720, 780, 5, fwhm_nm=15)
        ch_spec = spec.to_channels(cs)
        assert isinstance(ch_spec, ChannelizedSpectrum)
        assert len(ch_spec) == 5
        assert ch_spec.values.shape == (5,)

    def test_noise_propagates(self):
        """Шум на тонкой сетке должен проявиться и в каналах."""
        spec = (Spectrum.from_range(700, 800, step_nm=0.05)
                .add_gauss_peak(760, 2, 0.5)
                .add_noise(sigma=0.05, seed=42))

        cs = ChannelSet.uniform(720, 780, 10, fwhm_nm=15)
        ch_spec = spec.to_channels(cs)

        # values_T (с шумом) и values_T_true (без) должны различаться
        assert not np.allclose(ch_spec.values_T, ch_spec.values_T_true)

    def test_out_of_range_warns_and_returns_one(self):
        spec = Spectrum.from_range(700, 800, step_nm=0.1)
        cs = ChannelSet.uniform(900, 950, 3, fwhm_nm=10)  # вне диапазона
        with pytest.warns(UserWarning, match='вне диапазона'):
            ch_spec = spec.to_channels(cs)
        # Все каналы вне диапазона → T=1
        assert np.allclose(ch_spec.values_T, 1.0)


# ---------------------------------------------------------------------------
# ChannelizedSpectrum
# ---------------------------------------------------------------------------

class TestChannelizedSpectrum:

    @pytest.fixture
    def ch_spec(self):
        spec = (Spectrum.from_range(700, 800, step_nm=0.05)
                .add_gauss_peak(760, 5, 0.3))
        cs = ChannelSet.uniform(720, 780, 10, fwhm_nm=15)
        return spec.to_channels(cs)

    def test_absorbance_property(self, ch_spec):
        a = ch_spec.absorbance
        assert a.shape == (10,)
        # Канал с поглощением имеет A > 0
        assert a.max() > 0

    def test_optical_depth_property(self, ch_spec):
        od = ch_spec.optical_depth
        assert od.shape == (10,)
        assert od.max() > 0

    def test_centers_fwhms_properties(self, ch_spec):
        assert ch_spec.centers.shape == (10,)
        assert ch_spec.fwhms.shape == (10,)
        assert all(ch_spec.fwhms == 15)

    def test_repr(self, ch_spec):
        r = repr(ch_spec)
        assert '10 каналов' in r

    def test_plot_does_not_crash(self, ch_spec):
        fig, ax = ch_spec.plot()
        plt.close(fig)

    def test_plot_with_noise_shows_compare(self):
        spec = (Spectrum.from_range(700, 800, step_nm=0.05)
                .add_gauss_peak(760, 5, 0.3)
                .add_noise(sigma=0.02, seed=42))
        cs = ChannelSet.uniform(720, 780, 10, fwhm_nm=15)
        ch_spec = spec.to_channels(cs)
        fig, ax = ch_spec.plot()
        # Compare-режим: 1 bar set + 1 line
        plt.close(fig)
