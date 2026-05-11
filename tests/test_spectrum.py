"""Тесты класса Spectrum (без сетевых вызовов HITRAN)."""

import numpy as np
import pytest

from spectrolib import Spectrum, GaussILS, NoiseModel


class TestCreation:

    def test_from_range_step(self):
        s = Spectrum.from_range(750, 770, step_nm=0.01)
        assert s.wavelength_nm[0] == pytest.approx(750.0)
        assert s.wavelength_nm[-1] == pytest.approx(770.0, abs=0.01)
        assert np.allclose(np.diff(s.wavelength_nm), 0.01)

    def test_from_range_n_points(self):
        s = Spectrum.from_range(750, 770, n_points=2001)
        assert len(s.wavelength_nm) == 2001
        assert s.wavelength_nm[0] == 750
        assert s.wavelength_nm[-1] == 770

    def test_from_range_either_or(self):
        with pytest.raises(ValueError):
            Spectrum.from_range(750, 770, step_nm=0.01, n_points=100)
        with pytest.raises(ValueError):
            Spectrum.from_range(750, 770)

    def test_non_monotonic_raises(self):
        wl = np.array([100, 200, 150, 300])
        with pytest.raises(ValueError):
            Spectrum(wl)


class TestCleanVsNoisy:
    """Главный инвариант: clean всегда доступна и не меняется при добавлении шума."""

    def test_no_noise_clean_equals_observed(self):
        s = (Spectrum.from_range(750, 770, step_nm=0.01)
             .add_gauss_peak(center_nm=760, fwhm_nm=0.5, amplitude=0.3))
        assert np.allclose(s.optical_depth, s.true_optical_depth)
        assert np.allclose(s.transmittance, s.true_transmittance)

    def test_noise_does_not_corrupt_clean(self):
        s = (Spectrum.from_range(750, 770, step_nm=0.01)
             .add_gauss_peak(center_nm=760, fwhm_nm=0.5, amplitude=0.3))
        clean_before = s.true_optical_depth.copy()

        s.add_noise(sigma=0.05, seed=42)

        # Истина сохранилась
        assert np.allclose(s.true_optical_depth, clean_before)
        # Наблюдение — другое
        assert not np.allclose(s.optical_depth, clean_before)

    def test_reset_noise_restores_observed(self):
        s = (Spectrum.from_range(750, 770, step_nm=0.01)
             .add_gauss_peak(center_nm=760, fwhm_nm=0.5, amplitude=0.3)
             .add_noise(sigma=0.05, seed=42))
        s.reset_noise()
        assert np.allclose(s.optical_depth, s.true_optical_depth)


class TestFluentAndCopy:

    def test_methods_return_self(self):
        s = Spectrum.from_range(750, 770, step_nm=0.01)
        assert s.add_gauss_peak(760, 0.5, 0.3) is s
        assert s.convolve_gauss(0.1) is s
        assert s.add_noise(sigma=0.01, seed=0) is s
        assert s.reset_noise() is s
        assert s.add_baseline(slope=1e-4) is s

    def test_copy_is_independent(self):
        s1 = (Spectrum.from_range(750, 770, step_nm=0.01)
              .add_gauss_peak(760, 0.5, 0.3))
        s2 = s1.copy()
        s2.add_gauss_peak(765, 0.5, 0.5)
        # Изменение копии не повлияло на оригинал
        assert s1.true_optical_depth.max() == pytest.approx(0.3)
        assert s2.true_optical_depth.max() == pytest.approx(0.5)


class TestILSConvolution:

    def test_convolve_gauss_widens_peak(self):
        s = Spectrum.from_range(750, 770, step_nm=0.001)
        s.add_gauss_peak(760, fwhm_nm=0.05, amplitude=0.3)
        peak_before = s.true_optical_depth.max()
        s.convolve_gauss(fwhm_nm=0.5)
        peak_after = s.true_optical_depth.max()
        # Свёртка с более широким ядром снижает амплитуду пика
        assert peak_after < peak_before

    def test_convolve_preserves_area(self):
        """Свёртка с нормированным ядром сохраняет ∫OD·dλ."""
        s = Spectrum.from_range(750, 770, step_nm=0.001)
        s.add_gauss_peak(760, fwhm_nm=0.5, amplitude=0.5)
        area_before = np.trapezoid(s.true_optical_depth, s.wavelength_nm)
        s.convolve_gauss(fwhm_nm=1.0)
        area_after = np.trapezoid(s.true_optical_depth, s.wavelength_nm)
        assert area_after == pytest.approx(area_before, rel=1e-3)


class TestMetadata:

    def test_history_accumulates(self):
        s = (Spectrum.from_range(750, 770, step_nm=0.01)
             .add_gauss_peak(760, 0.5, 0.3)
             .convolve_gauss(0.1)
             .add_noise(sigma=0.01, seed=42))
        # Проверяем, что все три операции в истории
        assert any('add_gauss_peak' in h for h in s.history)
        assert any('convolve_ils' in h for h in s.history)
        assert any('add_noise_model' in h for h in s.history)

    def test_metadata_complete(self):
        s = (Spectrum.from_range(750, 770, step_nm=0.01)
             .convolve_gauss(0.5)
             .add_noise_model(NoiseModel(thermal_sigma=0.01,
                                         shot_n_photons_max=1e4),
                              seed=42))
        meta = s.metadata
        assert meta['ils']['type'] == 'GaussILS'
        assert meta['noise']['thermal_sigma'] == 0.01
        assert meta['noise']['shot_n_photons_max'] == 1e4
        assert meta['noise']['seed'] == 42
        assert meta['n_points'] > 1
        assert meta['step_nm'] == pytest.approx(0.01)

    def test_seed_reproducibility(self):
        s1 = (Spectrum.from_range(750, 770, step_nm=0.01)
              .add_gauss_peak(760, 0.5, 0.3)
              .add_noise(sigma=0.01, seed=42))
        s2 = (Spectrum.from_range(750, 770, step_nm=0.01)
              .add_gauss_peak(760, 0.5, 0.3)
              .add_noise(sigma=0.01, seed=42))
        assert np.allclose(s1.optical_depth, s2.optical_depth)


class TestGaussPeak:

    def test_peak_at_correct_position(self):
        s = Spectrum.from_range(750, 770, step_nm=0.01)
        s.add_gauss_peak(center_nm=760.0, fwhm_nm=0.5, amplitude=0.3)
        idx_max = np.argmax(s.true_optical_depth)
        assert s.wavelength_nm[idx_max] == pytest.approx(760.0, abs=0.01)
        assert s.true_optical_depth[idx_max] == pytest.approx(0.3, rel=1e-3)

    def test_peak_in_transmittance_space(self):
        s = Spectrum.from_range(750, 770, step_nm=0.01)
        # Сначала линия поглощения в 760 нм
        s.add_gauss_peak(760, 0.5, 0.5, in_what='optical_depth')
        idx_center = np.argmin(np.abs(s.wavelength_nm - 760.0))
        T_center_before = s.true_transmittance[idx_center]
        # Потом «пик в пропускании» в той же точке (например, артефакт)
        s.add_gauss_peak(760, 0.3, 0.1, in_what='transmittance')
        T_center_after = s.true_transmittance[idx_center]
        # В центре линии T должно вырасти
        assert T_center_after > T_center_before


class TestRepr:
    """Проверяем что __repr__ не падает."""

    def test_repr_empty(self):
        s = Spectrum.from_range(750, 770, step_nm=0.01)
        r = repr(s)
        assert 'Spectrum' in r

    def test_repr_with_content(self):
        s = (Spectrum.from_range(750, 770, step_nm=0.01)
             .add_gauss_peak(760, 0.5, 0.3)
             .convolve_gauss(0.1)
             .add_noise(sigma=0.01, seed=0))
        r = repr(s)
        assert 'GaussILS' in r
        assert 'noise=yes' in r
