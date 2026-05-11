"""Тесты generate_averaged и snr_vs_n_realizations."""

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from spectrolib import (
    Instrument, GasMixture, NoiseModel, SpectrumGenerator,
    GaussILS, plot_snr_vs_n,
)
from spectrolib.spectrum import Spectrum


class _StructuredGenerator(SpectrumGenerator):
    """
    Тестовый генератор: вместо HITRAN использует фиктивный спектр
    с гауссовыми пиками. Нужен, чтобы тесты SNR не требовали сети
    и при этом имели нетривиальный сигнал (max-mean != 0).
    """

    def _generate_clean(self, mixture):
        spec = self.instrument.empty_spectrum()
        # Вместо HITRAN — синтетические пики
        for name, c_ppm in mixture.composition.items():
            # амплитуда пика пропорциональна c_ppm для тестов
            amp = c_ppm * 1e-4
            # центры берём из имени для разнообразия
            center = 760.0 if name == 'O2' else 763.0
            spec.add_gauss_peak(center_nm=center,
                                fwhm_nm=0.5,
                                amplitude=amp)
        if self.instrument.ils is not None:
            spec.convolve_ils(self.instrument.ils)
        spec.meta['mixture'] = {'composition': dict(mixture.composition)}
        return spec


@pytest.fixture
def setup():
    """Тестовый генератор с шумом и структурным сигналом."""
    inst = Instrument(wavelength_range=(750, 770), sampling_step=0.05)
    noise = NoiseModel(thermal_sigma=0.01)
    gen = _StructuredGenerator(instrument=inst, noise_model=noise, seed=42)
    mix = GasMixture(composition={'O2': 1000})  # → пик амплитудой 0.1
    return gen, mix


@pytest.fixture
def setup_no_signal():
    """Пустая смесь — для тестов, где сигнал не нужен."""
    inst = Instrument(wavelength_range=(750, 770), sampling_step=0.05)
    noise = NoiseModel(thermal_sigma=0.01)
    gen = _StructuredGenerator(instrument=inst, noise_model=noise, seed=42)
    mix = GasMixture(composition={})
    return gen, mix


class TestGenerateAveraged:

    def test_n_one_works(self, setup):
        gen, mix = setup
        s = gen.generate_averaged(mix, n_realizations=1)
        # Сигнал есть, шум есть → observed != true
        assert not np.allclose(s.transmittance, s.true_transmittance)

    def test_returns_spectrum_with_metadata(self, setup):
        gen, mix = setup
        s = gen.generate_averaged(mix, n_realizations=10)
        assert 'averaging' in s.meta
        assert s.meta['averaging']['n_realizations'] == 10
        assert s.meta['averaging']['domain'] == 'transmittance'
        assert any('generate_averaged' in h for h in s.history)

    def test_truth_unchanged(self, setup):
        gen, mix = setup
        s1 = gen.generate_averaged(mix, n_realizations=1)
        s10 = gen.generate_averaged(mix, n_realizations=10)
        assert np.allclose(s1.true_transmittance, s10.true_transmittance)

    def test_noise_decreases_with_n(self, setup):
        """RMS шума должна падать примерно как 1/√N."""
        gen, mix = setup
        rms = []
        for N in [1, 4, 16, 64]:
            s = gen.generate_averaged(mix, n_realizations=N)
            noise = s.transmittance - s.true_transmittance
            rms.append(float(np.sqrt(np.mean(noise ** 2))))

        assert rms[0] > rms[1] > rms[2] > rms[3]

        # rms[0]/rms[3] ≈ √64 = 8, допуск ±50% (одна реализация на N)
        ratio_observed = rms[0] / rms[3]
        ratio_expected = np.sqrt(64 / 1)
        assert 0.5 * ratio_expected < ratio_observed < 2.0 * ratio_expected

    def test_invalid_n_raises(self, setup):
        gen, mix = setup
        with pytest.raises(ValueError):
            gen.generate_averaged(mix, n_realizations=0)

    def test_invalid_domain_raises(self, setup):
        gen, mix = setup
        with pytest.raises(ValueError):
            gen.generate_averaged(mix, n_realizations=10, domain='abracadabra')

    def test_optical_depth_domain(self, setup):
        gen, mix = setup
        s = gen.generate_averaged(mix, n_realizations=100,
                                   domain='optical_depth')
        assert s.meta['averaging']['domain'] == 'optical_depth'
        # При сильном усреднении OD-шума должен быть мал
        noise = s.optical_depth - s.true_optical_depth
        assert np.sqrt(np.mean(noise ** 2)) < 0.005

    def test_no_noise_model_returns_clean(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.05)
        gen = _StructuredGenerator(instrument=inst, noise_model=None)
        mix = GasMixture(composition={'O2': 1000})
        s = gen.generate_averaged(mix, n_realizations=100)
        assert np.allclose(s.transmittance, s.true_transmittance)


class TestSNRvsN:

    def test_returns_correct_keys(self, setup):
        gen, mix = setup
        result = gen.snr_vs_n_realizations(
            mix, n_values=[1, 4, 16], n_trials=3
        )
        assert set(result.keys()) >= {
            'n_values', 'snr_mean', 'snr_std', 'rms_mean',
            'theoretical', 'domain',
        }
        assert len(result['snr_mean']) == 3

    def test_snr_grows_with_n(self, setup):
        gen, mix = setup
        result = gen.snr_vs_n_realizations(
            mix, n_values=[1, 4, 16, 64], n_trials=5,
        )
        snr = result['snr_mean']
        assert snr[0] < snr[1] < snr[2] < snr[3]

    def test_sqrt_n_law(self, setup):
        gen, mix = setup
        result = gen.snr_vs_n_realizations(
            mix, n_values=[1, 4, 16, 64], n_trials=10,
        )
        snr = result['snr_mean']
        ratio = snr[3] / snr[0]
        assert 6 < ratio < 10  # ±25% от 8

    def test_no_noise_raises(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.05)
        gen = _StructuredGenerator(instrument=inst, noise_model=None)
        mix = GasMixture(composition={'O2': 1000})
        with pytest.raises(ValueError):
            gen.snr_vs_n_realizations(mix, n_values=[1, 10])

    def test_constant_signal_raises(self, setup_no_signal):
        """SNR не определён для константного сигнала — должна быть ошибка."""
        gen, mix = setup_no_signal
        with pytest.raises(ValueError, match='константа'):
            gen.snr_vs_n_realizations(mix, n_values=[1, 10])


class TestSNRPlot:

    def test_plot_does_not_crash(self, setup):
        gen, mix = setup
        result = gen.snr_vs_n_realizations(
            mix, n_values=[1, 4, 16, 64], n_trials=3,
        )
        fig, ax = plot_snr_vs_n(result)
        assert ax.get_xlabel()
        assert ax.get_ylabel()
        plt.close(fig)

