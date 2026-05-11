"""Тесты модуля spectrolib.noise."""

import numpy as np
import pytest

from spectrolib.noise import (
    NoiseModel,
    thermal_noise, shot_noise, colored_ar1_noise,
    baseline_drift, periodic_interference, spike_noise,
)


@pytest.fixture
def grid():
    return np.linspace(750, 770, 2001)


@pytest.fixture
def transmittance(grid):
    # Реалистичный T: около 1, с одной «линией поглощения»
    return np.ones_like(grid) - 0.3 * np.exp(-((grid - 760) / 0.2) ** 2)


class TestThermal:

    def test_zero_mean(self, transmittance):
        rng = np.random.default_rng(0)
        n = thermal_noise(transmittance, sigma=0.01, rng=rng)
        assert abs(n.mean()) < 0.005   # достаточно близко к 0 при N=2001

    def test_sigma_correct(self, transmittance):
        rng = np.random.default_rng(0)
        n = thermal_noise(transmittance, sigma=0.01, rng=rng)
        assert n.std() == pytest.approx(0.01, rel=0.1)

    def test_independent_of_signal(self, transmittance):
        """Тепловой шум — аддитивный, не зависит от уровня сигнала."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        n1 = thermal_noise(transmittance, sigma=0.01, rng=rng1)
        n2 = thermal_noise(2 * transmittance, sigma=0.01, rng=rng2)
        assert np.allclose(n1, n2)


class TestShot:

    def test_scales_with_sqrt_intensity(self, transmittance):
        """Дробовой шум должен иметь СКО ∝ √I."""
        rng = np.random.default_rng(0)
        n = shot_noise(transmittance, n_photons_max=1e4, rng=rng)
        # Локальное СКО в области высокого T должно быть больше,
        # чем в области низкого T (там где есть линия поглощения)
        n_high_T = n[transmittance > 0.95]
        n_low_T = n[transmittance < 0.8]
        assert n_high_T.std() > n_low_T.std()
        # Численно: σ(I) ≈ √(I/n_max) → отношение должно быть √(I_high/I_low)
        I_high = float(transmittance[transmittance > 0.95].mean())
        I_low = float(transmittance[transmittance < 0.8].mean())
        ratio_expected = np.sqrt(I_high / I_low)
        ratio_actual = n_high_T.std() / n_low_T.std()
        assert ratio_actual == pytest.approx(ratio_expected, rel=0.2)

    def test_more_photons_less_noise(self, transmittance):
        rng1 = np.random.default_rng(0)
        rng2 = np.random.default_rng(0)
        low = shot_noise(transmittance, n_photons_max=1e3, rng=rng1)
        high = shot_noise(transmittance, n_photons_max=1e6, rng=rng2)
        assert low.std() > high.std()


class TestColoredAR1:

    def test_white_when_rho_zero(self, transmittance):
        rng = np.random.default_rng(0)
        n = colored_ar1_noise(transmittance, sigma=0.01,
                               ar_coefficient=0.0, rng=rng)
        # При ρ=0 — обычный белый шум
        assert n.std() == pytest.approx(0.01, rel=0.1)
        # Автокорреляция на лаге 1 должна быть ≈ 0
        ac1 = np.corrcoef(n[:-1], n[1:])[0, 1]
        assert abs(ac1) < 0.1

    def test_correlation_matches_rho(self, transmittance):
        rng = np.random.default_rng(0)
        rho = 0.7
        n = colored_ar1_noise(transmittance, sigma=0.01,
                               ar_coefficient=rho, rng=rng)
        ac1 = np.corrcoef(n[:-1], n[1:])[0, 1]
        # Теоретическая автокорреляция на лаге 1 = ρ
        assert ac1 == pytest.approx(rho, abs=0.05)

    def test_invalid_rho_raises(self, transmittance):
        with pytest.raises(ValueError):
            colored_ar1_noise(transmittance, sigma=0.01,
                               ar_coefficient=1.5, rng=np.random.default_rng())


class TestBaselineDrift:

    def test_amplitude(self, grid):
        rng = np.random.default_rng(0)
        d = baseline_drift(grid, amplitude=0.05, n_terms=3, rng=rng)
        # После нормализации |max| == amplitude
        assert np.max(np.abs(d)) == pytest.approx(0.05, rel=1e-6)

    def test_smooth(self, grid):
        """Дрейф должен быть гладким — соседние отсчёты сильно скоррелированы."""
        rng = np.random.default_rng(0)
        d = baseline_drift(grid, amplitude=0.05, n_terms=3, rng=rng)
        ac1 = np.corrcoef(d[:-1], d[1:])[0, 1]
        assert ac1 > 0.99


class TestPeriodicInterference:

    def test_correct_period(self, grid):
        rng = np.random.default_rng(0)
        period = 1.0  # нм
        comp = [(period, 0.01, 0.0)]
        out = periodic_interference(grid, comp, rng)
        # Должна быть синусоида с этим периодом
        # Период в FFT: длина_сигнала / период_в_отсчётах
        fft = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(len(grid), d=grid[1] - grid[0])
        peak_freq = freqs[np.argmax(fft)]
        assert peak_freq == pytest.approx(1.0 / period, rel=0.05)


class TestSpikes:

    def test_rate_approximately_correct(self, transmittance):
        rng = np.random.default_rng(0)
        rate = 0.01
        out = spike_noise(transmittance, rate=rate,
                          amplitude_range=(0.1, 0.2), rng=rng)
        # Число ненулевых отсчётов / N ≈ rate
        n_spikes = (out != 0).sum()
        assert n_spikes / len(out) == pytest.approx(rate, abs=0.005)


class TestNoiseModelReproducibility:

    def test_same_seed_same_output(self, grid, transmittance):
        model = NoiseModel(
            thermal_sigma=0.005,
            shot_n_photons_max=1e4,
            colored_sigma=0.003, colored_ar=0.5,
            drift_amplitude=0.02, drift_n_terms=3,
            periodic=[(2.0, 0.005, 0.0)],
            spike_rate=0.005, spike_amplitude_range=(0.05, 0.15),
        )
        dT1, dOD1, _ = model.apply(transmittance, grid,
                                    np.random.default_rng(42))
        dT2, dOD2, _ = model.apply(transmittance, grid,
                                    np.random.default_rng(42))
        assert np.allclose(dT1, dT2)
        assert np.allclose(dOD1, dOD2)

    def test_different_seed_different_output(self, grid, transmittance):
        model = NoiseModel(thermal_sigma=0.01)
        dT1, _, _ = model.apply(transmittance, grid,
                                 np.random.default_rng(1))
        dT2, _, _ = model.apply(transmittance, grid,
                                 np.random.default_rng(2))
        assert not np.allclose(dT1, dT2)


class TestNoiseModelComposition:

    def test_disabled_when_none(self, grid, transmittance):
        """NoiseModel со всеми None — нулевой шум."""
        model = NoiseModel()
        dT, dOD, contrib = model.apply(transmittance, grid,
                                        np.random.default_rng(0))
        assert np.allclose(dT, 0)
        assert np.allclose(dOD, 0)
        assert contrib == {}

    def test_drift_goes_to_OD(self, grid, transmittance):
        """Дрейф базовой линии — в OD-пространство, не в T."""
        model = NoiseModel(drift_amplitude=0.05)
        dT, dOD, _ = model.apply(transmittance, grid,
                                  np.random.default_rng(0))
        assert np.allclose(dT, 0)
        assert not np.allclose(dOD, 0)

    def test_thermal_goes_to_T(self, grid, transmittance):
        model = NoiseModel(thermal_sigma=0.01)
        dT, dOD, _ = model.apply(transmittance, grid,
                                  np.random.default_rng(0))
        assert not np.allclose(dT, 0)
        assert np.allclose(dOD, 0)
