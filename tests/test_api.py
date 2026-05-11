"""Тесты объектного фасада (без сетевых вызовов HITRAN)."""

import numpy as np
import pytest

from spectrolib import (
    Instrument, GasMixture, NoiseModel, SpectrumGenerator,
    GaussILS, Spectrum,
)


class TestInstrument:

    def test_empty_spectrum(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        s = inst.empty_spectrum()
        assert isinstance(s, Spectrum)
        assert s.wavelength_nm[0] == pytest.approx(750)
        assert s.wavelength_nm[-1] == pytest.approx(770, abs=0.01)


class TestSpectrumGeneratorWithoutHITRAN:
    """
    Тесты работают без HITRAN — используем пустую смесь и проверяем,
    что пайплайн (ILS, шум, метаданные) собирается корректно.
    """

    def test_empty_mixture(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        gen = SpectrumGenerator(inst)
        # Пустая композиция — допустима, спектр будет нулевой
        spec = gen.generate(GasMixture(composition={}))
        assert np.allclose(spec.true_optical_depth, 0)

    def test_ils_applied(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.001,
                          ils=GaussILS(fwhm=0.5))
        gen = SpectrumGenerator(inst)
        spec = gen.generate(GasMixture(composition={}))
        assert spec.ils is not None
        assert spec.ils['type'] == 'GaussILS'

    def test_noise_applied(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        noise = NoiseModel(thermal_sigma=0.01)
        gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
        spec = gen.generate(GasMixture(composition={}))
        assert spec.noise is not None
        # Истина — нули, наблюдаемое — с шумом
        assert np.allclose(spec.true_optical_depth, 0)
        assert not np.allclose(spec.optical_depth, 0)

    def test_seed_increment_between_calls(self):
        """Два последовательных вызова generate с одним seed должны давать
        разные шумовые реализации (внутренний счётчик увеличивается)."""
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01)
        noise = NoiseModel(thermal_sigma=0.01)
        gen = SpectrumGenerator(inst, noise_model=noise, seed=42)
        s1 = gen.generate(GasMixture(composition={}))
        s2 = gen.generate(GasMixture(composition={}))
        # Разные реализации шума
        assert not np.allclose(s1.optical_depth, s2.optical_depth)

    def test_meta_populated(self):
        inst = Instrument(wavelength_range=(750, 770), sampling_step=0.01,
                          ils=GaussILS(fwhm=0.5))
        gen = SpectrumGenerator(inst, seed=7)
        mix = GasMixture(composition={}, T_K=310, p_atm=1.0, L_cm=10.0)
        spec = gen.generate(mix)
        assert 'instrument' in spec.meta
        assert 'mixture' in spec.meta
        assert spec.meta['mixture']['T_K'] == 310
        assert spec.meta['generator_seed'] == 7
